#!/usr/bin/env python3
"""Slurm-only multi-view renderer using one strictly verified checkpoint rollout."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).parent))
from export_videos import sha256


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def layout_shape(count, layout):
    if layout == 'horizontal':
        return 1, count
    if layout == 'vertical':
        return count, 1
    return (1, count) if count <= 2 else (2, 2)


def audio_filters(start, end, speed, tail=0.):
    filters = [f'atrim=start={start:.12g}:end={end+tail*speed:.12g}', 'asetpts=PTS-STARTPTS']
    while speed < .5:
        filters.append('atempo=0.5'); speed /= .5
    while speed > 2:
        filters.append('atempo=2'); speed /= 2
    filters.extend((f'atempo={speed:.12g}', 'apad'))
    return ','.join(filters)


def load_trace(source):
    import numpy as np
    from replay_arm_ik import read_trace
    trace = read_trace(source)
    report = json.loads((source/'report.json').read_text())
    prefix = float(report.get('preparation_prefix_seconds', 0))
    if prefix:
        prep = np.load(source/'preparation.npz', allow_pickle=False)
        if not np.isclose(prep['time'][-1], prefix) or not np.allclose(np.diff(prep['time']), trace['physics_dt']):
            raise ValueError('Fixed preparation prefix has incompatible timing')
        for name in ('qpos', 'mocap_pos', 'mocap_quat'):
            if not np.allclose(prep[name][-1], trace[name][0], atol=1e-5):
                raise ValueError('Fixed preparation does not join the recorded initial state: '+name)
            trace[name] = np.concatenate((prep[name][:-1], trace[name]))
        trace['time'] = np.concatenate((prep['time'][:-1], trace['time']+prefix))
    if not all(np.isfinite(trace[k]).all() for k in ('time', 'qpos', 'mocap_pos', 'mocap_quat')):
        raise ValueError('Nonfinite physics trace')
    if len(trace['time']) < 3 or np.any(np.diff(trace['time']) <= 0):
        raise ValueError('Incomplete or unordered trace')
    return trace, report


def sample_interval(time, at):
    import numpy as np
    index = int(np.clip(np.searchsorted(time, at, side='right')-1, 0, len(time)-2))
    fraction = float(np.clip((at-time[index])/(time[index+1]-time[index]), 0, 1))
    return index, fraction


def interpolate(values, time, at):
    index, fraction = sample_interval(time, at)
    return (1-fraction)*values[index] + fraction*values[index+1]


def crop_times(time, start, end):
    import numpy as np
    return np.concatenate(([start], time[(time > start+1e-10) & (time < end-1e-10)], [end]))


def set_state(model, data, trace, at):
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    import numpy as np
    index, f = sample_interval(trace['time'], at)
    data.qpos[:] = (1-f)*trace['qpos'][index]+f*trace['qpos'][index+1]
    count = trace['mocap_pos'].shape[1]
    data.mocap_pos[:count] = (1-f)*trace['mocap_pos'][index]+f*trace['mocap_pos'][index+1]
    for j in range(count):
        quats = trace['mocap_quat'][index:index+2, j]
        if np.allclose(quats[0], quats[1], atol=1e-12):
            data.mocap_quat[j] = quats[0]
        else:
            rotation = Slerp([0, 1], Rotation.from_quat(np.roll(quats, -1, axis=1)))([f])
            data.mocap_quat[j] = np.roll(rotation.as_quat()[0], 1)
    mujoco.mj_forward(model, data)


def visual_corners(model, data, *, piano_only=False):
    """World bounds from compiled piano/robot visuals, excluding floor/contacts."""
    import mujoco
    import numpy as np
    from itertools import product
    signs = np.array(list(product((-1., 1.), repeat=3)))
    points = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        body = model.body(model.geom_bodyid[gid]).name
        piano = name.startswith('piano/')
        robot = body.startswith(('right_/', 'left_/', 'tj_', 'hand_connector_'))
        if not (piano or not piano_only and robot) or model.geom_group[gid] >= 2:
            continue
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        local = model.geom_aabb[gid, :3] + signs*model.geom_aabb[gid, 3:]
        points.append(local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    if not points:
        raise ValueError('No visual geometry available for automatic camera framing')
    return np.concatenate(points)


def piano_corners(model):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return visual_corners(model, data, piano_only=True)


def camera_for(view, model, arms, custom, width=1280, height=720, piano_fill=1.,
               framing_points=None, caption_height=0, footer_height=0):
    import mujoco
    import numpy as np
    camera = mujoco.MjvCamera()
    # World X runs across the depth of the keyboard; Y follows the keyboard.
    angles = {'front': (0, -25), 'back': (180, -30), 'left': (90, -25),
              'right': (270, -25), 'top': (180, -90), 'overview': (140, -28), 'hands': (0, -45)}
    camera.azimuth, camera.elevation = angles[view]
    # In orthographic mode distance controls the camera's position, not zoom.
    camera.distance = 2.5
    camera.orthographic = 1
    for key, value in custom.items():
        setattr(camera, key, value)
    az, el = np.deg2rad([camera.azimuth, camera.elevation])
    right = np.array([np.sin(az), -np.cos(az), 0.])
    up = np.array([-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)])
    corners = piano_corners(model)
    horizontal, vertical = corners @ right, corners @ up
    piano_width = float(np.ptp(horizontal))
    if piano_width <= 0 or not 0 < piano_fill <= 1:
        raise ValueError('Invalid piano projection or fill ratio')
    ortho_height = piano_width / piano_fill * height / width
    center = (corners.min(0)+corners.max(0))/2
    center += right*((horizontal.min()+horizontal.max())/2-center @ right)
    content_vertical = vertical if framing_points is None else framing_points @ up
    caption_m = ortho_height*caption_height/height
    footer_m = ortho_height*footer_height/height
    wanted = (content_vertical.min()+content_vertical.max()+caption_m-footer_m)/2
    # Center the sampled performance vertically, while retaining the entire
    # keyboard whenever the chosen aspect ratio permits. Never zoom out to fit arms.
    lower = vertical.max()-ortho_height/2+caption_m
    upper = vertical.min()+ortho_height/2-footer_m
    vertical_center = float(np.clip(wanted, lower, upper)) if lower <= upper else float((vertical.min()+vertical.max())/2)
    center += up*(vertical_center-center @ up)
    camera.lookat[:] = center
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = ortho_height
    return camera, dict(view=view, azimuth=camera.azimuth, elevation=camera.elevation,
                        distance=camera.distance, lookat=camera.lookat.tolist(), projection='orthographic',
                        ortho_height_m=ortho_height, piano_projected_width_m=piano_width,
                        piano_width_fraction=piano_fill, framing='piano_width',
                        sampled_content_vertical_span_m=float(np.ptp(content_vertical)),
                        vertical_content_cropped=bool(content_vertical.min() < vertical_center-ortho_height/2+footer_m
                            or content_vertical.max() > vertical_center+ortho_height/2-caption_m),
                        caption_height=caption_height, footer_height=footer_height,
                        panel_width=width, panel_height=height)


def set_render_state(model, data, trace, at, overlay=None, q=None):
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tsensei_v3_2_pianist.preview_connector import offset_transform
    set_state(model,data,trace,at)
    if overlay is None:
        return
    arms=overlay['arms']
    path = overlay.get('arm_path')
    arms.data.qpos[:] = path(at) if path is not None else interpolate(q,trace['time'],at)
    mujoco.mj_forward(arms.model,arms.data)
    for mid,bid in overlay['arm_mapping']:
        data.mocap_pos[mid]=arms.data.xpos[bid];data.mocap_quat[mid]=arms.data.xquat[bid]
    for side,mid in overlay['connector_mapping'].items():
        bid=model.body(side+'_/base_link').id;h=np.eye(4)
        h[:3,:3]=data.xmat[bid].reshape(3,3);h[:3,3]=data.xpos[bid]
        h=h @ offset_transform(overlay['connector'].config['offsets'][side])
        data.mocap_pos[mid]=h[:3,3];data.mocap_quat[mid]=np.roll(Rotation.from_matrix(h[:3,:3]).as_quat(),1)
    mujoco.mj_forward(model,data)


def record_once(plan, output):
    if plan['recording']:
        source = Path(plan['recording'])
        for name, expected in plan['recording_sha256'].items():
            if sha256(source/name) != expected:
                raise ValueError('Reused recording changed since submission: '+name)
        return source
    archive = Path(plan['source_archive'])
    if sha256(archive) != plan['archive_sha256']:
        raise ValueError('Source archive changed since submission')
    source = output/'recording'
    command = [str(ROOT/'training_mjlab/.venv/bin/python'), str(ROOT/'scripts/eval_from_source_archive.py'),
               '--source-archive', str(archive), '--checkpoint', plan['checkpoint'], '--config', plan['config'],
               '--mode', 'render', '--capture-physics', '--output', str(source)]
    if not plan['audio']:
        command.append('--no-audio')
    subprocess.run(command, check=True)
    return source


def recover_fingering(plan, source, directory, duration):
    import numpy as np
    if sha256(plan['source_archive']) != plan['archive_sha256']:
        raise ValueError('Source archive changed since submission')
    subprocess.run([str(ROOT/'training_mjlab/.venv/bin/python'), str(ROOT/'scripts/export_score_annotations.py'),
                    '--source-archive', plan['source_archive'], '--config', plan['config'],
                    '--recording', str(source), '--output', str(directory)], check=True)
    metadata = json.loads((directory/'score_annotations.json').read_text())
    if (metadata['checkpoint_sha256'] != plan['checkpoint_sha256'] or
            metadata['annotations_sha256'] != sha256(directory/'score_annotations.npz')):
        raise ValueError('Recovered fingering checksum mismatch')
    with np.load(directory/'score_annotations.npz', allow_pickle=False) as arrays:
        score = {key: arrays[key].copy() for key in arrays.files}
    if not np.isclose(score['time'][-1], duration, rtol=0, atol=1e-7):
        raise ValueError('Fingering and physics recording durations differ')
    score['metadata'] = metadata
    return score


def fingering_at(score, at):
    """Display the target applied during this original-time control interval.

    Exact endpoints show the scored target, matching the archived renderer.
    Fixed preparation and the initial state have no applied score target.
    """
    import numpy as np
    if at <= score['metadata']['preparation_prefix_seconds']+1e-9:
        return np.zeros((10, 88), dtype=score['fingers'].dtype)
    index = int(np.searchsorted(score['time'], at-1e-9, side='left'))
    return score['fingers'][min(index, len(score['time'])-1)]


def draw_fingering(picture, assigned, footer_height, key_highlight):
    """Keep explicit R/L assignments readable even when 3D labels are hidden."""
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from tsensei_v3_2_pianist.preview_fingering import FINGER_COLORS, FINGER_NAMES, finger_label, note_name
    draw = ImageDraw.Draw(picture)
    width, height = picture.size
    top = height-footer_height
    draw.rectangle((0, top, width, height), fill=(18, 25, 36))
    colors = [tuple(int(255*x) for x in .35+.65*c[:3]) for c in FINGER_COLORS]
    rows = []
    for side, first in (('Right', 0), ('Left', 5)):
        spans = [(f'Score {side}:  ', '#c9d4e3')]
        spans += [(f'{finger_label(f)}:{note_name(k)}  ', colors[f % 5])
                  for f in range(first, first+5) for k in np.flatnonzero(assigned[f] > 0)]
        if len(spans) == 1:
            spans.append(('rest', '#c9d4e3'))
        rows.append(spans)
    rows.append([(f'{n+1} {name}   ', colors[n]) for n, name in enumerate(FINGER_NAMES)])
    if key_highlight:
        rows[-1].append(('| green = pressed', '#6cdd6c'))
    size = max(11, min(20, width//38))
    for row, spans in enumerate(rows):
        font_size = size
        while True:
            font = ImageFont.load_default(size=font_size)
            if sum(draw.textlength(text, font=font) for text, _ in spans) <= width-20 or font_size <= 6:
                break
            font_size -= 1
        x = 10
        for text, color in spans:
            draw.text((x, top+4+row*(size+8)), text, font=font, fill=color)
            x += draw.textlength(text, font=font)


def csv(path, time, values, names):
    import numpy as np
    np.savetxt(path, np.column_stack((time, values)), delimiter=',',
               header=','.join(['time_s', *names]), comments='', fmt='%.12g')


def audit_arm_path(path, arms, model, trace, report, folder, subdivisions=4):
    """Independently compare FK against the exact hand interpolation used by video."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    time = trace['time']
    audit_time = np.r_[(time[:-1, None]+np.diff(time)[:, None]*
                       np.arange(subdivisions)[None, :]/subdivisions).ravel(), time[-1]]
    data = mujoco.MjData(model)
    errors = np.empty((len(audit_time), 2, 2))
    body_ids = [model.body(side+'_/base_link').id for side in ('right', 'left')]
    for k, at in enumerate(audit_time):
        set_state(model, data, trace, at)
        arms.data.qpos[:] = path(at)
        mujoco.mj_forward(arms.model, arms.data)
        for i, side in enumerate(('right', 'left')):
            sid = arms.sides[side]['sid']; bid = body_ids[i]
            errors[k, i, 0] = np.linalg.norm(arms.data.site_xpos[sid]-data.xpos[bid])
            errors[k, i, 1] = np.linalg.norm(Rotation.from_matrix(
                data.xmat[bid].reshape(3, 3) @ arms.data.site_xmat[sid].reshape(3, 3).T).as_rotvec())
    maximum = errors.max(axis=0)
    passed = bool(np.isfinite(errors).all() and (maximum[:, 0] <= report['position_tolerance_m']).all()
                  and (maximum[:, 1] <= report['rotation_tolerance_rad']).all())
    audit = dict(passed=passed, samples=len(audit_time), subdivisions_per_physics_step=subdivisions,
                 maximum_error_m_rad=maximum.tolist(), rms_error_m_rad=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                 peak_time_s=[[float(audit_time[np.argmax(errors[:, i, j])]) for j in range(2)] for i in range(2)],
                 initial_error_m_rad=errors[0].tolist(), terminal_error_m_rad=errors[-1].tolist(),
                 method='Sampled FK audit of the exact renderer hand and arm interpolation; not a continuous pose-error proof')
    report['between_frame_tracking'] = audit
    csv(folder/'arm_tracking_errors.csv', audit_time, errors.reshape(len(audit_time), 4),
        ['right_position_m', 'right_rotation_rad', 'left_position_m', 'left_rotation_rad'])
    save_json(folder/'arm_tracking_audit.json', audit)
    if not passed:
        raise ValueError(f'Between-frame arm tracking failed: {maximum.tolist()}; tempo and tolerances were not relaxed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for j, scale, tolerance, unit in ((0, 1000, report['position_tolerance_m']*1000, 'mm'),
                                     (1, 180/np.pi, np.rad2deg(report['rotation_tolerance_rad']), 'deg')):
        for i, side in enumerate(('right', 'left')):
            axes[j].plot(audit_time, errors[:, i, j]*scale, label=side, lw=.7)
        axes[j].axhline(tolerance, color='#b32626', ls='--', label='acceptance limit')
        axes[j].set_ylabel(('Position' if j == 0 else 'Rotation')+' error ('+unit+')')
        axes[j].legend(); axes[j].grid(alpha=.2)
    axes[1].set_xlabel('Original replay time (s)')
    fig.suptitle('Absolute wrist target errors | original tempo | no integrated drift')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'arm_tracking_errors.{ext}', dpi=160)
    plt.close(fig)
    return audit


def hand_joint_ids(model, side):
    import mujoco
    kinds = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    # Cast pybind enums explicitly: numpy 2.5 scalar == enum may return False.
    return [j for j in range(model.njnt) if model.joint(j).name.startswith(side+'_/')
            and int(model.jnt_type[j]) in kinds]


def hand_motion_columns(model, poses):
    import mujoco
    import numpy as np
    columns, joints = [], []
    for side in ('right', 'left'):
        ids = hand_joint_ids(model, side)
        if not ids:
            raise ValueError('No scalar hand/wrist joints found: '+side)
        for jid in ids:
            columns.append(poses[:, model.jnt_qposadr[jid]])
            joints.append(dict(name=model.joint(jid).name, side=side,
                               position_min=float(model.jnt_range[jid, 0]) if model.jnt_limited[jid] else None,
                               position_max=float(model.jnt_range[jid, 1]) if model.jnt_limited[jid] else None,
                               unit='m' if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else 'rad'))
    return np.column_stack(columns), joints


def export_plots(plan, output, model, trace, positions, rotations, q, arms, arm_report, time, arm_path=None):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from joint_motion_plots import export_joint_motion
    from arm_motion_limits import load_m6_joint_limits
    folder = output/'plots'; folder.mkdir()
    summaries = {}
    sample = lambda values: np.stack([interpolate(values, trace['time'], t) for t in time])
    if 'arm-joints' in plan['plots']:
        columns = np.concatenate([arms.sides[s]['qadr'] for s in ('right', 'left')])
        joints = [dict(name=f'{s}_J{j}', side=s, unit='rad') for s in ('right', 'left') for j in range(1, 8)]
        for item, bounds in zip(joints, np.concatenate([arms.sides[s]['limits'] for s in ('right', 'left')])):
            item.update(position_min=float(bounds[0]), position_max=float(bounds[1]))
        summaries['arm_joints'] = export_joint_motion(
            folder/'arm_joints', time, (arm_path(time) if arm_path is not None else sample(q))[:, columns],
            joints, 'M6 arm', limits=load_m6_joint_limits(),
            continuous_path=arm_path)
    if 'tcp' in plan['plots']:
        directory = folder/'tcp'; directory.mkdir()
        # Compute FK at the exact cropped endpoint states instead of interpolating matrices.
        import mujoco
        data = mujoco.MjData(model)
        ids = [model.body(s+'_/base_link').id for s in ('right', 'left')]
        p, r = [], []
        for at in time:
            set_state(model, data, trace, at)
            p.append(data.xpos[ids].copy()); r.append(data.xmat[ids].reshape(2,3,3).copy())
        p, r = np.asarray(p), np.asarray(r)
        dt = np.diff(time); mid = (time[1:]+time[:-1])/2
        linear = np.diff(p, axis=0)/dt[:,None,None]
        angular = np.stack([Rotation.from_matrix(r[1:,i] @ r[:-1,i].transpose(0,2,1)).as_rotvec()/dt[:,None]
                            for i in range(2)], axis=1)
        columns, names = [], []
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
        for i, side in enumerate(('right', 'left')):
            for row, (values, label, unit) in enumerate(((linear, 'linear', 'm_s'), (angular, 'angular', 'rad_s'))):
                axes[row,i].plot(mid, np.linalg.norm(values[:,i],axis=1), label='norm', color='black')
                for j, axis in enumerate('xyz'):
                    axes[row,i].plot(mid, values[:,i,j], label=axis, alpha=.6, lw=.7)
                    columns.append(values[:,i,j]); names.append(f'{side}_{label}_{axis}_{unit}')
                columns.append(np.linalg.norm(values[:,i],axis=1)); names.append(f'{side}_{label}_norm_{unit}')
                axes[row,i].set_title(f'{side} wrist TCP {label} velocity ({unit.replace("_", "/")})')
                axes[row,i].grid(alpha=.2); axes[row,i].legend()
                axes[row,i].set_xlabel('Original replay time (s)')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(directory/f'wrist_velocities.{ext}', dpi=160)
        plt.close(fig)
        csv(directory/'wrist_velocities.csv', mid, np.column_stack(columns), names)
    if 'hand-joints' in plan['plots']:
        poses, joints = hand_motion_columns(model, sample(trace['qpos']))
        summaries['hand_joints'] = export_joint_motion(folder/'hand_joints', time, poses, joints, 'hand / wrist')
    return summaries


def render_video(plan, spec, output, source, trace, raw_model, overlay, q, start, end, fps, score=None):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight
    from tsensei_v3_2_pianist.preview_connector import offset_transform
    model = overlay['model'] if spec['arms'] else raw_model
    data = mujoco.MjData(model)
    width, height = plan['width'], plan['height']
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    active_overlay = overlay if spec['arms'] else None
    framing_points=[]
    for at in np.linspace(start,end,21):
        set_render_state(model,data,trace,at,active_overlay,q)
        framing_points.append(visual_corners(model,data))
    framing_points=np.concatenate(framing_points)
    caption_height=2*max(11,min(20,width//38))+14 if plan['overlay'] else 0
    footer_height=3*(max(11,min(20,width//38))+8)+8 if score is not None else 0
    cameras = [camera_for(v, model, spec['arms'], spec['camera'], width, height, plan.get('piano_fill', 1.),
                          framing_points, caption_height, footer_height)
               for v in spec['views']]
    rows, cols = layout_shape(len(cameras),spec['layout'])
    total_width, total_height = width*cols, height*rows
    directory=output/'videos'; directory.mkdir(exist_ok=True)
    final=directory/(spec['name']+'.mp4'); silent=directory/(spec['name']+'.silent.mp4')
    ffmpeg=shutil.which('ffmpeg')
    count=int(np.ceil(((end-start)/plan['speed']+plan['hold'])*fps))+1
    encoder=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{total_width}x{total_height}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-threads','4','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)],stdin=subprocess.PIPE)
    option=mujoco.MjvOption();option.geomgroup[2:5]=0;option.sitegroup[:]=0
    highlight=ActiveKeyHighlight(model)
    fingering=FingeringHighlight(model,score['metadata'],highlight) if score is not None else None
    try:
        with mujoco.Renderer(model,height=height,width=width,max_geom=4000) as renderer:
            for frame in range(count):
                at=min(start+frame/fps*plan['speed'],end)
                set_render_state(model,data,trace,at,active_overlay,q)
                assigned=fingering_at(score,at) if fingering is not None else None
                canvas=Image.new('RGB',(total_width,total_height),(15,20,28))
                for n,(camera,settings) in enumerate(cameras):
                    # Free-camera projection is a model global: restore each
                    # panel's scale here, rather than reusing the last camera's.
                    model.vis.global_.orthographic = 1
                    model.vis.global_.fovy = settings['ortho_height_m']
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    if plan['key_highlight']:highlight.color_render_scene(renderer.scene,data)
                    if fingering is not None:
                        first_annotation=renderer.scene.ngeom
                        fingering.append(renderer.scene,data,assigned)
                        if not plan.get('finger_labels',False):
                            for geom in renderer.scene.geoms[first_annotation:renderer.scene.ngeom]:
                                geom.label=''
                    picture=Image.fromarray(renderer.render())
                    if plan['overlay']:
                        draw=ImageDraw.Draw(picture);font_size=max(11,min(20,width//38))
                        draw.rectangle((0,0,width,2*font_size+14),fill=(18,25,36))
                        draw.text((10,4),f'{settings["view"]} | {"kinematic arms" if spec["arms"] else "hands"} | t={at:.3f}s | {plan["speed"]:g}x',fill='white',font_size=font_size)
                        draw.text((10,font_size+8),Path(plan['checkpoint_origin']).parent.name+'/'+Path(plan['checkpoint_origin']).name,fill='#c9d4e3',font_size=max(10,font_size-3))
                    if fingering is not None:
                        draw_fingering(picture,assigned,footer_height,plan['key_highlight'])
                    canvas.paste(picture,((n%cols)*width,(n//cols)*height))
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if frame == min(count-1,round(fps)):
                    canvas.save(directory/(spec['name']+'.png'))
    finally:
        encoder.stdin.close(); code=encoder.wait()
    if code:raise RuntimeError(f'ffmpeg encoder failed for {spec["name"]}: {code}')
    if plan['audio']:
        filters=audio_filters(start,end,plan['speed'],plan['hold'] if np.isclose(end,trace['time'][-1]) else 0)
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-i',str(silent),'-i',str(source/'played.wav'),
            '-map','0:v:0','-map','1:a:0','-af',filters,'-t',str(count/fps),'-c:v','copy','-c:a','aac','-b:a','192k',
            '-movflags','+faststart',str(final)],check=True)
        silent.unlink()
    else:silent.rename(final)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-xerror','-i',str(final),'-f','null','-'],check=True)
    return dict(file=str(final.relative_to(output)),preview=str(final.with_suffix('.png').relative_to(output)),
                frames=count,fps=fps,duration=count/fps,width=total_width,height=total_height,
                cameras=[c[1] for c in cameras],arms=spec['arms'],audio=plan['audio'],
                fingering=score is not None,finger_labels=score is not None and plan.get('finger_labels',False),
                decoded=True,sha256=sha256(final))


def run(plan):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU export must run through export_video.sh / scripts/hpc/submit.sh export')
    import mujoco
    import numpy as np
    from replay_arm_ik import read_targets, overlay_models, visual_scene
    from tsensei_v3_2_pianist.preview_trajectory import differential_track
    output=Path(plan['output'])
    def status(stage, **extra):
        save_json(output/'status.json',dict(state='running',stage=stage,job_id=os.environ['SLURM_JOB_ID'],**extra))
        print(stage,flush=True)
    for key in ('checkpoint','config'):
        if sha256(plan[key]) != plan[key+'_sha256']:raise ValueError('Frozen input mismatch: '+key)
    if not shutil.which('ffmpeg'):raise FileNotFoundError('ffmpeg is required')
    # STL/XML/recorded render scenes are internal job inputs, not deliverables.
    job_directory=Path(plan['source_project'])/'logs/hpc'/os.environ['SLURM_JOB_ID']
    if os.environ.get('TSENSEI_ALLOCATION_STEP') == '1':
        job_directory=job_directory/'steps'/os.environ['SLURM_STEP_ID']
    workspace=job_directory/'export_work'
    workspace.mkdir(parents=True, exist_ok=False)
    status('recording')
    source=record_once(plan,workspace);trace,record_report=load_trace(source)
    if record_report['checkpoint_sha256'] != plan['checkpoint_sha256']:raise ValueError('Checkpoint/recording SHA mismatch')
    duration=float(trace['time'][-1]);start=plan['start'];end=duration if plan['end'] is None else plan['end']
    if not 0 <= start < end <= duration+1e-8:raise ValueError(f'Clip must be within the full trajectory [0,{duration}] seconds')
    end=min(end,duration);fps=plan['fps'] or float(record_report['fps'])
    raw_model=mujoco.MjModel.from_xml_path(str(source/'kinematic_scene/scene.xml'))
    if raw_model.nq != raw_model.nv or raw_model.nq != trace['qpos'].shape[1]:
        raise ValueError('Exporter requires the scalar hinge/slide hand model from this project')
    if any((raw_model.body(i).name or '').startswith('tj_') for i in range(raw_model.nbody)):
        raise ValueError('Checkpoint already includes an arm model; noarms/IK overlay requires a hand-only checkpoint')
    status('extracting wrist poses')
    positions,rotations=read_targets(source,trace)
    data_dir=output/'data';data_dir.mkdir()
    score=None
    if plan.get('fingering',False):
        status('recovering verified score fingering')
        score=recover_fingering(plan,source,data_dir,duration)
    np.savez_compressed(data_dir/'wrist_poses.npz',time=trace['time'],position=positions,rotation=rotations)
    from scipy.spatial.transform import Rotation
    values=[];names=[]
    for i,side in enumerate(('right','left')):
        values.extend((positions[:,i],Rotation.from_matrix(rotations[:,i]).as_quat()))
        names.extend(f'{side}_{axis}' for axis in ('x_m','y_m','z_m','qx','qy','qz','qw'))
    csv(data_dir/'wrist_poses.csv',trace['time'],np.concatenate(values,axis=1),names)
    overlay=None;q=None;arms=None;arm_report=None;arm_path=None
    if any(v['arms'] for v in plan['videos']) or 'arm-joints' in plan['plots']:
        status('differential IK')
        root,connector,arms=overlay_models(plan['connector_config'],plan['arm_config'])
        q,errors=differential_track(arms,positions,rotations)
        motion_report = None
        if plan.get('arm_ik', 'raw') == 'constrained':
            if plan['speed'] != 1 and any(v['arms'] for v in plan['videos']):
                raise ValueError('Constrained arm videos require original tempo (--speed 1)')
            from constrained_arm_motion import solve_fixed_time, save_path
            from arm_motion_limits import load_m6_joint_limits
            status('fixed-time arm motion optimization')
            np.savez_compressed(data_dir/'arm_trajectory_raw.npz', time=trace['time'], q=q, ik_errors=errors)
            arm_path,q,errors,motion_report=solve_fixed_time(
                arms,trace['time'],q,positions,rotations,load_m6_joint_limits(),
                position_tolerance_m=plan['arm_position_tolerance_mm']/1000,
                rotation_tolerance_rad=np.deg2rad(plan['arm_rotation_tolerance_deg']))
            status('checking arm tracking between physics frames')
            audit_arm_path(arm_path,arms,raw_model,trace,motion_report,data_dir)
            save_path(data_dir/'arm_spline.npz',arm_path,motion_report)
        arm_report=dict(label=Path(plan['checkpoint_origin']).name,recorded_physics_hz=1/float(trace['physics_dt']),
            maximum_ik_error_m_rad=errors.max(axis=(0,1)).tolist(),connector=connector.config,arm_mount=arms.config,
            velocity_definition='analytic B-spline derivative' if arm_path is not None else 'unfiltered delta q / delta original time',
            limits_enforced='continuous joint angles, SDK velocity and acceleration' if arm_path is not None else 'joint angles only',
            motion_plan=motion_report,
            limitations=['Kinematic overlay; no arm dynamics/collision forces/hardware validation.',
                         'Provisional shoulder/flange calibration; right hand is derived by mirroring.'])
        velocity_time=(trace['time'][:-1]+trace['time'][1:])/2
        velocity=arm_path(velocity_time,1) if arm_path is not None else np.diff(q,axis=0)/np.diff(trace['time'])[:,None]
        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise ValueError('Nonfinite arm IK trajectory')
        arm_report['maximum_joint_speed_rad_s']={side:np.abs(velocity[:,a['dof']]).max(0).tolist() for side,a in arms.sides.items()}
        arm_report['joint_diagnostics']={}
        for side,arm in arms.sides.items():
            angles=q[:,arm['qadr']]
            margin=np.minimum(angles-arm['limits'][:,0],arm['limits'][:,1]-angles)
            if margin.min() < -1e-8:raise ValueError('IK exceeded the arm angle range: '+side)
            arm_report['joint_diagnostics'][side]=dict(
                urdf_speed_limits_rad_s=arm['velocity'].tolist(),
                velocity_over_limit_intervals=(np.abs(velocity[:,arm['dof']])>arm['velocity']+1e-8).sum(0).tolist(),
                minimum_angle_limit_margin_rad=margin.min(0).tolist(),
                samples_at_angle_limit=(margin<1e-8).sum(0).tolist())
        np.savez_compressed(data_dir/'arm_trajectory.npz',time=trace['time'],q=q,ik_errors=errors)
        csv(data_dir/'arm_joint_angles.csv',trace['time'],q,[f'{s}_J{j}_rad' for s in ('right','left') for j in range(1,8)])
        csv(data_dir/'arm_joint_velocities.csv',velocity_time,velocity,
            [f'{s}_J{j}_rad_s' for s in ('right','left') for j in range(1,8)])
        save_json(data_dir/'arm_report.json',arm_report)
        if any(v['arms'] for v in plan['videos']):
            model=visual_scene(source,workspace,root)
            if model.nq != raw_model.nq:raise ValueError('Overlay changed physical coordinates')
            overlay=dict(model=model,arms=arms,connector=connector,arm_path=arm_path,
                arm_mapping=[(int(model.body_mocapid[model.body(v.full_identifier).id]),arms.model.body(name).id) for v,name in arms.visuals],
                connector_mapping={s:int(model.body_mocapid[model.body(b.full_identifier).id]) for s,b in connector.bodies.items()})
    plot_reports = {}
    if plan['plots']:
        status('plotting')
        plot_reports = export_plots(plan,output,raw_model,trace,positions,rotations,q,arms,arm_report,
                                    crop_times(trace['time'],start,end),arm_path=arm_path)
    videos=[]
    for spec in plan['videos']:
        status('rendering '+spec['name'])
        videos.append(render_video(plan,spec,output,source,trace,raw_model,overlay,q,start,end,fps,score))
    result=dict(schema=1,checkpoint_origin=plan['checkpoint_origin'],checkpoint_sha256=plan['checkpoint_sha256'],
                recording=str(source),internal_workspace=str(workspace),
                physics_trace_sha256=sha256(source/'physics_trace.npz'),
                physics_hz=1/float(trace['physics_dt']),recorded_duration=duration,
                clip_start=start,clip_end=end,speed=plan['speed'],plots_time_basis='original time within clip',
                full_episode_phase_metrics=record_report.get('phase_metrics'),
                metrics_scope='full recorded episode, not recomputed for a cropped clip',
                fingering_annotations=score['metadata'] if score is not None else None,
                videos=videos,arm_analysis=arm_report,joint_motion=plot_reports,source_job=os.environ['SLURM_JOB_ID'],
                source_step=os.environ.get('SLURM_STEP_ID'))
    save_json(output/'report.json',result)
    blocks=['<!doctype html><meta charset="utf-8"><title>Video export</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;background:#121923;color:#eef}video{width:100%;max-height:700px}a{color:#9cf}img{max-width:100%}</style>',
        '<h1>Checkpoint video export</h1><p>'+html.escape(plan['checkpoint_origin'])+'</p>',
        '<p><a href="report.json">Report</a> · <a href="export_plan.json">Parameters</a> · <a href="data/">CSV / NPZ</a></p>']
    for video in videos:
        file=html.escape(video['file']);blocks.append(f'<h2>{file}</h2><video controls preload="metadata" src="{file}"></video>')
    if 'arm_joints' in plot_reports:
        from arm_motion_limits import limits_table_html
        blocks.append(limits_table_html(plot_reports['arm_joints']['reference_limits'], 'plots/arm_joints'))
        for side in ('right', 'left'):
            for kind in ('velocities', 'accelerations'):
                rel=f'plots/arm_joints/{side}_joint_{kind}.png'
                blocks.append(f'<p><a href="{rel}"><img loading="lazy" src="{rel}"></a></p>')
    if arm_path is not None:
        audit=arm_report['motion_plan']['between_frame_tracking']
        maximum=np.array(audit['maximum_error_m_rad'])
        blocks.append('<h2>原节奏限速轨迹</h2><p>两臂连续轨迹的关节速度、加速度和角范围验证通过。'
                      f'帧间位置误差最大 {maximum[:,0].max()*1000:.4f} mm；'
                      f'姿态误差最大 {np.rad2deg(maximum[:,1].max()):.4f}°。'
                      '运动学预览，未作真机动力学或避障验证。</p>'
                      '<p><a href="data/arm_spline.json">连续限值与求解报告</a> · '
                      '<a href="data/arm_tracking_errors.csv">跟踪误差CSV</a></p>'
                      '<img src="data/arm_tracking_errors.png">')
    for f in sorted((output/'plots').rglob('*.png')):
        rel=html.escape(str(f.relative_to(output)));blocks.append(f'<p><a href="{rel}">{rel}</a></p>')
    (output/'index.html').write_text('\n'.join(blocks)+'\n')
    deliverables=[f for directory in ('videos','plots','data') for f in (output/directory).rglob('*') if f.is_file()]
    deliverables += [output/n for n in ('index.html','report.json','export_plan.json')]
    save_json(output/'manifest.json',{str(f.relative_to(output)):sha256(f) for f in deliverables})
    if plan['zip']:
        with zipfile.ZipFile(output/'export_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for f in deliverables+[output/'manifest.json']:archive.write(f,f.relative_to(output))
    save_json(output/'status.json',dict(state='completed',job_id=os.environ['SLURM_JOB_ID'],videos=len(videos),index=str(output/'index.html')))
    print('Export complete: '+str(output/'index.html'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    try:
        if plan.get('kind') == 'recorded_trajectory_current_model':
            from render_trajectory_bundle import run as render_trajectory
            render_trajectory(plan)
        else:
            run(plan)
    except Exception as error:
        save_json(Path(plan['output'])/'status.json',dict(state='failed',error=f'{type(error).__name__}: {error}',job_id=os.environ.get('SLURM_JOB_ID')))
        raise


if __name__ == '__main__':
    main()
