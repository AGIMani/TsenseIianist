#!/usr/bin/env python3
# 2026-10-25: The hand base_link frame is named palm_base; exported pose coordinates explicitly name world.
"""Add M6 kinematic IK to a recorded hand replay; export velocities and videos.

GPU rendering must run under scripts/hpc/submit.sh arm-replay on the cluster.
The recorded hand physics and checkpoint are never changed or re-evaluated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tsensei_v3_2_pianist.preview_trajectory import SIDES, DEFAULT_LIMITS, differential_track, scale_path


def read_trace(folder):
    trace = np.load(folder / 'physics_trace.npz')
    data = {key: trace[key] for key in trace.files}
    for key in ('qpos', 'mocap_pos', 'mocap_quat'):
        data[key] = np.concatenate((data['initial_'+key][None], data[key]))
    data['time'] = np.arange(len(data['qpos'])) * float(data['physics_dt'])
    return data


def read_targets(folder, trace, *, model=None):
    # Legacy overlay callers use portable XML across Composer/GPU versions.
    # Native export supplies its captured binary model in the learner runtime.
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(folder / 'kinematic_scene/scene.xml'))
    data = mujoco.MjData(model)
    positions = np.empty((len(trace['time']), 2, 3))
    rotations = np.empty((len(trace['time']), 2, 3, 3))
    ids = [model.body(f'{side}_/base_link').id for side in SIDES]
    for i in range(len(positions)):
        data.qpos[:] = trace['qpos'][i]
        data.mocap_pos[:] = trace['mocap_pos'][i]
        data.mocap_quat[:] = trace['mocap_quat'][i]
        mujoco.mj_forward(model, data)
        positions[i], rotations[i] = data.xpos[ids], data.xmat[ids].reshape(2, 3, 3)
    return positions, rotations


def recorded_tcp_metrics(positions, rotations, dt):
    """Audit at the recorded physics rate, before interpolation/start-stop assumptions."""
    result = {}
    for i, side in enumerate(SIDES):
        velocity = np.diff(positions[:, i], axis=0)/dt
        angular = Rotation.from_matrix(rotations[1:, i] @ rotations[:-1, i].transpose(0, 2, 1)).as_rotvec()/dt
        values = (velocity, np.diff(velocity, axis=0)/dt, angular, np.diff(angular, axis=0)/dt)
        result[side] = dict(zip(DEFAULT_LIMITS, [float(np.linalg.norm(v, axis=1).max()) for v in values]))
    return result


def overlay_models(connector_config=None, arm_config=None):
    from dm_control import mjcf
    from tsensei_v3_2_pianist.preview_connector import ConnectorPreview, DEFAULT_CONFIG, tool_transforms
    from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview, DEFAULT_CONFIG as ARM_CONFIG
    root = mjcf.RootElement(model='arm_overlay')
    connector = ConnectorPreview(root, connector_config or DEFAULT_CONFIG)
    arms = ArmPreview(root, arm_config or ARM_CONFIG, tool_overrides=tool_transforms(connector.config))
    return root, connector, arms


def charts(folder, arms, audit, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time, factor = audit['time'], report['time_scale']
    columns = [time, time*factor]
    names = ['original_time_s', 'limited_time_s']
    for side in SIDES:
        arm = arms.sides[side]
        fig, axes = plt.subplots(7, 2, figsize=(16, 17), sharex='col')
        for index, dof in enumerate(arm['dof']):
            speed = audit['qd'][:, dof]
            limit = arm['velocity'][index]
            for ax, scale in zip(axes[index], (1., factor)):
                ax.plot(time*scale, speed/scale, color='#2879b9', lw=1)
                ax.axhline(limit, color='#c6453d', ls='--', lw=.8)
                ax.axhline(-limit, color='#c6453d', ls='--', lw=.8)
                ax.set_ylim(-max(limit*1.15, np.max(np.abs(speed/scale))*1.1), max(limit*1.15, np.max(np.abs(speed/scale))*1.1))
                ax.set_ylabel(f'J{index+1} (rad/s)')
                ax.grid(alpha=.2)
            columns.extend((audit['q'][:, dof], speed, speed/factor))
            names.extend((f'{side}_J{index+1}_angle_rad', f'{side}_J{index+1}_original_rad_s', f'{side}_J{index+1}_limited_rad_s'))
            single, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(time, speed, lw=1)
            ax.axhline(limit, color='#c6453d', ls='--', label='URDF velocity limit')
            ax.axhline(-limit, color='#c6453d', ls='--')
            ax.set(xlabel='Original replay time (s)', ylabel='Joint velocity (rad/s)',
                   title=f'{side.capitalize()} M6 J{index+1} — differential IK')
            ax.grid(alpha=.2)
            ax.legend()
            single.tight_layout()
            single.savefig(folder / f'{side}_J{index+1}_velocity.png', dpi=160)
            plt.close(single)
        axes[0, 0].set_title('Original timing: IK speed demand')
        axes[0, 1].set_title(f'TCP-limited timing: duration x {factor:.3f}')
        axes[-1, 0].set_xlabel('Original time (s)')
        axes[-1, 1].set_xlabel('Limited time (s)')
        fig.suptitle(f'{side.capitalize()} M6 — all seven joints; red = vendor velocity limits', fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(folder / f'{side}_joint_velocities.png', dpi=140)
        fig.savefig(folder / f'{side}_joint_velocities.pdf')
        plt.close(fig)
    np.savetxt(folder / 'joint_velocities.csv', np.column_stack(columns), delimiter=',', header=','.join(names), comments='', fmt='%.9g')
    original = np.linalg.norm(audit['tcp_vectors'], axis=-1)
    limited = np.linalg.norm(audit['limited_tcp_vectors'], axis=-1)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex='col')
    labels = ('TCP speed (m/s)', 'TCP acceleration (m/s²)', 'TCP angular speed (rad/s)', 'TCP angular acceleration (rad/s²)')
    for row, (label, limit) in enumerate(zip(labels, report['limits'].values())):
        for col, values in enumerate((original, limited)):
            for i, side in enumerate(SIDES):
                axes[row, col].plot(time*(1 if col == 0 else factor), values[:, i, row], label=side, lw=1)
            axes[row, col].axhline(limit, color='#c6453d', ls='--', label=f'limit {limit:g}')
            axes[row, col].set_ylabel(label)
            axes[row, col].grid(alpha=.2)
    axes[0, 0].set_title('Original timing')
    axes[0, 1].set_title(f'Limited timing (x {factor:.3f})')
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(folder / 'tcp_limits.png', dpi=140)
    fig.savefig(folder / 'tcp_limits.pdf')
    plt.close(fig)
    tcp_columns = [time, time*factor]
    tcp_names = ['original_time_s', 'limited_time_s']
    for label, values in (('original', original), ('limited', limited)):
        for i, side in enumerate(SIDES):
            for j, field in enumerate(DEFAULT_LIMITS):
                tcp_columns.append(values[:, i, j])
                tcp_names.append(f'{label}_{side}_{field}')
    np.savetxt(folder / 'tcp_limits.csv', np.column_stack(tcp_columns), delimiter=',', header=','.join(tcp_names), comments='', fmt='%.9g')


def sampled_charts(folder, arms, time, q, report):
    """Export the measured IK increments without spline overshoot or retiming."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mid = (time[1:] + time[:-1]) / 2
    velocity = np.diff(q, axis=0) / np.diff(time)[:, None]
    angle_columns, speed_columns, names = [time], [mid], []
    report['maximum_joint_speed_rad_s'] = {}
    report['minimum_joint_limit_margin_rad'] = {}
    for side in SIDES:
        arm = arms.sides[side]
        angles, speeds = q[:, arm['qadr']], velocity[:, arm['dof']]
        margin = np.minimum(angles-arm['limits'][:, 0], arm['limits'][:, 1]-angles)
        if not np.isfinite(angles).all() or not np.isfinite(speeds).all() or margin.min() < -1e-8:
            raise ValueError('Invalid sampled joint trajectory: ' + side)
        report['minimum_joint_limit_margin_rad'][side] = margin.min(0).tolist()
        report['maximum_joint_speed_rad_s'][side] = {
            'original': np.abs(speeds).max(0).tolist(), 'vendor_limits': arm['velocity'].tolist()}
        fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
        for j in range(7):
            label = f'{side}_J{j+1}'
            names.append(label)
            angle_columns.append(angles[:, j])
            speed_columns.append(speeds[:, j])
            axes[j, 0].plot(time, angles[:, j], lw=.9)
            axes[j, 0].set_ylabel(f'J{j+1} angle (rad)')
            axes[j, 0].grid(alpha=.2)
            single, ax = plt.subplots(figsize=(11, 3.5))
            for target in (axes[j, 1], ax):
                target.plot(mid, speeds[:, j], lw=.9)
                target.axhline(arm['velocity'][j], color='#c6453d', ls='--', label='URDF velocity limit')
                target.axhline(-arm['velocity'][j], color='#c6453d', ls='--')
                target.set_ylabel(f'J{j+1} speed (rad/s)')
                target.grid(alpha=.2)
            ax.set(title=f'{side.capitalize()} M6 J{j+1} | original timing | differential IK', xlabel='Time (s)')
            ax.legend()
            single.tight_layout()
            single.savefig(folder / f'{label}_velocity.png', dpi=160)
            plt.close(single)
        axes[0, 0].set_title('Joint angle at recorded physics samples')
        axes[0, 1].set_title('Interval velocity: delta angle / physics dt')
        for ax in axes[-1]:
            ax.set_xlabel('Original replay time (s)')
        fig.suptitle(f'{side.capitalize()} M6 | {report["label"]} | {report["recorded_physics_hz"]:g} Hz', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .975))
        for suffix in ('png', 'pdf'):
            fig.savefig(folder / f'{side}_joint_velocities.{suffix}', dpi=150)
        plt.close(fig)
    np.savetxt(folder / 'joint_angles.csv', np.column_stack(angle_columns), delimiter=',',
               header=','.join(['time_s']+[s+'_rad' for s in names]), comments='', fmt='%.12g')
    np.savetxt(folder / 'joint_velocities.csv', np.column_stack(speed_columns), delimiter=',',
               header=','.join(['interval_midpoint_time_s']+[s+'_rad_s' for s in names]), comments='', fmt='%.12g')
    np.savez_compressed(folder / 'joint_velocity_samples.npz', time=mid, qd=velocity)


def visual_scene(source, output, root):
    directory = output / 'scene'
    if directory.exists():
        raise FileExistsError('Render scene already exists: ' + str(directory))
    shutil.copytree(source / 'kinematic_scene', directory)
    tree = ET.parse(directory / 'scene.xml')
    xml = tree.getroot()
    extra = ET.fromstring(root.to_xml_string())
    for name, content in root.get_assets().items():
        (directory / name).write_bytes(content)
    for node in extra.iter():
        if 'unnamed_' in node.get('name', ''):
            node.set('name', 'ik_overlay_' + node.get('name').replace('/', '_'))
        if node.get('class') == '/':
            node.attrib.pop('class')
        if node.get('childclass') == '/':
            node.attrib.pop('childclass')
    for tag in ('asset', 'worldbody'):
        for node in extra.find(tag):
            xml.find(tag).append(node)
    # Apply the user's new mounting geometry to this display-only replay.
    # No mj_step is called: dynamics and F1 remain those of the recorded rollout.
    for side in SIDES:
        urdf_path = ROOT / f'tsensei_v3_2_pianist/assets/urdf/TsenseI_{side}.urdf'
        urdf = ET.parse(urdf_path).getroot()
        for link in ('base_link', side.capitalize()+'_hand_palm_link'):
            mesh = urdf.find(f"link[@name='{link}']/visual/geometry/mesh")
            mesh_path = (urdf_path.parent / mesh.get('filename')).resolve()
            filename = mesh_path.name
            body = xml.find(f".//body[@name='{side}_/{link}']")
            visual = body.find(f"geom[@name='{side}_/{link}_visual_0']")
            if visual is None or visual.get('mesh') is None:
                raise ValueError(f'Missing recorded visual mesh for {side}/{link}')
            node = xml.find(f"asset/mesh[@name='{visual.get('mesh')}']")
            dest = f'{side}_{filename}'
            shutil.copy2(mesh_path, directory / dest)
            node.set('file', dest)
    tree.write(directory / 'scene.xml', encoding='utf-8', xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))


def render(source, output, root, connector, arms, trace, q, report, fps, timing='both', label='Recorded policy'):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Cluster GPU rendering requires scripts/hpc/submit.sh arm-replay')
    from scipy.interpolate import CubicSpline
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    from tsensei_v3_2_pianist.preview_connector import offset_transform
    model = visual_scene(source, output, root)
    data = mujoco.MjData(model)
    if model.nq != trace['qpos'].shape[1]:
        raise ValueError('Visual overlay changed recorded hand coordinates')
    mapping = [(int(model.body_mocapid[model.body(v.full_identifier).id]), arms.model.body(name).id)
               for v, name in arms.visuals]
    connector_ids = {s: int(model.body_mocapid[model.body(b.full_identifier).id]) for s, b in connector.bodies.items()}
    hand_ids = {s: model.body(f'{s}_/base_link').id for s in SIDES}
    camera = mujoco.MjvCamera()
    arms.camera(camera)
    camera.azimuth = 140
    camera.elevation = -28
    option = mujoco.MjvOption()
    option.geomgroup[2:5] = 0
    option.sitegroup[:] = 0
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    highlight = ActiveKeyHighlight(model)
    sampled = report.get('path_interpolation') == 'linear between recorded IK samples'
    if sampled and timing != 'original':
        raise ValueError('Sampled IK has no retimed path; use --render-timing original')
    spline = None if sampled else CubicSpline(trace['time'], q, bc_type=((1, np.zeros(q.shape[1])), (1, np.zeros(q.shape[1]))))
    nmocap = trace['mocap_pos'].shape[1]
    duration = float(trace['time'][-1])
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise FileNotFoundError('ffmpeg is required for sound and video muxing')
    with mujoco.Renderer(model, height=900, width=1400, max_geom=3000) as renderer:
        for mode, factor in (('original', 1.), ('limited', report['time_scale'])):
            if timing != 'both' and mode != timing:
                continue
            video = output / f'arms_{mode}_silent.mp4'
            frames = int(np.ceil((duration*factor + 1.)*fps)) + 1
            with imageio.get_writer(video, fps=fps, codec='libx264', quality=8, macro_block_size=2) as writer:
                for frame in range(frames):
                    t = min(frame/fps/factor, duration)
                    index = min(int(t/float(trace['physics_dt'])), len(trace['time'])-2)
                    fraction = np.clip((t-trace['time'][index])/float(trace['physics_dt']), 0, 1)
                    # All hand qpos are scalar hinges/slides; mocap orientation is fixed.
                    data.qpos[:] = (1-fraction)*trace['qpos'][index] + fraction*trace['qpos'][index+1]
                    data.mocap_pos[:nmocap] = (1-fraction)*trace['mocap_pos'][index] + fraction*trace['mocap_pos'][index+1]
                    data.mocap_quat[:nmocap] = trace['mocap_quat'][index]
                    mujoco.mj_forward(model, data)
                    arms.data.qpos[:] = ((1-fraction)*q[index] + fraction*q[index+1]) if sampled else spline(t)
                    mujoco.mj_forward(arms.model, arms.data)
                    for mid, bid in mapping:
                        data.mocap_pos[mid] = arms.data.xpos[bid]
                        data.mocap_quat[mid] = arms.data.xquat[bid]
                    for side, mid in connector_ids.items():
                        bid = hand_ids[side]
                        h = np.eye(4)
                        h[:3, :3], h[:3, 3] = data.xmat[bid].reshape(3, 3), data.xpos[bid]
                        h = h @ offset_transform(connector.config['offsets'][side])
                        data.mocap_pos[mid] = h[:3, 3]
                        data.mocap_quat[mid] = np.roll(Rotation.from_matrix(h[:3, :3]).as_quat(), 1)
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    highlight.color_render_scene(renderer.scene, data)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 1400, 64), fill=(20, 30, 44))
                    title = f'{label} | original timing | differential IK' if mode == 'original' else f'{label} | TCP-limited kinematic replay | duration x {factor:.3f}'
                    draw.text((22, 12), title, fill='white', font_size=22)
                    draw.text((22, 39), f'STEP mount + recorded hand trajectory | kinematic arms | original t={t:.2f}s', fill='#c9d4e3', font_size=17)
                    writer.append_data(np.asarray(image))
                    if frame == fps:
                        image.save(output / f'arms_{mode}_preview.png')
            filters = [f'atrim=duration={duration:.9g}', 'asetpts=PTS-STARTPTS']
            tempo = 1/factor
            while tempo < .5:
                filters.append('atempo=0.5')
                tempo /= .5
            filters.extend((f'atempo={tempo:.12g}', 'apad'))
            if mode == 'original':
                # Keep the recorded piano decay during the final video hold.
                filters = ['apad']
            final = output / f'arms_{mode}.mp4'
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-i', str(video),
                            '-i', str(source/'played.wav'), '-map', '0:v:0', '-map', '1:a:0',
                            '-af', ','.join(filters), '-t', str(frames/fps), '-c:v', 'copy', '-c:a', 'aac',
                            '-movflags', '+faststart', str(final)], check=True)
            video.unlink()
            print('Saved ' + str(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connector-config', type=Path)
    parser.add_argument('--arm-config', type=Path)
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--render-timing', choices=('original', 'limited', 'both'), default='both')
    parser.add_argument('--label', default='Recorded policy')
    parser.add_argument('--original-samples', action='store_true',
                        help='Keep recorded timing; export interval speeds and bounded linear video interpolation')
    for key, value in DEFAULT_LIMITS.items():
        parser.add_argument('--'+key.replace('_', '-'), type=float, default=value)
    args = parser.parse_args()
    if args.analysis_only and args.render_only:
        parser.error('Choose analysis-only or render-only')
    if args.original_samples and args.render_timing != 'original':
        parser.error('--original-samples requires --render-timing original')
    source, output = args.replay.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = read_trace(source)
    root, connector, arms = overlay_models(args.connector_config, args.arm_config)
    if args.render_only:
        saved = np.load(output/'arm_trajectory.npz')
        q = saved['q']
        report = json.loads((output/'report.json').read_text())
        if report['connector'] != connector.config or report['arm_mount'] != arms.config:
            raise ValueError('Saved IK was computed with a different mounting configuration')
        if report['physics_trace_sha256'] != hashlib.sha256((source/'physics_trace.npz').read_bytes()).hexdigest():
            raise ValueError('Recorded hand trace changed after IK analysis')
    else:
        positions, rotations = read_targets(source, trace)
        q, errors = differential_track(arms, positions, rotations)
        print('Differential IK complete; max error (m, rad): ' + str(errors.max(axis=(0, 1))), flush=True)
        if args.original_samples:
            report = {'time_scale': 1., 'original_duration_s': float(trace['time'][-1]),
                      'path_interpolation': 'linear between recorded IK samples',
                      'velocity_definition': 'delta q / physics_dt, timestamp at interval midpoint; no filtering',
                      'palm_base_frame': 'hand base_link origin with configured Link7-to-hand transform; not flange origin'}
        else:
            _, report, audit = scale_path(trace['time'], q, arms, {k: getattr(args, k) for k in DEFAULT_LIMITS})
        report.update(replay=str(source), physics_trace_sha256=hashlib.sha256((source/'physics_trace.npz').read_bytes()).hexdigest(),
                      connector=connector.config, arm_mount=arms.config,
                      recorded_physics_hz=1./float(trace['physics_dt']),
                      recorded_physics_maxima=recorded_tcp_metrics(positions, rotations, float(trace['physics_dt'])),
                      label=args.label,
                      maximum_ik_error_m_rad=errors.max(axis=(0, 1)).tolist(),
                      limitations=['Recorded hand physics uses the original training model and mass.',
                                   'Only the display uses the revised STEP wrist mount; no new F1 is calculated.',
                                   'Arms and six adapter pieces are kinematic, without added dynamics or contact forces.',
                                   'TJ-ARM-02/03: shoulder and flange clocking remain provisional.',
                                   'Limited replay changes tempo; 1000 Hz path audit with 2% scale margin is not hardware certification.'])
        if args.original_samples:
            report['limitations'][-1] = 'Original timing is preserved; joint speed limits are diagnostic, not enforced; no acceleration feasibility claim.'
        np.savez_compressed(output/'arm_trajectory.npz', time=trace['time'], q=q, ik_errors=errors,
                            target_positions=positions, target_rotations=rotations, time_scale=report['time_scale'])
        if args.original_samples:
            sampled_charts(output, arms, trace['time'], q, report)
        else:
            np.savez_compressed(output/'arm_dynamics_1000hz.npz', **audit)
            charts(output, arms, audit, report)
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report['maximum_joint_speed_rad_s'], indent=2), flush=True)
    if not args.analysis_only:
        render(source, output, root, connector, arms, trace, q, report, args.fps, args.render_timing, args.label)


if __name__ == '__main__':
    main()
