#!/usr/bin/env python3
"""Render a confirmed J1..J7 left-arm CSV with the existing fixed A3.2 hand.

This is a CPU software-rendered kinematic replay, with no policy or dynamics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_csv(path):
    import numpy as np
    values = np.genfromtxt(path, delimiter=',', names=True)
    names = ['time_s'] + [f'marvin_0_arm_joint_{i}_{suffix}'
                         for suffix in ('q_rad', 'qd_rad_s', 'qdd_rad_s2') for i in range(7)]
    if list(values.dtype.names) != names:
        raise ValueError('Expected time and seven ordered q/qd/qdd columns with explicit SI units')
    arrays = [np.asarray(values[names[0]])] + [np.column_stack([values[n] for n in names[a:a+7]])
                                            for a in (1, 8, 15)]
    time, q, qd, qdd = arrays
    if len(time) < 2 or not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('CSV must contain at least two finite samples')
    if time[0] != 0 or not np.allclose(np.diff(time), .005, rtol=0, atol=1e-10):
        raise ValueError('This importer requires the supplied 200 Hz clock starting at zero')
    return time, q, qd, qdd


def build_scene(manifest):
    import numpy as np
    from dm_control import mjcf
    from scipy.spatial.transform import Rotation
    from tsensei_v3_2_pianist.models.piano.piano import Piano
    root = mjcf.RootElement(model='left_arm_csv')
    root.compiler.angle = 'radian'
    global_settings = getattr(root.visual, 'global')
    global_settings.offwidth = 1440
    global_settings.offheight = 720
    global_settings.orthographic = True
    root.visual.headlight.ambient = [.5, .5, .5]
    root.visual.headlight.diffuse = [.5, .5, .5]
    root.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[.7, .7, .7])
    root.worldbody.add('geom', name='floor', type='plane', size=[4, 4, .1],
                       pos=[0, 0, -.075], rgba=[.15, .18, .22, 1], contype=0, conaffinity=0)
    piano = Piano()
    root.attach(piano.mjcf_model)
    bodies, joints, initial = {}, {}, {}
    for component in manifest['models']:
        cid = component['id']
        if component['parent']:
            p = component['parent']
            parent = bodies[(p['model'], p['link'])]
            transform = np.array(p['transform'])
        else:
            parent = root.worldbody
            transform = np.array(component['root_transform'])
        incoming = {j['child']: j for j in component['joints']}
        children = {}
        for j in component['joints']:
            children.setdefault(j['parent'], []).append(j['child'])
        links = {link['name']: link for link in component['links']}

        def add_link(name, parent, matrix):
            body = parent.add('body', name=cid+'__'+name, pos=matrix[:3, 3],
                              quat=np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1))
            # FK-only inertial proxy. No simulation step uses this mass or inertia.
            body.add('inertial', pos=[0, 0, 0], mass=1, diaginertia=[.01]*3)
            bodies[(cid, name)] = body
            joint = incoming.get(name)
            if joint and joint['type'] != 'fixed':
                hinge = body.add('joint', name=cid+'__'+joint['urdf_name'],
                                 type='hinge' if joint['type'] == 'revolute' else 'slide',
                                 axis=joint['axis'], limited=True,
                                 range=[joint['limit']['lower'], joint['limit']['upper']])
                joints[(cid, joint['name'])] = hinge
                initial[(cid, joint['name'])] = component['initial_positions'].get(joint['name'], 0.)
            for k, v in enumerate(links[name]['visuals']):
                mesh = root.asset.add('mesh', name=f'{cid}__{name}__{k}',
                                      file=manifest['assets'][v['asset_id']], scale=v['scale'])
                tm = np.array(v['transform'])
                rgba = v['color'] if cid == 'hand_left' and k else [.84, .87, .91, 1]
                body.add('geom', name=f'{cid}__{name}__geom{k}', type='mesh', mesh=mesh,
                         pos=tm[:3, 3], quat=np.roll(Rotation.from_matrix(tm[:3, :3]).as_quat(), 1),
                         rgba=rgba, contype=0, conaffinity=0, group=1)
            for child in children.get(name, []):
                add_link(child, body, np.array(incoming[child]['origin']))
            return body
        add_link(component['root_link'], parent, transform)
    return root, bodies, joints, initial


def plots(output, time, q, qd, qdd, limits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for name, values, unit, reference in [
        ('positions', q, 'rad', None), ('velocities', qd, 'rad/s', limits['velocity']),
        ('accelerations', qdd, 'rad/s²', limits['acceleration'])]:
        fig, axes = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time, values[:, i], lw=.8, color='#287ca6')
            ax.set_ylabel(f'J{i+1}\n{unit}')
            ax.grid(alpha=.2)
            if reference is not None:
                ax.axhline(reference[i], color='#c54b4b', ls='--', lw=.8)
                ax.axhline(-reference[i], color='#c54b4b', ls='--', lw=.8)
                over = np.abs(values[:, i]) > reference[i]
                ax.scatter(time[over], values[over, i], s=3, color='#c54b4b')
        axes[-1].set_xlabel('CSV time (seconds; original speed)')
        fig.suptitle(f'Left arm {name}: supplied 200 Hz CSV samples\n'
                     + ('Dashed lines: bundled M6 SDK reference; supplied derivatives are preserved'
                        if reference is not None else 'J1..J7 mapping confirmed by user; no position modification'))
        fig.tight_layout(rect=[0, 0, 1, .95])
        fig.savefig(output/f'left_arm_{name}.png', dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirmed-j1-j7', action='store_true')
    parser.add_argument('--fps', type=int, default=40)
    args = parser.parse_args()
    if not args.confirmed_j1_j7:
        parser.error('Confirm arm model, J1..J7 order, zero and direction before applying the CSV')
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.fps < 1 or args.fps > 200 or 200 % args.fps:
        parser.error('FPS must divide the 200 Hz source clock')
    # Software-only renderer, enforced before any graphics imports.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from online_control.models import build_scene_manifest, forward_kinematics
    from tsensei_v3_2_pianist.preview_backend import current_gl_info
    from preview.scripts.arm_motion_limits import load_m6_joint_limits

    time, q, qd, qdd = load_csv(args.csv)
    manifest = build_scene_manifest()
    manifest['models'] = [m for m in manifest['models'] if m['id'] in ('arm_left', 'hand_left')]
    if {m['id'] for m in manifest['models']} != {'arm_left', 'hand_left'}:
        raise ValueError('Current local left arm and left hand models are required')
    used_assets = {v['asset_id'] for c in manifest['models'] for link in c['links'] for v in link['visuals']}
    manifest['assets'] = {k:v for k,v in manifest['assets'].items() if k in used_assets}
    arm = next(m for m in manifest['models'] if m['id'] == 'arm_left')
    arm_joints = [j for j in arm['joints'] if j['type'] != 'fixed']
    expected = [f'Joint{i}_L' for i in range(1, 8)]
    if [j['urdf_name'] for j in arm_joints] != expected:
        raise ValueError('Local arm is not the confirmed J1..J7 model')
    bounds = np.array([[j['limit']['lower'], j['limit']['upper']] for j in arm_joints])
    if np.any(q < bounds[:, 0]-1e-10) or np.any(q > bounds[:, 1]+1e-10):
        raise ValueError('CSV positions exceed current URDF limits; values are never clipped')
    root, bodies, joints, initial = build_scene(manifest)
    model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
    data = mujoco.MjData(model)
    for key, joint in joints.items():
        data.qpos[model.joint(joint.full_identifier).qposadr[0]] = initial[key]
    addresses = [model.joint(joints[('arm_left', j['name'])].full_identifier).qposadr[0]
                 for j in arm_joints]
    dofs = [model.joint(joints[('arm_left', j['name'])].full_identifier).dofadr[0]
            for j in arm_joints]
    hand = next(m for m in manifest['models'] if m['id'] == 'hand_left')
    hand_id = model.body(bodies[('hand_left', hand['root_link'])].full_identifier).id
    body_ids = {key: model.body(value.full_identifier).id for key, value in bodies.items()}
    robot_geoms = np.array([i for i in range(model.ngeom)
                           if (model.geom(i).name or '').startswith(('arm_left__', 'hand_left__'))])
    radius = model.geom_rbound[robot_geoms, None]
    hand_geoms = np.array([i for i in robot_geoms if model.geom(i).name.startswith('hand_left__')])
    hand_radius = model.geom_rbound[hand_geoms, None]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    close_height = 0.
    wrists = []
    fk_error = 0.
    validation_indices = set(np.linspace(0, len(time)-1, 65).round().astype(int))
    for i in range(len(time)):
        data.qpos[addresses] = q[i]
        data.qvel[dofs] = qd[i]
        mujoco.mj_forward(model, data)
        assert np.array_equal(data.qpos[addresses], q[i])
        low = np.minimum(low, (data.geom_xpos[robot_geoms]-radius).min(0))
        high = np.maximum(high, (data.geom_xpos[robot_geoms]+radius).max(0))
        hand_low = (data.geom_xpos[hand_geoms]-hand_radius).min(0)
        hand_high = (data.geom_xpos[hand_geoms]+hand_radius).max(0)
        close_height = max(close_height, float(np.linalg.norm(hand_high-hand_low)*1.8))
        wrists.append(data.xpos[hand_id].copy())
        if i in validation_indices:
            independent = forward_kinematics(manifest,
                {'arm_left': dict(zip([j['name'] for j in arm_joints], q[i]))}, use_defaults=True)
            for key, bid in body_ids.items():
                matrix = np.eye(4)
                matrix[:3, 3] = data.xpos[bid]
                matrix[:3, :3] = data.xmat[bid].reshape(3, 3)
                fk_error = max(fk_error, float(np.abs(matrix-independent[key[0]][key[1]]).max()))
    if fk_error > 1e-10:
        raise ValueError('Independent URDF FK and rendered MuJoCo transforms disagree')
    output = args.output
    output.mkdir(parents=True)
    (output/'plots').mkdir()
    (output/'data').mkdir()
    shutil.copy2(args.csv, output/'data/left_arm_200Hz.csv')
    save_json(output/'data/scene_manifest.json', manifest)
    save_json(output/'data/left_hand_fixed_pose.json', hand['initial_positions'])
    np.savez_compressed(output/'data/trajectory.npz', time=time, q=q, qd=qd, qdd=qdd,
                        hand_base_xyz=np.array(wrists), input_sample_hz=200)
    sdk = load_m6_joint_limits()
    # The existing SDK snapshot lists both sides; select the left arm explicitly.
    limits = dict(velocity=np.array([sdk['joints'][f'left_J{i}']['velocity'] for i in range(1, 8)]),
                  acceleration=np.array([sdk['joints'][f'left_J{i}']['acceleration'] for i in range(1, 8)]))
    plots(output/'plots', time, q, qd, qdd, limits)
    save_json(output/'data/sdk_reference_limits.json', sdk)
    print('CSV and independent FK verified; plotting complete.', flush=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.orthographic = True
    camera.azimuth, camera.elevation, camera.distance = 135., -25., 4.
    camera.lookat[:] = (low+high)/2
    # Conservative world bounding sphere covers every source pose and all robot meshes.
    height = float(np.linalg.norm(high-low)*1.18)
    ffmpeg = ROOT/'training_mjlab/hpc/tools/ffmpeg'
    final = output/'left_arm_with_hand.mp4'
    encoder = subprocess.Popen([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1440x720', '-framerate', str(args.fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(final)], stdin=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    stride = 200//args.fps
    samples = list(range(0, len(time), stride))
    # Inclusive endpoint plus remaining hold frames total exactly music_duration+1 seconds.
    frames = samples + [len(time)-1]*(args.fps-1)
    rendered_times = []
    try:
        with mujoco.Renderer(model, height=720, width=960) as wide, \
                mujoco.Renderer(model, height=720, width=480) as close:
            gl = current_gl_info()
            if not gl['software_renderer']:
                raise RuntimeError('This command requires CPU llvmpipe; GPU rendering belongs in Slurm')
            for frame, index in enumerate(frames):
                data.qpos[addresses] = q[index]
                data.qvel[dofs] = qd[index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = (low+high)/2
                model.vis.global_.fovy = height
                wide.update_scene(data, camera=camera)
                wide.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
                first = wide.render()
                hand_low = (data.geom_xpos[hand_geoms]-hand_radius).min(0)
                hand_high = (data.geom_xpos[hand_geoms]+hand_radius).max(0)
                camera.lookat[:] = (hand_low+hand_high)/2
                # Portrait panel needs extra width allowance; frame the complete hand,
                # including the installed connector, rather than centering on its base.
                model.vis.global_.fovy = close_height
                close.update_scene(data, camera=camera)
                close.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
                second = close.render()
                picture = Image.fromarray(np.concatenate([first, second], axis=1))
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 1440, 62), fill='#121b26')
                draw.text((15, 7), 'Left M6 arm + A3.2 hand | supplied CSV | kinematic replay', font=font, fill='white')
                draw.text((15, 35), f't={time[index]:.3f} s | 200 Hz CSV / {args.fps} fps video | 1x | hand fixed at scene origin pose', font=small, fill='#bcdbe5')
                draw.text((980, 8), 'Wrist-following close view', font=small, fill='#ffd380')
                draw.rectangle((0, 653, 1440, 720), fill='#121b26')
                draw.text((15, 660), 'q [rad]: '+ '  '.join(f'J{i+1} {v:+.3f}' for i, v in enumerate(q[index])), font=small, fill='white')
                draw.text((15, 685), 'qd [rad/s]: '+ '  '.join(f'J{i+1} {v:+.3f}' for i, v in enumerate(qd[index])), font=small, fill='#acdacf')
                encoder.stdin.write(np.asarray(picture).tobytes())
                rendered_times.append(float(time[index]))
                if frame in (0, len(samples)//2, len(samples)-1):
                    picture.save(output/f'frame_{frame:04d}.png')
                if frame % 200 == 0:
                    print(f'Rendered {frame}/{len(frames)} frames', flush=True)
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg encoding failed: {code}')
    subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-xerror',
                    '-i', str(final), '-f', 'null', '-'], check=True)
    np.save(output/'data/video_source_times.npy', np.array(rendered_times))
    report = dict(mode='external_csv_kinematic_replay', original_csv=str(args.csv),
        csv_sha256=sha(args.csv), joint_mapping={f'marvin_0_arm_joint_{i}': j['name'] for i, j in enumerate(arm_joints)},
        mapping_confirmation='User confirmed J1..J7 order, zero and positive direction in this chat',
        source_hz=200, source_samples=len(time), source_duration_seconds=float(time[-1]),
        render_fps=args.fps, render_frames=len(frames), video_duration_seconds=len(frames)/args.fps,
        final_hold_seconds=1., speed=1, audio=False, renderer=gl,
        independent_fk_max_matrix_error=fk_error, independent_fk_sampled_poses=len(validation_indices),
        q_min_rad=q.min(0).tolist(), q_max_rad=q.max(0).tolist(),
        qd_abs_max_rad_s=np.abs(qd).max(0).tolist(), qdd_abs_max_rad_s2=np.abs(qdd).max(0).tolist(),
        sdk_velocity_over_samples=(np.abs(qd)>limits['velocity']).sum(0).tolist(),
        sdk_acceleration_over_samples=(np.abs(qdd)>limits['acceleration']).sum(0).tolist(),
        central_difference_vs_supplied_velocity_max_abs=np.abs(np.gradient(q,time,axis=0)-qd).max(0).tolist(),
        left_hand_pose_source='Existing current scene initial_positions; fixed throughout replay',
        scene_manifest='data/scene_manifest.json', decoded_all_frames=True,
        limitations=['Kinematic CSV visualization; no policy inference, dynamics stepping, or hardware execution.',
                    'Current shoulder and flange mounting are provisional, inherited without adjustment.',
                    'CSV has no finger trajectory, original MIDI, or physical key activation recording; music comparison is not applicable.'],
        model_inputs_sha256={str(Path(manifest['assets'][v['asset_id']]).relative_to(ROOT)):
                             sha(manifest['assets'][v['asset_id']]) for c in manifest['models']
                             for link in c['links'] for v in link['visuals']})
    save_json(output/'report.json', report)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Left arm CSV replay</title>'
        '<style>body{font:16px sans-serif;background:#121b26;color:#eef;max-width:1440px;margin:24px auto}'
        'video,img{width:100%}a{color:#9cf}</style><h1>Left arm + A3.2 hand</h1>'
        '<p>31 s / 200 Hz supplied CSV, J1..J7 mapping confirmed; fixed hand origin pose. '
        'Kinematic replay, provisional mounting, no piano performance score.</p>'
        '<video controls src="left_arm_with_hand.mp4"></video><p><a href="report.json">Provenance and checks</a> · '
        '<a href="data/left_arm_200Hz.csv">Original CSV</a> · <a href="data/left_hand_fixed_pose.json">Fixed hand pose</a></p>'
        + ''.join(f'<h2>{name}</h2><img src="plots/left_arm_{name}.png">' for name in ('positions','velocities','accelerations')))
    deliverables = [p for p in output.rglob('*') if p.is_file()]
    save_json(output/'manifest.json', {str(p.relative_to(output)):sha(p) for p in deliverables})
    with zipfile.ZipFile(output/'replay_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in deliverables+[output/'manifest.json']:
            archive.write(p, p.relative_to(output))
    with zipfile.ZipFile(output/'replay_bundle.zip') as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC check failed')
    print(json.dumps({k:v for k,v in report.items() if k not in ('model_inputs_sha256','limitations')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
