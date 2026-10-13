#!/usr/bin/env python3
"""View the frozen MJLab scene, or audit the exact training arm workspace.

``--training-config`` is the synchronized path.  It rebuilds the source
recipe and passes the saved reward, control, preparation and current rigid_wrist_command arm
options into ``PianoTaskEnv``.  The legacy scene-only invocation remains
available for inspecting an exported scene, but it is not a training replay.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=None)
    parser.add_argument('--training-config', type=Path,
                        help='Full training config.json; rebuilds the identical scene/task and arm contract')
    parser.add_argument('--reachability-audit', type=Path,
                        help='Run a headless CPU arm workspace scan into this directory')
    parser.add_argument('--reachability-tx-points', type=int, default=41)
    parser.add_argument('--reachability-ty-points', type=int, default=9)
    parser.add_argument('--reachability-pitch-points', type=int, default=5)
    parser.add_argument('--reachability-preparation-points', type=int, default=None)
    parser.add_argument('--reachability-extra-attempts', type=int, default=12)
    parser.add_argument('--contact-controls', action='store_true', help='Live solimp panel; explicitly changes this preview physics')
    parser.add_argument('--show-hand-collisions', action='store_true',
                        help='Show the loaded scene collision hulls, contact points and forces; display only')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--task', type=Path, help='Optional exported fingered music task; report goals, reward and F1')
    parser.add_argument('--show-fingering', action=argparse.BooleanOptionalAction, default=True,
                        help='Show finger-colored targets, fingertips and assignments when --task is loaded (default: on)')
    parser.add_argument('--checkpoint', type=Path, help='Play a Torch SAC policy deterministically; requires --task')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--camera-view', choices=('front','performer','overview'), default='front')
    parser.add_argument('--camera-framing', choices=('full','hands'), default='full')
    parser.add_argument('--camera-distance', type=float)
    parser.add_argument('--camera-elevation', type=float)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--bottom-out', action='store_true', help='Press at -25 mm wrist with +0.1 rad middle MCP flexion')
    parser.add_argument('--demo', action='store_true', help='Repeat wrist lowering/pressing/releasing with GPU physics')
    args = parser.parse_args()
    if args.training_config is not None and (args.scene is not None or args.task is not None or args.checkpoint is not None):
        parser.error('--training-config cannot be combined with --scene, --task or --checkpoint')
    if args.reachability_audit is not None and args.training_config is None:
        parser.error('--reachability-audit requires --training-config')
    if args.reachability_audit is not None and any((args.contact_controls, args.demo, args.bottom_out, args.checkpoint)):
        parser.error('--reachability-audit is headless and cannot run viewer/checkpoint controls')
    if args.training_config is not None:
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import load_training_config
        config = load_training_config(args.training_config)
    else:
        config = None
    if args.reachability_audit is None and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('Open this preview in a desktop terminal with DISPLAY set; '
                     'for headless acceptance use scripts/validate_mjlab_env.py --render')
    if args.reachability_audit is not None:
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_bundle_from_config
        from tsensei_v3_2_pianist.mjlab_env.arm_reachability import scan_reachability
        bundle, scene_directory, _ = build_bundle_from_config(config)
        summary = scan_reachability(
            bundle,
            args.reachability_audit,
            tx_points=args.reachability_tx_points,
            ty_points=args.reachability_ty_points,
            pitch_points=args.reachability_pitch_points,
            preparation_points=args.reachability_preparation_points,
            extra_attempts=args.reachability_extra_attempts,
        )
        print(json.dumps({
            'config': config['_preview_config_path'],
            'scene': str(scene_directory),
            'sides': summary['sides'],
            'output': str(Path(args.reachability_audit).expanduser().resolve()),
        }, indent=2), flush=True)
        return
    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=False, gpu='auto')
    from tsensei_v3_2_pianist.mjlab_env.paths import CURRENT_GRAVCOMP_SCENE, for_checkpoint
    if args.checkpoint is not None:
        args.scene,args.task=for_checkpoint(args.checkpoint,args.scene,args.task)
    elif config is not None:
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_source_from_config
        args.scene,args.task,_=build_source_from_config(config)
    elif args.scene is None:
        args.scene=CURRENT_GRAVCOMP_SCENE
    gravity_compensation = json.loads((args.scene / 'manifest.json').read_text())['gravity_compensation']
    print(f'Gravity compensation: {"on" if gravity_compensation else "off"}; scene={args.scene}', flush=True)
    if args.checkpoint and (args.task is None or args.demo or args.bottom_out):
        parser.error('--checkpoint requires --task and cannot be combined with --demo/--bottom-out')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    import mujoco
    import mujoco.viewer
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    from tsensei_v3_2_pianist.mjlab_env.acceptance_cases import press_action
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    task_env = None
    observation = None
    if args.task is not None:
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.checkpoint import reward_options
        if config is not None:
            from tsensei_v3_2_pianist.mjlab_env.preview_contract import task_kwargs
            options = task_kwargs(config)
        else:
            options = reward_options(args.checkpoint) if args.checkpoint else {}
        task_env = PianoTaskEnv(args.scene, args.task, device=args.device, **options)
        scene = task_env.scene
        observation = task_env.reset()
    else:
        scene = PianoScene(args.scene, device=args.device)
    policy = None
    if args.checkpoint is not None:
        from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent, reward_options
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        policy = load_agent(args.checkpoint, scene_sha256=sha256(args.scene/'manifest.json'),
                            task_sha256=sha256(args.task/'task.json'), obs_size=task_env.observation_size,
                            action_size=task_env.action_size, device=args.device,
                            action_contract=scene.action_contract,
                            control_limits_contract=scene.control_limits,
                            finger_clearance_contract=scene.finger_clearance,
                            finger_coupling_contract=scene.finger_coupling,
                            coupling_observation_contract=task_env.coupling_observation_contract,
                            arm_follow_contract=(scene.arm_follower.contract if scene.arm_follower is not None else None),
                            preparation_contract=scene.bundle.preparation, **options)
    key_highlight = ActiveKeyHighlight(scene.model)
    fingering_highlight = None
    if task_env is not None and args.show_fingering:
        from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight
        fingering_highlight = FingeringHighlight(scene.model, task_env.bundle.manifest, key_highlight)
    if args.show_hand_collisions:
        from tsensei_v3_2_pianist.preview_geometry import color_hand_collision_geoms
        color_hand_collision_geoms(scene.model)
    b = scene.bundle
    data = scene.cpu_snapshot()
    from preparation_frames import frame_indices, snapshot as preparation_snapshot, timeline, phase_at
    preparation_timing = timeline(b, None if task_env is None else task_env.bundle.manifest)
    preparation_trace = preparation_timing['prefix_trace']
    preparation_indices = (np.empty(0, dtype=int) if preparation_trace is None else
                           frame_indices(preparation_trace, b.manifest['control_dt'], include_endpoint=True))
    preparation_frame = 0
    if preparation_trace is not None:
        preparation_snapshot(scene.model, preparation_trace, 0, data)
        print('Each reset shows the controlled raised-to-origin preparation before policy playback; '
              'preparation is outside score time and F1.', flush=True)
    elif preparation_timing['policy_active']:
        print('Policy starts immediately in the raised pose. Preparation frames are actual '
              'GPU actions with normal rewards and sound-producing contacts.', flush=True)
    from tsensei_v3_2_pianist.physics_settings import from_model, shadow_comparison
    compiled_contacts = from_model(scene.model, b.manifest['control_dt'])
    shadow_contact = shadow_comparison(compiled_contacts)
    effective = shadow_contact['hand_key_contact']
    state = 'MATCH' if shadow_contact['matches_shadow'] else 'DIFF'
    print(f'Shadow contact parameters: {state}; hand-key solref={effective["mixed_solref"]}, '
          f'solimp={effective["mixed_solimp"]}, refsafe time constant='
          f'{effective["refsafe_timeconst_s"]:.6g}s', flush=True)
    if shadow_contact['differences']:
        print(f'Shadow contact differences: {shadow_contact["differences"]}', flush=True)
    events = {'paused': args.paused, 'reset': False}
    def key(code):
        if code == 32:
            events['paused'] = not events['paused']
        elif code in (ord('R'), ord('r')):
            events['reset'] = True
    print('MJLab CUDA physics. Space: pause; R: reset. No pose writing here.', flush=True)
    print('Torch SAC deterministic playback; policy controls all joints and pedal.' if policy is not None
          else 'Control: drag targets; press Space to run the position controllers.', flush=True)
    tick = 0
    last_active = None
    last_fingers = None
    last_fingering_text = None
    last_arm_text = None
    if task_env is not None:
        print(f'Music task: {task_env.bundle.manifest["length"]} frames, '
              f'{task_env.observation_size} observations. Wrong keys affect reward only. '
              'At score completion press R to replay.', flush=True)
    panel = None
    contacts = None
    if args.contact_controls:
        from tsensei_v3_2_pianist.physics_settings import SolimpController
        from tsensei_v3_2_pianist.preview_contacts import ContactPanel
        from datetime import datetime
        contacts = SolimpController(scene.model, compiled_contacts, scene)
        # A frozen/old checkpoint's contact edits are saved as a separate experiment.
        panel = ContactPanel(contacts, save_path=ROOT/'preview/outputs/contact_settings'/f'gpu-{datetime.now():%Y%m%d-%H%M%S-%f}.json')
        print('Contact editing enabled: this playback is an altered-physics experiment.',flush=True)
    previous_threads = set(threading.enumerate())
    viewer_threads = []
    try:
        with mujoco.viewer.launch_passive(scene.model, data, key_callback=key, show_right_ui=True) as viewer:
            # MuJoCo 3.11 launches a daemon render thread. close() only signals it;
            # wait for its GL teardown before Python/CUDA interpreter shutdown.
            viewer_threads = [t for t in threading.enumerate() if t not in previous_threads
                              and '_launch_internal' in t.name]
            from tsensei_v3_2_pianist.preview_camera import configure_camera
            configure_camera(viewer.cam,scene.model,view=args.camera_view,framing=args.camera_framing,
                             distance=args.camera_distance,elevation=args.camera_elevation)
            if 'm6' in b.manifest and args.camera_distance is None:
                configure_camera(viewer.cam,scene.model,view='front',framing='full',distance=1.65,elevation=-25)
                viewer.cam.lookat[:]=(.22,0,.28)
            if args.show_hand_collisions:
                from tsensei_v3_2_pianist.preview_geometry import configure_options
                configure_options(viewer.opt, show_hand_collisions=True)
            start = time.monotonic()
            while viewer.is_running() and (args.duration is None or time.monotonic()-start < args.duration):
                if panel is not None:
                    panel.poll()
                    with viewer.lock():
                        contacts.apply()
                begin = time.monotonic()
                # A state-only sync deliberately ignores the MuJoCo Control panel's
                # edits to ``data.ctrl``.  That is correct for deterministic policy
                # playback, but it made hand-only MJLab previews appear immune to
                # the sliders.  Manual previews must use the full sync so the
                # edited actuator targets reach the physical-action conversion below.
                viewer.sync(state_only=(policy is not None))
                if events['reset']:
                    if task_env is not None:
                        observation = task_env.reset()
                    else:
                        scene.reset()
                    events['reset'] = False; tick = 0
                    data.ctrl[:] = b.reset_ctrl
                    preparation_frame = 0
                    if preparation_trace is not None:
                        preparation_snapshot(scene.model, preparation_trace, 0, data)
                    last_fingering_text = None
                if task_env is not None and task_env.logic.done[0].item() and not events['paused']:
                    events['paused'] = True
                    print('The score has ended; press R before resuming.', flush=True)
                is_preparing = preparation_frame < len(preparation_indices)
                if not events['paused'] and not is_preparing:
                    if policy is not None:
                        action = policy.act(observation, deterministic=True)
                    else:
                        # The observation/physical contract retains removed wrist
                        # slots; only active actuators appear in render data.ctrl.
                        physical = b.neutral_action.copy()
                        physical[b.active_action_indices] = data.ctrl[b.action_ids]
                        physical[-1] = 0.0
                        if args.demo or args.bottom_out:
                            physical = b.neutral_action.copy()
                            if 10 <= tick % 60 < 22:
                                physical = press_action(b)
                        action = scene.policy_action(np.clip(physical, b.low, b.high)[None])
                    if task_env is None:
                        scene.step(action)
                    else:
                        result = task_env.step(action)
                        observation = result.observation
                        if tick % 10 == 0 or result.done[0].item():
                            goals = np.flatnonzero(result.info['goal_applied'][0,:88].cpu().numpy()).tolist()
                            print(f'Score frame {result.info["score_index"][0].item()}: target={goals}, '
                                  f'reward={result.reward[0].item():.4f}, '
                                  f'mean frame F1={result.info["episode_metrics"][0,2].item():.4f}', flush=True)
                        if result.done[0].item():
                            print('Score complete. Press R to reset, Space to resume.', flush=True)
                            events['paused'] = True
                    tick += 1
                if is_preparing:
                    trace_index = preparation_indices[preparation_frame]
                    snapshot = preparation_snapshot(scene.model, preparation_trace, trace_index)
                    active = np.flatnonzero(preparation_trace['key_activation'][trace_index]).tolist()
                    text = (f'Preparing to play {snapshot.time:.2f}/{preparation_trace["time"][-1]:.2f}s\n'
                            'Returning smoothly to the saved starting pose\n'
                            'Playback starts after preparation')
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
                    if not events['paused']:
                        preparation_frame += 1
                else:
                    snapshot = scene.cpu_snapshot()
                    active = np.flatnonzero(scene.state()['key_activation'][0].cpu().numpy()).tolist()
                    text = ''
                    if preparation_trace is not None and fingering_highlight is None:
                        text = ('Preparation complete; policy playback' if policy is not None else
                                'Preparation complete; controls active')
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                          mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
                if active != last_active:
                    print(f'Active piano keys (0..87): {active}', flush=True)
                    last_active = active
                if fingering_highlight is not None and not is_preparing:
                    # Match the current observation/next action, including initial
                    # pause and reset; completion retains the final score frame.
                    score_index = min(int(task_env.logic.index[0].item()), task_env.logic.length - 1)
                    fingers = task_env.bundle.fingers[score_index]
                    completed = bool(task_env.logic.done[0].item())
                    text = fingering_highlight.text(fingers, score_index, task_env.logic.length, completed)
                    phase, music_index = phase_at(score_index, b.manifest['control_dt'], preparation_timing)
                    if phase == 'preparation':
                        text = (f'Preparing to play {score_index*b.manifest["control_dt"]:.2f}/'
                                f'{preparation_timing["policy_preparation_seconds"]:.2f}s\n' + text)
                    if text != last_fingering_text:
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                          mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
                        last_fingering_text = text
                    if last_fingers is None or not np.array_equal(fingers, last_fingers):
                        print('Fingering instruction:\n' + text, flush=True)
                        last_fingers = fingers.copy()
                if scene.arm_follower is not None and not is_preparing:
                    lines = ['Arm request -> executed FK (not a reachability proof):']
                    for side in ('left', 'right'):
                        diagnostic = scene.arm_follower.last.get(side)
                        if not diagnostic:
                            continue
                        position_mm = float(diagnostic['position_error'][0].item()) * 1000.
                        orientation_mrad = float(diagnostic['orientation_error'][0].item()) * 1000.
                        status = 'tracking' if position_mm <= .1 and orientation_mrad <= 1. else 'request not reached yet'
                        lines.append(f'{side}: {position_mm:.2f} mm, {orientation_mrad:.2f} mrad ({status})')
                    arm_text = '\n'.join(lines)
                    if arm_text != last_arm_text and (tick % 10 == 0 or last_arm_text is None):
                        print(arm_text, flush=True)
                        last_arm_text = arm_text
                    text = (text + '\n' if text else '') + arm_text
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
                with viewer.lock():
                    data.qpos[:] = snapshot.qpos
                    data.qvel[:] = snapshot.qvel
                    if scene.model.nmocap:
                        data.mocap_pos[:] = snapshot.mocap_pos
                        data.mocap_quat[:] = snapshot.mocap_quat
                    if not events['paused'] or is_preparing:
                        data.ctrl[:] = snapshot.ctrl
                    mujoco.mj_forward(scene.model, data)  # rendering derived quantities only
                    viewer.user_scn.ngeom = 0
                    key_highlight.append(viewer.user_scn, data)
                    if fingering_highlight is not None and not is_preparing:
                        fingering_highlight.append(viewer.user_scn, data, fingers)
                time.sleep(max(0, b.manifest['control_dt']-(time.monotonic()-begin)))
    finally:
        if panel is not None:
            panel.close()
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError('MuJoCo render thread did not close within 10 seconds')
    print('MJLab preview closed cleanly.', flush=True)


if __name__ == '__main__':
    main()
