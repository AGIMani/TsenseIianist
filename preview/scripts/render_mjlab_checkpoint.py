#!/usr/bin/env python3
"""Render one complete Torch SAC episode using independent MJLab GPU physics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _accumulate_note_on_phase_metrics(events, metrics, phases, targets, control_dt,
                                      prefix_seconds, timing, *, exact_events=None):
    """Assign physical key edges to their real phase; retain old grid semantics."""
    import numpy as np
    from preparation_frames import phase_at

    for event in events:
        if event['type'] != 'note_on':
            continue
        score_time = event['time'] - prefix_seconds
        if exact_events is not None:
            if not 0 <= score_time <= len(phases) * control_dt + 1e-10:
                continue
            # An edge at .9975 s is still preparation even if its control
            # endpoint at 1.000 s is classified as music. The epsilon only
            # absorbs floating-point arithmetic at exact score boundaries.
            phase, _ = phase_at((score_time + 1e-10) / control_dt, control_dt, timing)
            row = metrics.setdefault(phase, dict(frames=0, mean_frame_f1=None,
                key_frame_micro_f1=None, wrong_key_frames=0, frames_with_wrong_keys=0,
                episode_return=0., note_on_events=0, wrong_note_on_events=0))
            expected = ((exact_events[:, 0] <= score_time + 1e-10)
                        & (exact_events[:, 1] > score_time + 1e-10)
                        & (exact_events[:, 2] == event['note'] - 21)).any()
        else:
            # Historical key edges belong to the interval that applied their
            # grid target, including that interval's final physical sample.
            policy_index = int(np.floor((score_time - 1e-9) / control_dt))
            if not 0 <= policy_index < len(phases):
                continue
            row = metrics[str(phases[policy_index])]
            expected = targets[policy_index, event['note'] - 21]
        row['note_on_events'] += 1
        row['wrong_note_on_events'] += int(not expected)


def main():
    if "--recording" in sys.argv[1:]:
        from render_recorded_native import main as render_recorded
        return render_recorded()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--scene', type=Path, default=None)
    p.add_argument('--task', type=Path, default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--camera-view', choices=('front','performer','overview'), default='front')
    p.add_argument('--camera-framing', choices=('full','hands'), default='full')
    p.add_argument('--camera-distance', type=float)
    p.add_argument('--camera-elevation', type=float)
    p.add_argument('--audio', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--soundfont', type=Path)
    p.add_argument('--finger-labels', action='store_true', help='Add floating text; off to avoid covering fingers')
    p.add_argument('--output', required=True, type=Path, help='New output directory')
    args = p.parse_args()
    # Select EGL before source reconstruction imports MuJoCo.
    from tsensei_v3_2_pianist.preview_backend import configure_backend, current_gl_info
    configure_backend(headless=True)
    from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
    args.scene, args.task = for_checkpoint(args.checkpoint, args.scene, args.task)
    if args.output.exists():
        p.error('Output directory already exists; choose a new directory')
    if not args.checkpoint.is_file() or shutil.which('ffmpeg') is None:
        p.error('An existing checkpoint and ffmpeg are required')
    import mujoco
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent, reward_options
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight, FINGER_COLORS, FINGER_NAMES, finger_label, note_name
    from preparation_frames import (frame_indices, snapshot as preparation_snapshot,
                                    audio_prefix, timeline, phase_at)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    options = reward_options(args.checkpoint)
    env = PianoTaskEnv(args.scene, args.task, device=args.device, **options)
    agent = load_agent(args.checkpoint, scene_sha256=sha256(args.scene/'manifest.json'),
                       task_sha256=sha256(args.task/'task.json'), obs_size=env.observation_size,
                       action_size=env.action_size, device=args.device,
                       action_contract=env.scene.action_contract,
                       control_limits_contract=env.scene.control_limits,
                       finger_clearance_contract=env.scene.finger_clearance,
                       finger_coupling_contract=env.scene.finger_coupling,
                       coupling_observation_contract=env.coupling_observation_contract,
                       arm_follow_contract=(env.scene.arm_follower.contract if env.scene.arm_follower is not None else None),
                       preparation_contract=env.scene.bundle.preparation, **options)
    obs = env.reset()
    audio_synth = None
    recorded_substeps = 0
    if args.audio:
        from tsensei_v3_2_pianist.preview_audio import PianoSynth, events_from_states, write_midi
        audio_synth = PianoSynth(args.soundfont)
        count = env.logic.length * env.scene.decimation
        audio_keys = torch.empty((count,88),device=env.device)
        audio_pedal = torch.empty(count,device=env.device)
        def record_audio(scene):
            nonlocal recorded_substeps
            audio_keys[recorded_substeps].copy_(scene.data.qpos[0,scene.key_qadr])
            audio_pedal[recorded_substeps].copy_(scene.sustain[0])
            recorded_substeps += 1
        # Reuse the existing read-only substep hook on this playback instance.
        # No changes to task code, physics integrator, or the running trainer.
        from functools import partial
        env.scene.step = partial(env.scene.step, substep_callback=record_audio)
    model = env.scene.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    camera = mujoco.MjvCamera()
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    camera_settings = configure_camera(camera,model,view=args.camera_view,framing=args.camera_framing,
                                       distance=args.camera_distance,elevation=args.camera_elevation)
    if 'm6' in env.scene.bundle.manifest and args.camera_framing == 'full':
        camera_settings = configure_camera(camera, model, view=args.camera_view, framing='full',
            distance=args.camera_distance if args.camera_distance is not None else 1.65,
            elevation=args.camera_elevation if args.camera_elevation is not None else -25)
        camera.lookat[:] = (.22, 0, .28)
        camera_settings['lookat'] = camera.lookat.tolist()
    keys = ActiveKeyHighlight(model)
    fingering = FingeringHighlight(model, env.bundle.manifest, keys)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 25)
    dt = env.bundle.manifest['control_dt']
    preparation_timing = timeline(env.scene.bundle, env.bundle.manifest)
    preparation_trace = preparation_timing['prefix_trace']
    preparation_seconds = preparation_timing['prefix_seconds']
    fps = 1 / dt
    hold_frames = round(fps)
    args.output.mkdir(parents=True, exist_ok=False)
    if preparation_trace is not None:
        np.savez_compressed(args.output/'preparation.npz', **preparation_trace)
    if env.scene.bundle.preparation is not None:
        (args.output/'preparation.json').write_text(json.dumps(env.scene.bundle.preparation, indent=2)+'\n')
    video = args.output/('silent.mp4' if args.audio else 'replay.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    trajectory = {k: [] for k in ('actions', 'qpos', 'ctrl', 'goals', 'actual', 'rewards',
                                  'reward_terms', 'phase', 'music_index', 'policy_time')}
    if env.exact_score is not None:
        trajectory.update(goals_at_control_start=[], goals_substeps=[])
    if model.nmocap:
        trajectory.update(mocap_pos=[],mocap_quat=[])
    clearance_fields = ('finger_projection_mask', 'finger_fallback_mask', 'finger_actual_deficit_m',
                        'finger_requested_deficit_m', 'finger_applied_deficit_m') if env.scene.finger_clearance_guard is not None else ()
    trajectory.update({key: [] for key in clearance_fields})
    key_contact_tracker = getattr(env.scene, 'key_contact_diagnostics', None)
    key_contact_fields = key_contact_tracker.COLUMNS if key_contact_tracker is not None else ()
    trajectory.update({key: [] for key in key_contact_fields})
    frames = 0
    try:
        with mujoco.Renderer(model, height=720, width=1280, max_geom=4000) as renderer:
            graphics = current_gl_info()
            print('Renderer:', graphics, flush=True)
            if preparation_trace is not None:
                for trace_index in frame_indices(preparation_trace, dt):
                    data = preparation_snapshot(model, preparation_trace, trace_index)
                    renderer.update_scene(data, camera=camera)
                    keys.color_render_scene(renderer.scene, data)
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0,0,1280,100), fill=(18,23,32))
                    draw.text((20,10), 'V3.2 | preparing to play', font=title, fill='white')
                    draw.text((20,48), f't={data.time:.2f}/{preparation_seconds:.2f}s | raised initial -> origin | score paused', font=font, fill='white')
                    draw.text((20,75), 'Returning smoothly to the saved starting pose', font=font, fill=(180,230,195))
                    encoder.stdin.write(np.asarray(img).tobytes())
                    frames += 1
            # Fixed initialization gets a prefix; policy preparation starts at
            # t=0 in the raised GPU reset state and gets no additional prefix.
            initial = env.scene.cpu_snapshot()
            renderer.update_scene(initial,camera=camera)
            initial_img = Image.fromarray(renderer.render())
            initial_draw = ImageDraw.Draw(initial_img)
            initial_draw.rectangle((0,0,1280,100),fill=(18,23,32))
            initial_draw.text((20,10),f'V3.2 | MJLab + Torch SAC | {args.checkpoint.name}',font=title,fill='white')
            initial_label = 'raised starting pose' if preparation_timing['policy_active'] else 'starting pose'
            initial_draw.text((20,48),f't={preparation_seconds:.2f}s | {initial_label} | audio follows actual key strikes',font=font,fill='white')
            encoder.stdin.write(np.asarray(initial_img).tobytes())
            frames += 1
            for index in range(env.logic.length):
                action = agent.act(obs, deterministic=True)
                result = env.step(action)
                obs = result.observation
                # CPU data is a render-only snapshot; physics stays on CUDA.
                data = env.scene.cpu_snapshot()
                info = result.info
                scored_index = int(info['score_index'][0].item())
                phase, music_index = phase_at(
                    scored_index + int(env.exact_score is not None), dt, preparation_timing)
                assigned = env.bundle.fingers[scored_index]
                active = info['key_activation'][0].cpu().numpy()
                values = {'actions': action[0].cpu().numpy(), 'qpos': data.qpos.copy(),
                          'ctrl': data.ctrl.copy(), 'goals': info['goal_applied'][0].cpu().numpy(),
                          'actual': active, 'rewards': result.reward[0].item(),
                          'reward_terms': [info['reward_terms'][k][0].item() for k in env.reward_term_names],
                          'phase': phase, 'music_index': music_index, 'policy_time': (scored_index+1)*dt}
                if model.nmocap:
                    values.update(mocap_pos=data.mocap_pos.copy(),mocap_quat=data.mocap_quat.copy())
                if env.exact_score is not None:
                    values.update(goals_at_control_start=info['goal_at_control_start'][0].cpu().numpy(),
                                  goals_substeps=info['goal_substeps'][0].cpu().numpy())
                values.update({key: info[key][0].cpu().numpy().copy() for key in clearance_fields})
                values.update({key: int(info[key][0].item()) for key in key_contact_fields})
                for key, value in values.items():
                    trajectory[key].append(value)
                renderer.update_scene(data, camera=camera)
                keys.color_render_scene(renderer.scene, data)
                before_annotations = renderer.scene.ngeom
                fingering.append(renderer.scene, data, assigned)
                if not args.finger_labels:
                    for geom in renderer.scene.geoms[before_annotations:renderer.scene.ngeom]:
                        geom.label = ''
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 1280, 100), fill=(18, 23, 32))
                draw.text((20, 10), f'V3.2 | MJLab + Torch SAC | {args.checkpoint.name}', font=title, fill='white')
                phase_label = (f'preparing {(index+1)*dt:.2f}/{preparation_timing["policy_preparation_seconds"]:.2f}s'
                               if phase == 'preparation' else 'waiting' if phase == 'buffer' else
                               'releasing keys' if phase == 'release_tail' else
                               f'playing {(music_index+1)*dt:.2f}s')
                draw.text((20, 48), f't={preparation_seconds+(index+1)*dt:.2f}s | {phase_label} | frame {index+1}/{env.logic.length} | F1={info["episode_metrics"][0,2].item():.4f}', font=font, fill='white')
                draw.text((20, 75), 'Deterministic GPU replay | green = pressed (includes wrong keys)', font=font, fill=(180, 230, 195))
                draw.rectangle((0, 614, 1280, 720), fill=(18, 23, 32))
                for finger in range(5):
                    draw.text((20+245*finger, 622), f'{finger+1}: {FINGER_NAMES[finger]}', font=font,
                              fill=tuple(int(v*255) for v in FINGER_COLORS[finger,:3]))
                target = '  '.join(f'{finger_label(f)}:{note_name(k)}' for f,k in zip(*np.nonzero(assigned))) or 'rest'
                draw.text((20, 651), f'Scored target / fingering: {target}', font=font, fill='white')
                draw.text((20, 679), 'Pressed: '+(', '.join(note_name(k) for k in np.flatnonzero(active)) or 'none'), font=font, fill=(90,235,100))
                frame = np.asarray(img).tobytes()
                encoder.stdin.write(frame)
                frames += 1
                if index == env.logic.length//3:
                    img.save(args.output/'preview.png')
                if result.done[0].item():
                    for _ in range(hold_frames):
                        encoder.stdin.write(frame)
                        frames += 1
                    break
                if (index+1) % 40 == 0:
                    print(f'Replayed {index+1}/{env.logic.length}', flush=True)
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code or not result.terminated[0].item():
        raise RuntimeError(f'Replay incomplete or encoding failed: ffmpeg exit {code}')
    audio_report = False
    if audio_synth is not None:
        try:
            q = audio_keys[:recorded_substeps].cpu().numpy()
            pedal = audio_pedal[:recorded_substeps].cpu().numpy()
            prep_q, prep_pedal = audio_prefix(env.scene.bundle, env.scene.key_qadr.cpu().numpy())
            preparation_substeps = len(prep_q)
            q = np.concatenate((prep_q, q), axis=0)
            pedal = np.concatenate((prep_pedal, pedal))
            physics_dt = env.scene.bundle.manifest['physics_dt']
            events = events_from_states(q,env.scene.key_limits.cpu().numpy(),pedal,physics_dt,
                                       env.scene.bundle.manifest['key_threshold'])
            (args.output/'audio_events.json').write_text(json.dumps(events,indent=2)+'\n')
            write_midi(args.output/'played.mid',events)
            audio_report = audio_synth.render(events,frames/fps,args.output/'played.wav')
            audio_report.update(physics_dt=physics_dt,recorded_substeps=len(q),
                                policy_recorded_substeps=recorded_substeps,
                                preparation_recorded_substeps=preparation_substeps,
                                preparation_prefix_substeps=preparation_substeps,
                                policy_preparation_recorded_substeps=min(recorded_substeps, round(
                                    preparation_timing['policy_preparation_seconds']/physics_dt)),
                                note_on_events=sum(e['type']=='note_on' for e in events),
                                score_used_for_audio=False,initial_video_frame=True)
            np.savez_compressed(args.output/'audio_substeps.npz', key_qpos=q, sustain=pedal,
                                physics_dt=physics_dt,
                                key_limits=env.scene.key_limits.cpu().numpy(),
                                key_threshold=env.scene.bundle.manifest['key_threshold'],
                                preparation_prefix_substeps=preparation_substeps)
        finally:
            audio_synth.close()
        final_video = args.output/'replay.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(video),
                        '-i',str(args.output/'played.wav'),'-map','0:v:0','-map','1:a:0',
                        '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',
                        str(final_video)],check=True)
        video.unlink()
        video = final_video
    names = ('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1')
    metrics = dict(zip(names, info['episode_metrics'][0].cpu().tolist()))
    tp, fp, fn = env.logic.key_counts[0].cpu().tolist()
    metrics.update({'return': info['episode_return'][0].item(), 'length': info['episode_steps'][0].item(),
                    'key_frame_micro_f1': 2*tp/max(2*tp+fp+fn,1)})
    # Report score performance separately: quiet policy preparation frames have
    # frame-F1=1, so the whole-episode mean is not directly comparable to a run
    # without those frames. Keep the learner's original aggregate unchanged.
    phase_metrics = {}
    phases = np.asarray(trajectory['phase'])
    actual = np.asarray(trajectory['actual'], dtype=bool)
    target = np.asarray(trajectory['goals'])[:, :88].astype(bool)
    tp_by_frame = (actual & target).sum(1)
    fp_by_frame = (actual & ~target).sum(1)
    fn_by_frame = (~actual & target).sum(1)
    denominator = 2*tp_by_frame + fp_by_frame + fn_by_frame
    frame_f1 = np.divide(2*tp_by_frame, denominator,
                         out=np.ones(len(phases), dtype=float), where=denominator > 0)
    for phase in ('preparation', 'buffer', 'music', 'release_tail'):
        selected = phases == phase
        if not selected.any():
            continue
        tp, fp, fn = (int(values[selected].sum()) for values in (tp_by_frame, fp_by_frame, fn_by_frame))
        phase_metrics[phase] = dict(frames=int(selected.sum()), mean_frame_f1=float(frame_f1[selected].mean()),
            key_frame_micro_f1=2*tp/max(2*tp+fp+fn, 1), wrong_key_frames=fp,
            frames_with_wrong_keys=int(np.count_nonzero(fp_by_frame[selected])),
            episode_return=float(np.asarray(trajectory['rewards'])[selected].sum()))
        phase_metrics[phase]['note_on_events'] = 0 if args.audio else None
        phase_metrics[phase]['wrong_note_on_events'] = 0 if args.audio else None
    if args.audio:
        _accumulate_note_on_phase_metrics(events, phase_metrics, phases, target,
            dt, preparation_seconds, preparation_timing,
            exact_events=(env.bundle.exact_note_events if env.exact_score is not None else None))
    logged = None
    log = args.checkpoint.parent/'metrics.jsonl'
    if log.is_file():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # Training may be appending its final line.
                continue
            if 'eval' in row and args.checkpoint.name == f"step-{row['vector_step']}.pt":
                logged = row
    report = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256(args.checkpoint),
              'reward_options': options,
              'note_timing_contract': env.note_timing_contract,
              'control_limits_contract': env.scene.control_limits,
              'finger_clearance_contract': env.scene.finger_clearance,
              'key_contact_diagnostics': ({
                  'contract': key_contact_tracker.contract,
                  'physics_sample_hz': 1/env.scene.bundle.manifest['physics_dt'],
                  **{key: int(np.sum(trajectory[key])) for key in key_contact_fields}}
                  if key_contact_fields else None),
              'finger_clearance_diagnostics': ({
                  'projection_frames': int(np.any(trajectory['finger_projection_mask'],axis=1).sum()),
                  'fallback_frames': int(np.any(trajectory['finger_fallback_mask'],axis=1).sum()),
                  **{key.replace('finger_', ''): float(np.max(trajectory[key])) for key in clearance_fields[2:]},
                  'sample_hz': fps} if clearance_fields else None),
              'scene_sha256': sha256(args.scene/'manifest.json'), 'task_sha256': sha256(args.task/'task.json'),
              'policy': 'deterministic tanh(mean) + profile shift',
              'observation_size': env.observation_size, 'action_size': env.action_size,
              'action_profile': env.scene.action_profile, 'metrics': metrics, 'logged_evaluation': logged,
              'logged_metrics_match': None if logged is None else all(np.isclose(v,logged['eval'][k],rtol=1e-5,atol=1e-6) for k,v in metrics.items()),
              'graphics': graphics, 'fps': fps, 'video_frames': frames, 'video_seconds': frames/fps,
              'final_hold_seconds': hold_frames/fps, 'initial_frame_seconds': dt,
              'preparation_seconds': preparation_seconds,
              'preparation_prefix_seconds': preparation_seconds,
              'policy_time_offset_seconds': preparation_seconds,
              'preparation_mode': preparation_timing['mode'],
              'policy_preparation_seconds': preparation_timing['policy_preparation_seconds'],
              'music_start_policy_seconds': preparation_timing['music_start_policy_seconds'],
              'music_end_step': env.bundle.music_end_step,
              'release_tail_steps': env.bundle.release_tail_steps,
              'release_tail_end_active_keys': int(actual[-1].sum()) if env.bundle.release_tail_steps else None,
              'phase_metrics': phase_metrics,
              'preparation': getattr(env.scene.bundle, 'preparation', None),
              'rollout_scope': ('all policy frames, including the learnable preparation phase'
                                if preparation_timing['policy_active'] else
                                'policy only; prescribed reset preparation is saved separately'),
              'audio': audio_report, 'camera': camera_settings,
              'reward_term_names': env.reward_term_names, 'video': video.name}
    metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['metadata']
    if 'source_runtime' in metadata:
        report.update(source_runtime=metadata['source_runtime'], no_arms=metadata.get('no_arms', False))
    np.savez_compressed(args.output/'rollout.npz', **trajectory)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
