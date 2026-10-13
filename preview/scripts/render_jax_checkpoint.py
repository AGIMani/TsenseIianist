#!/usr/bin/env python3
"""Render the best logged JAX checkpoint, with verified historical source recovery.

Run with training/.venv/bin/python. This does not load Torch .pt checkpoints.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'tsensei_v3_2_pianist'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def historical_package(contract):
    """Recover only known edits; require exact saved hashes before any import."""
    with tempfile.TemporaryDirectory(prefix='v32-jax-replay-') as directory:
        target = Path(directory) / PACKAGE
        shutil.copytree(ROOT / PACKAGE, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        restored = []
        for name, expected in contract['source_sha256'].items():
            path = target / name
            if digest(path) == expected:
                continue
            source = path.read_text()
            if name == 'models/hands/tsensei_hand_constants.py':
                source = source.replace('(-0.025, 0)', '(-0.04, 0)')
            elif name == 'suite/tasks/base.py':
                source = source.replace('_PHYSICS_TIMESTEP = 0.0005', '_PHYSICS_TIMESTEP = 0.002')
                start = source.index('        # Harden the piano keys.')
                end = source.index('        self.set_timesteps(', start)
                source = source[:start] + '''        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
        self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

''' + source[end:]
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                raise ValueError(f'Cannot reconstruct saved source: {name}; provide its historical version')
            path.write_text(source)
            restored.append(name)
        sys.path.insert(0, directory)
        try:
            yield restored
        finally:
            sys.path.remove(directory)


def select_checkpoint(requested, best):
    rows = [json.loads(line) for line in (requested.parent / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    available = [r for r in rows if 'eval' in r and (requested.parent / f"step-{r['step']}.msgpack").is_file()]
    if best:
        if not available:
            raise ValueError('No saved checkpoint has logged evaluation metrics')
        row = max(available, key=lambda r: (r['eval']['f1'], r['eval']['return']))
        return requested.parent / f"step-{row['step']}.msgpack", row
    row = next((r for r in available if requested.name == f"step-{r['step']}.msgpack"), None)
    return requested, row


def render(args, chosen, logged, config, restored):
    from tsensei_v3_2_pianist.preview_backend import configure_backend, current_gl_info
    configure_backend(headless=True)
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.rl import checkpoint
    from tsensei_v3_2_pianist.rl.environment import EnvironmentConfig, make_environment
    from tsensei_v3_2_pianist.rl.sac import SAC, SACConfig
    from tsensei_v3_2_pianist.rl.specs import EnvironmentSpec
    from tsensei_v3_2_pianist.rl.train import binary_metrics
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight, FINGER_COLORS, FINGER_NAMES, finger_label, note_name

    seed = config['cli']['seed']
    env, raw, contract = make_environment(EnvironmentConfig(**config['contract']['environment']), seed + 1)
    try:
        if contract != config['contract']:
            raise ValueError('Restored runtime contract differs from training config')
        agent = SAC.initialize(EnvironmentSpec.make(env), SACConfig(**config['sac_config']), seed, config['discount'])
        agent, metadata = checkpoint.load(chosen, agent, config)
        ts = env.reset()
        task = raw.task
        m = raw.physics.model.ptr
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
        m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (.03, 0, .06)
        camera.distance, camera.azimuth, camera.elevation = 1.05, 155, -50
        options = mujoco.MjvOption()
        options.label = mujoco.mjtLabel.mjLABEL_SELECTION
        keys = ActiveKeyHighlight(m)
        fingers = FingeringHighlight(m, {'fingertip_sites': [s.full_identifier for h in (task.right_hand, task.left_hand) for s in h.fingertip_sites]}, keys)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 25)
        fps = round(1 / raw.control_timestep())
        args.output.mkdir(parents=True, exist_ok=False)
        video = args.output / 'best_replay.mp4'
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', str(fps),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
        actions, positions, goals, actuals, rewards, metrics = [], [], [], [], [], []
        sustain_metrics = []
        frames_written = 0
        try:
            with mujoco.Renderer(m, height=720, width=1280, max_geom=4000) as renderer:
                graphics = current_gl_info()
                print('Renderer:', graphics, flush=True)
                for index in range(args.max_steps):
                    scored_index = task._t_idx
                    action = np.asarray(agent.eval_actions(ts.observation))
                    ts = env.step(action)
                    actual = np.asarray(task.piano.activation, dtype=bool).copy()
                    goal = np.asarray(task._goal_current[:-1], dtype=bool).copy()
                    actions.append(action)
                    positions.append(raw.physics.data.qpos.copy())
                    goals.append(goal)
                    actuals.append(actual)
                    rewards.append(float(ts.reward))
                    metrics.append(binary_metrics(goal, actual))
                    sustain_metrics.append(binary_metrics(np.asarray(task._goal_current[-1:], dtype=bool), np.asarray(task.piano.sustain_activation, dtype=bool)))
                    assigned = np.zeros((10, 88))
                    for note in task._notes[scored_index]:
                        if 0 <= note.fingering < 10:
                            assigned[note.fingering, note.key] = 1
                    data = raw.physics.data.ptr
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    keys.color_render_scene(renderer.scene, data)
                    fingers.append(renderer.scene, data, assigned)
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, 1280, 100), fill=(18, 23, 32))
                    draw.text((20, 10), ('Twinkle Twinkle | V3.2 | best saved JAX policy' if args.best and config['contract']['environment']['midi'] == 'TwinkleTwinkleRousseau' else f"{config['contract']['environment']['midi']} | V3.2 | JAX policy"), font=title_font, fill='white')
                    draw.text((20, 48), f'{chosen.name}    t={(index+1)/fps:.2f}s    frame {index+1}/{len(task._notes)}    mean key F1={np.mean(metrics,axis=0)[2]:.4f}', font=font, fill='white')
                    draw.text((20, 75), 'Deterministic replay | original training physics | green = pressed (includes wrong keys)', font=font, fill=(180, 230, 195))
                    draw.rectangle((0, 614, 1280, 720), fill=(18, 23, 32))
                    for finger in range(5):
                        color = tuple(int(v*255) for v in FINGER_COLORS[finger, :3])
                        draw.text((20+245*finger, 622), f'{finger+1}: {FINGER_NAMES[finger]}', font=font, fill=color)
                    target_text = '  '.join(f'{finger_label(f)}:{note_name(k)}' for f,k in zip(*np.nonzero(assigned))) or 'rest'
                    draw.text((20, 651), f'Scored target / fingering: {target_text}', font=font, fill='white')
                    draw.text((20, 679), 'Pressed: ' + (', '.join(note_name(k) for k in np.flatnonzero(actual)) or 'none'), font=font, fill=(90, 235, 100))
                    frame = np.asarray(img).tobytes()
                    encoder.stdin.write(frame)
                    frames_written += 1
                    if index == len(task._notes)//2:
                        img.save(args.output / 'preview.png')
                    if ts.last():
                        for _ in range(fps):  # One second hold; no extra simulation.
                            encoder.stdin.write(frame)
                            frames_written += 1
                        break
                    if (index+1) % 40 == 0:
                        print(f'Replayed {index+1}/{len(task._notes)} frames', flush=True)
        finally:
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f'ffmpeg exited with {code}')
        if not ts.last():
            raise RuntimeError('Replay capped before episode ended; video is incomplete')
        result = dict(zip(('precision', 'recall', 'f1'), map(float, np.mean(metrics, axis=0))))
        result.update(zip(('sustain_precision', 'sustain_recall', 'sustain_f1'), map(float, np.mean(sustain_metrics, axis=0))))
        g, a = np.asarray(goals), np.asarray(actuals)
        tp, fp, fn = int((g&a).sum()), int((~g&a).sum()), int((g&~a).sum())
        result.update({'return': sum(rewards), 'length': len(actions), 'key_frame_micro_f1': 2*tp/max(2*tp+fp+fn,1)})
        report = {'requested_checkpoint': str(args.checkpoint), 'selected_checkpoint': str(chosen),
            'checkpoint_sha256': digest(chosen), 'selection': 'highest logged evaluation key F1, return breaks ties' if args.best else 'explicit checkpoint',
            'logged_evaluation': logged, 'replay': result, 'evaluation_seed': seed+1,
            'logged_metrics_match': bool(logged and all(np.isclose(result[k], logged['eval'][k], rtol=1e-6, atol=1e-6) for k in result)),
            'restored_sources_exact_sha256': restored, 'full_runtime_contract_verified': True,
            'physics_dt': m.opt.timestep, 'control_dt': raw.control_timestep(),
            'graphics': graphics, 'fps': fps, 'video_frames': frames_written, 'video_seconds': frames_written/fps,
            'final_hold_seconds': 1, 'audio': False, 'policy': 'deterministic tanh(mean)',
            'video': video.name, 'contract': contract}
        np.savez_compressed(args.output/'rollout.npz', actions=actions, qpos=positions, goals=goals, actual=actuals, rewards=rewards)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k,v in report.items() if k != 'contract'}, indent=2), flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--best', action='store_true', help='Select best saved checkpoint in the same run by logged key F1')
    p.add_argument('--output', type=Path, required=True, help='New directory; never overwrite an existing replay')
    p.add_argument('--max-steps', type=int, default=20000)
    args = p.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file():
        p.error('checkpoint does not exist')
    if shutil.which('ffmpeg') is None:
        p.error('ffmpeg is required')
    chosen, logged = select_checkpoint(args.checkpoint, args.best)
    config = json.loads((chosen.parent/'config.json').read_text())
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    with historical_package(config['contract']) as restored:
        render(args, chosen, logged, config, restored)


if __name__ == '__main__':
    main()
