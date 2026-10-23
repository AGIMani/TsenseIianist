#!/usr/bin/env python3
"""Preview the fixed raised-to-origin pose path without a learned policy.

Run with training_mjlab/.venv/bin/python scripts/preview_preparation.py
--output preview/outputs/preparation/NAME. No training is started.
This fixed_v1 demonstration does not validate policy_v1 preparation actions.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'preview/scripts'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--lift', type=float, default=.02, help='World-Z initial lift in meters')
    parser.add_argument('--height', type=float, default=.10, help='Unraised origin mounting height')
    parser.add_argument('--ty', type=float, default=.04)
    parser.add_argument('--yaw', type=float, default=.15)
    parser.add_argument('--camera-view', choices=('front', 'performer', 'overview'), default='front')
    parser.add_argument('--camera-framing', choices=('full', 'hands'), default='hands')
    parser.add_argument('--audio', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing artifacts are preserved')
    if shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required')
    from tsensei_v3_2_pianist.preview_backend import configure_backend, current_gl_info
    configure_backend(headless=True)
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from preparation_frames import frame_indices, snapshot, audio_prefix
    recipe = make_recipe(height=args.height, ty=args.ty, yaw=args.yaw,
                         control_hz=20., physics_hz=200., initial_buffer_time=0.,
                         task_profile='tsensei_collision_v1', collision_profile='convex_hull_v1',
                         drive_profile='shadow_v1', preparation_time=args.duration,
                         preparation_lift=args.lift, preparation_mode='fixed_v1')
    scene, _, _ = build(recipe)
    bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True,
                         preparation=recipe['preparation'])
    trace = bundle.preparation_trace
    if trace is None:
        parser.error('A positive preparation duration and lift are required for this preview')
    model = bundle.model
    dt = bundle.manifest['control_dt']
    fps = 1 / dt
    camera = mujoco.MjvCamera()
    camera_settings = configure_camera(camera, model, view=args.camera_view, framing=args.camera_framing)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    keys = ActiveKeyHighlight(model)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'preparation.npz', **trace)
    (args.output/'origin_initial.json').write_text(json.dumps(bundle.preparation, indent=2)+'\n')
    (args.output/'source_recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
    video = args.output/('silent.mp4' if args.audio else 'preparation.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    count = 0
    try:
        with mujoco.Renderer(model, height=720, width=1280, max_geom=4000) as renderer:
            graphics = current_gl_info()
            for index in frame_indices(trace, dt, include_endpoint=True):
                data = snapshot(model, trace, index)
                renderer.update_scene(data, camera=camera)
                keys.color_render_scene(renderer.scene, data)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.rectangle((0,0,1280,115), fill=(18,23,32))
                draw.text((20,10), 'V3.2 | preparation motion preview', font=font, fill='white')
                draw.text((20,46), f't={data.time:.2f}/{trace["time"][-1]:.2f}s | world-Z lift +{args.lift*100:.1f} cm -> origin', font=font, fill='white')
                draw.text((20,82), 'Fixed hand pose; learned policy is not running', font=font, fill=(180,230,195))
                frame = np.asarray(img).tobytes()
                encoder.stdin.write(frame)
                count += 1
                if index == 0:
                    img.save(args.output/'initial.png')
            img.save(args.output/'origin.png')
            for _ in range(round(fps)):
                encoder.stdin.write(frame)
                count += 1
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited with {code}')
    audio_report = None
    key_joints = [model.joint(name).id for name in bundle.manifest['piano_joints']]
    key_qadr = model.jnt_qposadr[key_joints]
    q, pedal = audio_prefix(bundle, key_qadr)
    if args.audio:
        from tsensei_v3_2_pianist.preview_audio import PianoSynth, events_from_states, write_midi
        events = events_from_states(q, model.jnt_range[key_joints], pedal,
                                   bundle.manifest['physics_dt'], bundle.manifest['key_activation_fraction'])
        (args.output/'audio_events.json').write_text(json.dumps(events, indent=2)+'\n')
        write_midi(args.output/'played.mid', events)
        synth = PianoSynth()
        try:
            audio_report = synth.render(events, count/fps, args.output/'played.wav')
        finally:
            synth.close()
        final_video = args.output/'preparation.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(video),
                        '-i', str(args.output/'played.wav'), '-map', '0:v:0', '-map', '1:a:0',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', str(final_video)], check=True)
        video.unlink()
        video = final_video
    report = dict(scope='fixed_v1 pose-path demonstration only; does not validate learned policy_v1 preparation',
                  preparation=bundle.preparation, graphics=graphics, camera=camera_settings,
                  video=video.name, fps=fps, frames=count, video_seconds=count/fps,
                  preparation_seconds=float(trace['time'][-1]),
                  key_activation_count=int(trace['key_activation'].sum()), audio=audio_report)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'preparation'}, indent=2))


if __name__ == '__main__':
    main()
