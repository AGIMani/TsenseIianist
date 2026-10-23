"""Render a verified recorded trajectory with CURRENT assets, without policy/physics stepping.

Historical config/XML are read only as data for provenance and key-channel limits.
They are never passed to model builders or checkpoint loaders.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def load_inputs(bundle_path, directory):
    import numpy as np
    directory = Path(directory)
    with zipfile.ZipFile(bundle_path) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt ZIP')
        manifest = json.loads(z.read('manifest.json'))
        for name, expected in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError('Bundle hash mismatch: '+name)
        meta = json.loads(z.read('metadata.json'))
        with np.load(io.BytesIO(z.read('trajectory_200Hz.npz')), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    for name, field in [('physics_trace.npz', 'physics_trace_sha256'),
                        ('original_config.json', 'config_sha256'), ('original_scene.xml', 'scene_xml_sha256')]:
        if sha(directory/name) != meta[field]:
            raise ValueError('Historical data differs from trajectory provenance: '+name)
    config = json.loads((directory/'original_config.json').read_text())
    report = json.loads((directory/'original_report.json').read_text())
    if report['checkpoint_sha256'] != meta['checkpoint_sha256'] or report['source_runtime'] != config['source_runtime']:
        raise ValueError('Original recording/config mismatch')
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as data:
        trace = {k: data[k] for k in data.files}
    time = arrays['time_s']; dt = float(trace['physics_dt'])
    if not np.allclose(time, np.arange(len(time))*dt, atol=1e-10, rtol=0):
        raise ValueError('Invalid timestamps')
    q = np.concatenate([trace['initial_qpos'][None], trace['qpos']])
    v = np.concatenate([np.full_like(trace['qvel'][:1], np.nan), trace['qvel']])
    if len(q) != len(time) or not np.isfinite(q).all():
        raise ValueError('Invalid or incomplete original state trace')
    xml = ET.parse(directory/'original_scene.xml').getroot()
    joints = list(xml.find('worldbody').iter('joint'))
    if len(joints) != q.shape[1] or any(j.get('type', 'hinge') not in ('hinge', 'slide') for j in joints):
        raise ValueError('Expected scalar joint mapping')
    for side in ('left', 'right'):
        channels = [c for c in meta['channel_mapping'] if c['side'] == side]
        names = [c['name'] for c in channels]
        if names != arrays[side+'_joint_names'].tolist() or len(names) != 23:
            raise ValueError('Invalid hand channels')
        for c in channels:
            if joints[c['qpos_index']].get('name') != c['name']:
                raise ValueError('Joint index/name mismatch')
        for kind, full, index in [('position', q, 'qpos_index'), ('velocity', v, 'qvel_index')]:
            if not np.array_equal(full[:, [c[index] for c in channels]], arrays[side+'_joint_'+kind], equal_nan=True):
                raise ValueError('Original trace does not match every supplied hand sample')
        T = arrays['T_piano_'+side+'_hand']
        if T.shape != (len(time), 4, 4) or not np.isfinite(T).all():
            raise ValueError('Invalid hand transforms')
        if not np.allclose(T[:, 3], [0, 0, 0, 1], atol=1e-10):
            raise ValueError('Invalid homogeneous transforms')
        R = T[:, :3, :3]
        if not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-10) or not np.allclose(np.linalg.det(R), 1, atol=1e-10):
            raise ValueError('Invalid rotations')
    defaults = {d.get('class'): d.find('joint') for d in xml.iter('default') if d.find('joint') is not None}
    key_ids, limits, names = [], [], []
    for index, joint in enumerate(joints):
        match = re.fullmatch(r'piano/(white|black)_joint_(\d+)', joint.get('name', ''))
        if match:
            if int(match[2]) != len(key_ids):
                raise ValueError('Piano key order mismatch')
            key_ids.append(index); names.append(joint.get('name'))
            limits.append(np.fromstring(joint.get('range', defaults[joint.get('class')].get('range')), sep=' '))
    if len(key_ids) != 88:
        raise ValueError('Missing recorded piano channels')
    # Use only unchanged, hash-verified CURRENT score/threshold readers.
    references = ['tsensei_v3_2_pianist/key_activation.py', 'tsensei_v3_2_pianist/preview_keys.py', 'tsensei_v3_2_pianist/music/library.py', 'tsensei_v3_2_pianist/music/midi_file.py',
                  'tsensei_v3_2_pianist/models/piano/piano.py',
                  'tsensei_v3_2_pianist/music/data/rousseau/twinkle-twinkle-trimmed.mid']
    expected = config['source_runtime']['source_sha256']
    for name in references:
        if sha(ROOT/name) != expected[name]:
            raise ValueError('Cannot verify original score/threshold from current source: '+name)
    if config['source_runtime']['environment']['midi'] != 'TwinkleTwinkleRousseau':
        raise ValueError('This trajectory importer currently supports verified TwinkleTwinkleRousseau only')
    from tsensei_v3_2_pianist.music.library import twinkle_twinkle_rousseau
    midi = twinkle_twinkle_rousseau(); raw = midi.seq
    notes = [[n.start_time, n.end_time, n.pitch, n.part] for n in raw.notes]
    sequence = midi.trim_silence().seq
    offset = float(meta['music_start_seconds'])
    if not np.isclose(offset, report['music_start_policy_seconds']+report['preparation_prefix_seconds']):
        raise ValueError('Music offset mismatch')
    events = np.asarray([[n.start_time+offset, n.end_time+offset, n.pitch, n.part] for n in sequence.notes])
    shift = raw.notes[0].start_time
    original_notes = sorted((n.pitch, n.part, n.start_time-shift, n.end_time-shift) for n in raw.notes)
    normalized_notes = sorted((n.pitch, n.part, n.start_time, n.end_time) for n in sequence.notes)
    if len(original_notes) != len(normalized_notes) or not np.allclose(original_notes, normalized_notes, atol=1e-10, rtol=0):
        raise ValueError('Score normalization changed original notes')
    return arrays, dict(metadata=meta, dt=dt, key_qpos=q[:, key_ids], key_limits=np.asarray(limits),
                        key_names=names, sustain=trace['sustain'], events=events, original_events=np.asarray(notes),
                        music_offset=offset, original_midi_start=raw.notes[0].start_time,
                        original_report=report, reference_sha256={n: expected[n] for n in references})


def export_comparison(output, arrays, source):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from plot_recorded_midi_comparison import intervals, draw
    from tsensei_v3_2_pianist.preview_keys import KEY_ACTIVATION_FRACTION
    time = arrays['time_s'][1:]; q = source['key_qpos'][1:]; limits = source['key_limits']
    actual = np.clip(q, limits[:, 0], limits[:, 1]) >= limits[:, 1]*KEY_ACTIVATION_FRACTION
    expected = np.zeros_like(actual)
    for start, stop, pitch, finger in source['events']:
        expected[(time >= start-1e-10) & (time < stop-1e-10), int(pitch)-21] = True
    tp = int((actual & expected).sum()); fp = int((actual & ~expected).sum()); fn = int((~actual & expected).sum())
    metrics = dict(tp=tp, fp=fp, fn=fn, precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                   physical_sample_micro_f1=2*tp/max(2*tp+fp+fn, 1), sample_hz=1/source['dt'],
                   scope='Original single physical rollout, full duration; not a current-model physics evaluation')
    sets = dict(original=source['events'][:, :3].tolist(), actual=intervals(actual, time, time[-1]),
                correct=intervals(actual & expected, time, time[-1]), extra=intervals(actual & ~expected, time, time[-1]),
                missing=intervals(~actual & expected, time, time[-1]))
    for name, rows in sets.items():
        np.savetxt(output/(name+'_events.csv'), np.asarray(rows).reshape(-1, 3), delimiter=',',
                   header='start_s,end_s,midi_pitch', comments='')
    np.savetxt(output/'original_midi_absolute_events.csv', source['original_events'], delimiter=',',
               header='original_start_s,original_end_s,midi_pitch,finger', comments='')
    np.savez_compressed(output/'key_comparison.npz', time=time, actual=actual, original=expected,
                        key_qpos=q, key_limits=limits, key_activation_fraction=KEY_ACTIVATION_FRACTION, physics_dt=source['dt'],
                        reference_events=source['events'])
    fig, axes = plt.subplots(3, 1, figsize=(17, 9), sharex=True, sharey=True)
    draw(axes[0], sets['original'], '#2563eb'); draw(axes[1], sets['correct'], '#159f73')
    draw(axes[1], sets['extra'], '#ea6d24'); draw(axes[2], sets['missing'], '#617fe5'); draw(axes[2], sets['extra'], '#ea6d24')
    pitches = [r[2] for r in sets['original']+sets['actual']]
    for ax, title in zip(axes, ['Original MIDI (continuous note times)', 'Original rollout: recorded actual keys (200 Hz)', 'Missing / extra activation vs MIDI']):
        ax.set_title(title, loc='left'); ax.set_ylabel('MIDI pitch'); ax.set_xlim(0, time[-1])
        ax.set_ylim(min(pitches)-2, max(pitches)+2); ax.axvspan(0, source['music_offset'], color='gray', alpha=.15); ax.grid(alpha=.2)
    axes[1].legend(handles=[Patch(color='#159f73', label='Correct'), Patch(color='#ea6d24', label='Extra / early / late')])
    axes[-1].set_xlabel(f'Replay time (s): original MIDI time - {source["original_midi_start"]:.10f} + {source["music_offset"]:g}; final display hold excluded')
    fig.suptitle('best19000 | checkpoint '+source['metadata']['checkpoint_sha256'][:12]+
                 f' | original single-rollout F1={metrics["physical_sample_micro_f1"]:.4f}\nCurrent-model kinematic visualization only; these keys/metrics belong to the original recording')
    fig.tight_layout(); fig.savefig(output/'piano_roll_comparison.png', dpi=160); plt.close(fig)
    save(output/'comparison.json', dict(**metrics, music_offset_seconds=source['music_offset'],
         original_midi_start_seconds=source['original_midi_start'], interval_resolution_seconds=source['dt'],
         final_sample_right_censored=True, references=source['reference_sha256']))
    return metrics


def run(plan):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU export requires scripts/hpc/submit.sh')
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_bundle_from_config
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight, KEY_ACTIVATION_FRACTION
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight
    from tsensei_v3_2_pianist.preview_audio import events_from_states, write_midi, PianoSynth
    from export_video_job import visual_corners, camera_for
    output = Path(plan['output']); inputs = output/'inputs'
    for name, digest in plan['input_sha256'].items():
        if sha(inputs/name) != digest:
            raise ValueError('Input changed after submission: '+name)
    arrays, source = load_inputs(inputs/'trajectory_bundle.zip', inputs)
    save(output/'status.json', dict(state='building_current_scene', job_id=os.environ['SLURM_JOB_ID']))
    recipe = make_recipe(collision_profile='convex_hull_v1', drive_profile='shadow_v1',
                         control_hz=20, physics_hz=1/source['dt'], initial_buffer_time=0, release_tail_time=.25,
                         score_validation='strict_v1', task_profile='tsensei_collision_v1', exact_note_timing=True,
                         preparation_time=1, preparation_lift=.025, preparation_mode='policy_v1', wrist_pitch_range_scale=.8)
    config = dict(kind='current_model_recorded_trajectory_visualization', source_runtime=recipe,
                  arm_follow=None, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'])
    save(output/'current_render_config.json', config)
    bundle, scene, _ = build_bundle_from_config(config); model = bundle.model; data = bundle.reset_cpu()
    key_overlay = ActiveKeyHighlight(model)
    if [model.joint(int(j)).name for j in bundle.key_ids] != source['key_names']:
        raise ValueError('Current piano key names differ from recorded key order')
    if not np.allclose(model.jnt_range[bundle.key_ids], source['key_limits'], atol=1e-7, rtol=0):
        raise ValueError('Current piano key limits differ from original recording')
    ids, body_ids, mocap_ids = {}, {}, {}
    for side in ('left', 'right'):
        ids[side] = np.array([model.joint(n).id for n in arrays[side+'_joint_names']])
        body_ids[side] = model.body(side+'_/base_link').id
        mocap_ids[side] = model.body_mocapid[model.body('preparation_'+side+'_mount').id]
    def body_transform(bid):
        T = np.eye(4); T[:3, :3] = data.xmat[bid].reshape(3, 3); T[:3, 3] = data.xpos[bid]; return T
    def set_sample(i):
        data.mocap_pos[:] = 0; data.mocap_quat[:] = [1, 0, 0, 0]
        for side in ids:
            data.qpos[model.jnt_qposadr[ids[side]]] = arrays[side+'_joint_position'][i]
        data.qpos[bundle.key_qadr] = source['key_qpos'][i]
        data.time = float(arrays['time_s'][i]); mujoco.mj_kinematics(model, data)
        piano = body_transform(model.body('piano/').id)
        for side, mid in mocap_ids.items():
            desired = piano @ arrays['T_piano_'+side+'_hand'][i]
            delta = desired @ np.linalg.inv(body_transform(body_ids[side]))
            data.mocap_pos[mid] = delta[:3, 3]
            data.mocap_quat[mid] = np.roll(Rotation.from_matrix(delta[:3, :3]).as_quat(), 1)
        mujoco.mj_forward(model, data)
        return max(float(np.max(np.abs(np.linalg.inv(piano) @ body_transform(body_ids[s])-arrays['T_piano_'+s+'_hand'][i]))) for s in ids)
    errors = []; framing = []
    for i in range(len(arrays['time_s'])):
        errors.append(set_sample(i))
        if i % 50 == 0: framing.append(visual_corners(model, data))
    if max(errors) > 1e-8:
        raise ValueError('Rendered hand-root transforms do not preserve input poses')
    np.savez_compressed(output/'pose_validation.npz', time_s=arrays['time_s'], max_transform_error=errors)
    metrics = export_comparison(output, arrays, source)
    fps = plan['fps']; end = float(arrays['time_s'][-1]); count = round((end+plan['hold'])*fps)+1
    events = events_from_states(source['key_qpos'][1:], source['key_limits'], source['sustain'], source['dt'], KEY_ACTIVATION_FRACTION)
    save(output/'played_events.json', events); write_midi(output/'played.mid', events)
    synth = PianoSynth()
    try: audio = synth.render(events, count/fps, output/'played.wav')
    finally: synth.close()
    width, height = 1280, 720
    model.vis.global_.offwidth=max(width, model.vis.global_.offwidth); model.vis.global_.offheight=max(height, model.vis.global_.offheight)
    cameras = [camera_for(view, model, False, {}, width, height, .96, np.concatenate(framing), 76, 36) for view in ('front', 'top')]
    tips = [s+'_/'+n+'_tip' for s in ('right', 'left') for n in ('thumb', 'index', 'mid', 'ring', 'pinky')]
    fingers = FingeringHighlight(model, dict(fingertip_sites=tips), key_overlay)
    opt = mujoco.MjvOption(); opt.geomgroup[2:] = 0; opt.sitegroup[:] = 0
    silent = output/'silent.mp4'; video = output/'trajectory_current_model.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{width*2}x{height}', '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(silent)], stdin=subprocess.PIPE)
    save(output/'status.json', dict(state='rendering', job_id=os.environ['SLURM_JOB_ID'], frames=count))
    try:
        with mujoco.Renderer(model, height=height, width=width, max_geom=6000) as renderer:
            for frame in range(count):
                i = min(round(frame/fps/source['dt']), len(arrays['time_s'])-1); at = arrays['time_s'][i]; set_sample(i)
                assigned = np.zeros((10, 88))
                for start, stop, pitch, finger in source['events']:
                    if start <= at < stop: assigned[int(finger), int(pitch)-21] = 1
                canvas = Image.new('RGB', (width*2, height))
                for panel, (camera, settings) in enumerate(cameras):
                    model.vis.global_.orthographic = 1; model.vis.global_.fovy = settings['ortho_height_m']
                    renderer.update_scene(data, camera=camera, scene_option=opt)
                    key_overlay.color_render_scene(renderer.scene, data); fingers.append(renderer.scene, data, assigned)
                    pic = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(pic)
                    draw.rectangle((0, 0, width, 76), fill='#121923')
                    draw.text((12, 7), f'CURRENT A3.2 | {settings["view"]} | recorded trajectory | t={at:.3f}s | 1x', fill='white', font_size=23)
                    draw.text((12, 38), 'Original recorded keys/audio; no new physics evaluation | best19000', fill='#c9d4e3', font_size=20)
                    draw.rectangle((0, height-36, width, height), fill='#121923')
                    draw.text((12, height-29), 'R/L 1 thumb  2 index  3 middle  4 ring  5 little | green: recorded key activation', fill='white', font_size=19)
                    canvas.paste(pic, (panel*width, 0))
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if frame in (0, 120, 260): canvas.save(output/f'frame_{frame:04d}.png')
                if frame % 80 == 0: print(f'Rendered {frame}/{count}', flush=True)
    finally:
        encoder.stdin.close(); code=encoder.wait()
    if code: raise RuntimeError('Video encoding failed')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(silent), '-i', str(output/'played.wav'),
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', str(count/fps), '-movflags', '+faststart', str(video)], check=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
    if shutil.which('ffprobe'):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
        if {s['codec_type'] for s in probe['streams']} != {'video', 'audio'}:
            raise ValueError('Missing audio or video stream')
    else:
        # Some user-local FFmpeg bundles have no ffprobe. Explicit stream maps
        # make a missing video/audio stream fatal instead of rerendering a valid file.
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(video),
                        '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
        probe = dict(validation='FFmpeg full decode with required video and audio stream maps',
                     ffprobe_available=False, encoder_frames=count, encoder_fps=fps)
    if audio['rms'] <= 0:
        raise ValueError('Audio/video validation failed')
    silent.unlink(); save(output/'ffprobe.json', probe)
    provenance = dict(job_id=os.environ['SLURM_JOB_ID'], source_bundle=source['metadata'], input_sha256=plan['input_sha256'],
        scope=plan['scope'], model='current convex_hull_v1 A3.2; trimmed palm; no arms',
        original_checkpoint_loaded=False, policy_rerun=False, physics_steps=0,
        joint_mapping='same named 46 scalar joint positions, without clipping or hardware remapping',
        hand_pose_max_matrix_error=max(errors), verified_pose_samples=len(errors), original_sample_hz=1/source['dt'],
        frames=count, fps=fps, video_seconds=count/fps, recorded_duration=end, final_hold_seconds=plan['hold'],
        metrics=metrics, audio=audio, cameras=[c[1] for c in cameras], decoded=True,
        limitations=['Current geometry differs from original model; finger endpoints/contact appearance may differ.',
                     'Green keys, audio and F1 are from the hash-matched original physical recording, not current-model contacts.',
                     'Finger colors use original continuous note times; they are score annotations, not inferred finger contacts.'])
    job_source = ROOT.parent/'source.tar.gz'
    if job_source.is_file(): provenance['current_render_source_archive_sha256'] = sha(job_source)
    save(output/'report.json', provenance)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Trajectory playback</title><h1>当前模型：原轨迹逐帧回放</h1><p>当前A3.2外观；按键、声音及F1来自匹配的原始物理记录，不是当前模型重新演奏。</p><video controls width="1200" src="trajectory_current_model.mp4"></video><p><a href="report.json">回放报告</a></p><img width="1200" src="piano_roll_comparison.png">')
    save(output/'status.json', dict(state='completed', job_id=os.environ['SLURM_JOB_ID'], metrics=metrics))
    save(output/'manifest.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--validate':
        arrays, source = load_inputs(Path(sys.argv[2]), Path(sys.argv[3]))
        print(json.dumps(dict(valid=True, samples=len(arrays['time_s']), original_notes=len(source['events']),
                             checkpoint_sha256=source['metadata']['checkpoint_sha256'])))
    else:
        raise SystemExit('Use export_video.sh --trajectory-bundle ... or --validate ZIP DATA_DIR')
