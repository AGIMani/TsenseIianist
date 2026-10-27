#!/usr/bin/env python3
"""Package an existing offline IK physics trace without loading or simulating a model.

Run with a Python environment containing numpy, matplotlib, pretty_midi and
note-seq. Audio uses the project's existing FluidSynth runtime. Model loading
and optional local OpenGL acceptance remain separate, protected launcher steps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_interactive_replay import install_local_launchers, install_replay_display
from tsensei_v3_2_pianist.key_activation import key_active, validate_fraction


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_score(score, original_midi):
    """Verify every note's attributes independently of the recorded target masks."""
    import pretty_midi
    # The generated protobuf suffices; importing note_seq itself loads audio/ML.
    path = distribution('note-seq').locate_file('note_seq/protobuf/music_pb2.py')
    spec = importlib.util.spec_from_file_location('planned_replay_music_pb2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence = module.NoteSequence.FromString(Path(score).read_bytes())
    midi = pretty_midi.PrettyMIDI(str(original_midi))
    raw = sorted((float(n.start), float(n.end), int(n.pitch), int(n.velocity), i, int(inst.program), int(inst.is_drum))
                 for i, inst in enumerate(midi.instruments) for n in inst.notes)
    assigned = sorted((n.start_time, n.end_time, n.pitch, n.velocity,
                       n.instrument, n.program, int(n.is_drum)) for n in sequence.notes)
    if not raw or len(raw) != len(assigned) or not np.allclose(raw, assigned, rtol=0, atol=1e-9):
        raise ValueError('Score differs from original MIDI note times/pitches/velocities/instruments/programs')
    events = sorted([float(n.start_time), float(n.end_time), int(n.pitch)-21, int(n.part)]
                    for n in sequence.notes)
    values = np.asarray(events)
    if (not np.isfinite(values).all() or np.any(values[:, 0] < 0)
            or np.any(values[:, 1] <= values[:, 0]) or np.any((values[:, 2] < 0) | (values[:, 2] > 87))
            or np.any((values[:, 3] < 0) | (values[:, 3] > 9))):
        raise ValueError('Score requires valid piano notes and fingers 0–9')
    return dict(events=events, original_note_attributes=raw, note_count=len(raw),
                original_midi_sha256=sha256(original_midi), score_sha256=sha256(score))


def recorded_physics_dt(time, declared=None):
    """Validate the full grid without multiplying median-difference roundoff."""
    time=np.asarray(time,dtype=float)
    if time.ndim!=1 or len(time)<2 or not np.isfinite(time).all():
        raise ValueError('Expected at least two finite physical timestamps')
    dt=float(declared) if declared is not None else float(time[-1]/(len(time)-1))
    if time[0]!=0 or not np.isfinite(dt) or dt<=0 or not np.allclose(time,np.arange(len(time))*dt,rtol=0,atol=1e-9):
        raise ValueError('Viewer requires uniform physics samples beginning at original MIDI t=0')
    return dt


def validate_trace(trace):
    """Array-only validation. Physical activation is recomputed from real key angles."""
    required = ('time', 'qpos', 'qvel', 'ctrl', 'key_qpos', 'key_limits',
                'key_activation_fraction', 'actual_keys', 'target_keys', 'assignment', 'original_events')
    missing = set(required) - trace.keys()
    if missing:
        raise ValueError(f'Missing physical trace fields: {sorted(missing)}')
    time = np.asarray(trace['time'], dtype=float)
    dt = recorded_physics_dt(time,trace.get('physics_dt'))
    for key in ('qpos', 'qvel', 'ctrl', 'key_qpos'):
        values = np.asarray(trace[key])
        if values.ndim != 2 or len(values) != len(time) or not np.isfinite(values).all():
            raise ValueError(f'Invalid recorded {key}')
    if trace['key_qpos'].shape != (len(time), 88) or trace['key_limits'].shape != (88, 2):
        raise ValueError('Expected 88 physical piano keys')
    fraction = validate_fraction(float(trace['key_activation_fraction']))
    actual = key_active(trace['key_qpos'], trace['key_limits'], fraction)
    if not np.array_equal(actual, trace['actual_keys']):
        raise ValueError('actual_keys differs from recorded key_qpos at the physical activation threshold')
    if 'key_qpos_addresses' in trace:
        addresses = np.asarray(trace['key_qpos_addresses'])
        if addresses.shape != (88,) or not np.issubdtype(addresses.dtype, np.integer):
            raise ValueError('Invalid key_qpos_addresses')
        if not np.array_equal(trace['qpos'][:, addresses], trace['key_qpos']):
            raise ValueError('key_qpos differs from full recorded qpos')
    events = np.asarray(trace['original_events'], dtype=float)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or np.any(events[:, 0] < 0)
            or np.any(events[:, 1] <= events[:, 0])
            or np.any(events[:, 2:] != np.floor(events[:, 2:]))
            or np.any((events[:, 2] < 0) | (events[:, 2] > 87))
            or np.any((events[:, 3] < 0) | (events[:, 3] > 9))):
        raise ValueError('Invalid original_events')
    target = np.zeros_like(actual)
    assignment = np.zeros((len(time), 10, 88), dtype=bool)
    for start, end, key, finger in events:
        # Match the planner's 0.1 ns floating-point tolerance at exact boundaries;
        # this does not quantize notes onto a control or physics grid.
        active = (time >= start-1e-10) & (time < end-1e-10)
        target[active, int(key)] = True
        assignment[active, int(finger), int(key)] = True
    for name, expected in (('target_keys', target), ('assignment', assignment)):
        if not np.array_equal(trace[name], expected):
            raise ValueError(f'{name} differs from exact original note timing/fingering')
    tp = int(np.count_nonzero(actual & target))
    fp = int(np.count_nonzero(actual & ~target))
    fn = int(np.count_nonzero(~actual & target))
    return dict(physics_dt=dt, duration_s=float(time[-1]), actual=actual, target=target,
                assignment=assignment, key_activation_fraction=fraction,
                metrics=dict(tp=tp, fp=fp, fn=fn, physical_sample_micro_f1=2*tp/max(2*tp+fp+fn, 1)))


def actual_intervals(actual, time):
    """Hold each recorded state until the next timestamp, never beyond the replay."""
    events = []
    for key in range(88):
        changes = np.diff(np.r_[False, actual[:, key], False].astype(np.int8))
        for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            start, end = float(time[first]), float(time[min(last, len(time)-1)])
            if end > start:
                events.append([start, end, key+21])
    return events


def audio_events(actual, time):
    """Actual edges at saved times, including t=0; fixed uncalibrated velocity 127."""
    events, previous = [], np.zeros(88, dtype=bool)
    for at, active in zip(time, actual):
        for key in np.flatnonzero(previous & ~active):
            events.append(dict(time=float(at), type='note_off', note=int(key)+21))
        for key in np.flatnonzero(active & ~previous):
            events.append(dict(time=float(at), type='note_on', note=int(key)+21, velocity=127))
        previous = active
    for key in np.flatnonzero(previous):
        events.append(dict(time=float(time[-1]), type='note_off', note=int(key)+21, end_of_replay=True))
    return events


def viewer_annotations(time, assignment):
    # The shared viewer uses searchsorted(score.time, at - 1e-9), whereas physics
    # uses floor lookup. Encode each state's interval end, just below that lookup
    # epsilon, so exact physical boundaries select the *new* assignment too.
    # No 1 ms/5 ms frame lead is introduced; numerical transition tolerance is 1 ns.
    changes = np.r_[0, np.flatnonzero(np.any(assignment[1:] != assignment[:-1], axis=(1, 2)))+1]
    endpoints = np.r_[np.nextafter(np.asarray(time)[changes[1:]]-1e-9, -np.inf), time[-1]]
    return dict(time=endpoints, fingers=np.asarray(assignment[changes], dtype=np.uint8))


def render_audio(events, duration, path):
    from tsensei_v3_2_pianist.preview_audio import PianoSynth
    synth = PianoSynth()
    try:
        report = synth.render(events, duration, path)
    finally:
        synth.close()
    report['source'] = 'same recorded physical key activation edges; no score notes or inferred pedal'
    return report


def plot_comparison(output, trace, checked, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    def draw(axis, events, color):
        polygons = [[(s, p-.4), (e, p-.4), (e, p+.4), (s, p+.4)] for s, e, p in events]
        axis.add_collection(PolyCollection(polygons, facecolors=color, edgecolors='none'))

    fig, axes = plt.subplots(2, 1, figsize=(17, 7), sharex=True, sharey=True, layout='constrained')
    original = [row[:3] for row in comparison['original_events']]
    draw(axes[0], original, '#2475bd')
    time = trace['time']
    actual, target = checked['actual'], checked['target']
    for mask, color in ((actual & target, '#26956b'), (actual & ~target, '#da4b48'),
                        (~actual & target, '#edc76a')):
        draw(axes[1], actual_intervals(mask, time), color)
    axes[0].set_title('Original MIDI: unmodified note onset/end times', loc='left')
    axes[1].set_title('Recorded physical activation: correct / extra / missed keys', loc='left')
    visible = [p for s, e, p in original if s <= time[-1] and e >= time[0]]
    visible += [p for _, _, p in comparison['actual_events']]
    lo, hi = (max(21, min(visible)-2), min(108, max(visible)+2)) if visible else (21, 108)
    for axis in axes:
        axis.set_xlim(time[0], time[-1]); axis.set_ylim(lo-.5, hi+.5)
        axis.set_ylabel('MIDI pitch'); axis.grid(axis='x', alpha=.15)
    axes[1].legend(handles=[Patch(color=c, label=t) for c, t in
                           [('#26956b', 'correct'), ('#da4b48', 'extra'), ('#edc76a', 'missed')]], loc='upper right')
    axes[1].set_xlabel('Original MIDI time (seconds); preparation excluded; no time offset or playback speed change')
    fig.suptitle(f'Offline planned controller — no trained checkpoint | physical sample micro-F1 '
                 f'{checked["metrics"]["physical_sample_micro_f1"]:.5f} | {1/checked["physics_dt"]:g} Hz\n'
                 f'{"Full piece" if comparison["full_piece"] else "Partial preview"} · '
                 f'trace SHA256 {comparison["source_trace_sha256"][:16]} · no dynamic safety claim')
    fig.savefig(output/'actual_vs_original_midi.png', dpi=160)
    plt.close(fig)


def package(recording, original_midi, score, output):
    recording, original_midi, score, output = (Path(p).resolve() for p in (recording, original_midi, score, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite replay output: {output}')
    verified = verify_score(score, original_midi)
    report_path, trace_path, model_path = (recording/n for n in ('replay_report.json', 'physical_trace.npz', 'model.mjb'))
    report = json.loads(report_path.read_text())
    source = report['source']
    for name in ('original_midi_sha256', 'score_sha256'):
        if source.get(name) != verified[name]:
            raise ValueError(f'Recording {name} differs from provided source')
    model_hash = sha256(model_path)
    if report.get('model_sha256', source.get('model_sha256')) != model_hash:
        raise ValueError('Recorded model_sha256 differs from model.mjb')
    archive_hash = source.get('archive_sha256')
    if not isinstance(archive_hash, str) or len(archive_hash) != 64 or any(c not in '0123456789abcdef' for c in archive_hash):
        raise ValueError('Recording requires verified source archive_sha256')
    versions = report.get('dependency_versions', {})
    if not versions.get('mujoco'):
        raise ValueError('Recording requires dependency_versions.mujoco')
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    checked = validate_trace(trace)
    if not np.isclose(float(report['duration_s']), checked['duration_s'], rtol=0, atol=1e-9):
        raise ValueError('Recording report duration differs from physical trace')
    if not np.isclose(float(report['physics_hz']), 1/checked['physics_dt'], rtol=1e-10, atol=1e-8):
        raise ValueError('Recording report physics_hz differs from physical trace')
    ordered = np.asarray(sorted(trace['original_events'].tolist()))
    if ordered.shape != np.asarray(verified['events']).shape or not np.allclose(ordered, verified['events'], rtol=0, atol=1e-9):
        raise ValueError('Recorded original_events differs from verified score/MIDI')
    if report.get('preparation_prefix_seconds', 0) != 0:
        raise ValueError('Planned trace must retain original MIDI origin; separate preparation is unsupported')
    for name in ('tp', 'fp', 'fn'):
        if report.get(name) != checked['metrics'][name]:
            raise ValueError(f'Recording report {name} differs from physical trace')
    if not np.isclose(report['original_midi_micro_f1'], checked['metrics']['physical_sample_micro_f1'], rtol=0, atol=1e-12):
        raise ValueError('Recording report F1 differs from physical trace')
    for name, width in (('mocap_pos', 3), ('mocap_quat', 4)):
        if name not in trace:
            if report.get('model_nmocap') != 0:
                raise ValueError(f'Missing {name}; cannot infer model mocap states')
            trace[name] = np.zeros((len(trace['time']), 0, width))
        value = trace[name]
        if value.ndim != 3 or value.shape[0] != len(trace['time']) or value.shape[2] != width or not np.isfinite(value).all():
            raise ValueError(f'Invalid recorded {name}')
    if trace['mocap_pos'].shape[1] != trace['mocap_quat'].shape[1]:
        raise ValueError('Mocap position/quaternion counts differ')
    if 'model_nmocap' in report and report['model_nmocap'] != trace['mocap_pos'].shape[1]:
        raise ValueError('Recorded model_nmocap differs from trace')

    output.mkdir(parents=True)
    inputs = output/'inputs'; inputs.mkdir()
    for src, name in ((original_midi, 'original.mid'), (score, 'score.proto'),
                      (report_path, 'replay_report.json'), (trace_path, 'physical_trace.npz')):
        shutil.copy2(src, inputs/name)
    for name in ('arguments.json', 'source_config.json', 'model_diagnostic.json',
                 'planner_source.py', 'contact_planner_source.py', 'geometry_helper.py', 'process.json'):
        if (recording/name).is_file():
            shutil.copy2(recording/name, inputs/name)
    shutil.copy2(model_path, output/'model.mjb')
    viewer_trace = dict(physics_dt=checked['physics_dt'])
    for key in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat'):
        viewer_trace['initial_'+key] = trace[key][0]
        viewer_trace[key] = trace[key][1:]
    np.savez_compressed(output/'physics_trace.npz', **viewer_trace)
    np.savez_compressed(output/'score_annotations.npz', **viewer_annotations(trace['time'], checked['assignment']))
    edges = audio_events(checked['actual'], trace['time'])
    audio = render_audio(edges, checked['duration_s'], output/'played.wav')
    with wave.open(str(output/'played.wav'), 'rb') as wav:
        audio_duration = wav.getnframes()/wav.getframerate()
        if abs(audio_duration-checked['duration_s']) > 1/wav.getframerate():
            raise ValueError('Rendered audio duration differs from recorded physics')
    write_json(output/'audio_events.json', edges)
    comparison = dict(original_events=[[s, e, int(k)+21, int(f)] for s, e, k, f in verified['events']],
                      actual_events=actual_intervals(checked['actual'], trace['time']),
                      original_note_attributes=verified['original_note_attributes'],
                      source_trace_sha256=sha256(trace_path), original_midi_sha256=verified['original_midi_sha256'],
                      score_sha256=verified['score_sha256'], sample_hz=1/checked['physics_dt'],
                      time_origin='original MIDI t=0; negative preparation excluded',
                      preparation_prefix_seconds=0, excluded_preparation_seconds=report.get('excluded_preparation_seconds'),
                      duration_s=checked['duration_s'], full_piece=bool(checked['duration_s'] >= max(e[1] for e in verified['events'])),
                      note_boundary_numerical_tolerance_seconds=1e-10,
                      metrics=checked['metrics'], checkpoint=None,
                      terminal_sample='terminal instantaneous state included in sample metrics; intervals end at last timestamp')
    write_json(output/'comparison_events.json', comparison)
    np.savez_compressed(output/'key_comparison.npz', time=trace['time'], actual=checked['actual'],
                        original=checked['target'], original_events=trace['original_events'],
                        key_qpos=trace['key_qpos'], key_limits=trace['key_limits'],
                        key_activation_fraction=checked['key_activation_fraction'], physics_dt=checked['physics_dt'])
    plot_comparison(output, trace, checked, comparison)
    meta = dict(display_contract='piano_key_colors_only_v1',
                controller='offline IK physical recording with recorded joint-limit handling', checkpoint=None,
                robot_state_projection=report.get('robot_state_projection','not reported'),
                physical_control=report.get('physical_control','not reported'),
                archive_sha256=archive_hash, native_arms=False, dependency_versions=versions,
                preparation_prefix_seconds=0, key_activation_fraction=checked['key_activation_fraction'],
                source_recording=str(recording), physics_trace_sha256=sha256(output/'physics_trace.npz'),
                source_physics_trace_sha256=sha256(trace_path), model_sha256=model_hash,
                annotation_timing='interval-end lookup adapted to shared viewer, 1 ns boundary tolerance',
                model_validation='pending protected viewer --validate')
    write_json(output/'replay.json', meta)
    install_replay_display(output)
    for name in ('preview_keys.py', 'key_activation.py'):
        shutil.copy2(ROOT/'tsensei_v3_2_pianist'/name, output/name)
    (output/'replay.py').write_text('from viewer import main\nif __name__ == "__main__": main()\n')
    install_local_launchers(output)
    with (output/'README.txt').open('a') as readme:
        readme.write('\n本包为离线规划控制器的真实物理轨迹；没有训练检查点，也不代表动态安全验收。\n'
                     '保留原始 MIDI 时间，负时间准备段未录入；声音来自同次录制的实际琴键激活。\n'
                     'actual_vs_original_midi.png / comparison_events.json / key_comparison.npz 提供同轨迹对照。\n'
                     'report.json 保留低分、碰撞及原始验收结果；模型加载验收须另运行受保护的 --validate。\n')
        readme.write('机器人状态限位处理（原报告）：'+meta['robot_state_projection']+'。\n'
                     '物理控制（原报告）：'+meta['physical_control']+'。不将带机器人关节投影的录制称为无投影动力学。\n')
    result = dict(kind='planned_physical_recording_package_v1', checkpoint=None,
                  controller=meta['controller'], duration_s=checked['duration_s'],
                  robot_state_projection=meta['robot_state_projection'],physical_control=meta['physical_control'],
                  physical_hz=1/checked['physics_dt'], metrics=checked['metrics'],
                  source_report=report, source_inputs=dict(recording=str(recording), score=str(score), original_midi=str(original_midi)),
                  source_archive_sha256=archive_hash, audio=audio, floating_labels=False,
                  full_piece=comparison['full_piece'], model_validation=meta['model_validation'],
                  limitations='Offline planned physical preview, not a trained policy or dynamic safety guarantee; '
                              'raw collision/IK errors and acceptance status are retained in source_report. '
                              'Fixed note velocity 127 is not acoustically calibrated. Preparation before t=0 is excluded. '
                              'Robot-state projection convention: '+meta['robot_state_projection']+'.')
    write_json(output/'report.json', result)
    write_json(output/'manifest.json', dict(kind='planned_replay_sha256_v1',
               files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'original-midi', 'score', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = package(args.recording, args.original_midi, args.score, args.output)
    print(json.dumps({key: result[key] for key in ('kind', 'duration_s', 'metrics', 'model_validation')}, indent=2))


if __name__ == '__main__':
    main()
