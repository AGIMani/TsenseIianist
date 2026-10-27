#!/usr/bin/env python3
"""Package an existing preparation-only recording, without advancing physics.

The CLI loads the recorded MJB on the remote CPU to recover key addresses/ranges.
Pure validation helpers below are suitable for small-array local tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_planned_replay import (
    actual_intervals, audio_events, recorded_physics_dt, render_audio, sha256,
    verify_score, write_json,
)
from scripts.package_interactive_replay import install_local_launchers, install_replay_display
from tsensei_v3_2_pianist.key_activation import key_active, validate_fraction


def verify_source_archive(path, source_hashes, expected_sha256=None):
    """Check the recipe's recorded source against archive bytes, never extract it."""
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('Missing recorded source hashes')
    archive_hash = sha256(path)
    if expected_sha256 is not None and archive_hash != expected_sha256:
        raise ValueError('Source archive SHA differs from recorded provenance')
    remaining = dict(source_hashes)
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name.removeprefix('./')
            if name not in remaining:
                continue
            if not member.isfile():
                raise ValueError(f'Expected source file in archive: {name}')
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != remaining.pop(name):
                raise ValueError(f'Archive source differs from recorded recipe: {name}')
    if remaining:
        raise ValueError(f'Archive missing recorded source files: {sorted(remaining)[:5]}')
    return dict(archive_sha256=archive_hash, matched_source_files=len(source_hashes),
                recorded_archive_identity_present=expected_sha256 is not None)


def validate_preparation_trace(trace, key_addresses, key_limits, dt, fraction, model_shape):
    time = np.asarray(trace['time'], dtype=float)
    dt = recorded_physics_dt(time, dt)
    rows = len(time)
    shapes = dict(qpos=(rows, model_shape['nq']), qvel=(rows, model_shape['nv']),
                  ctrl=(rows, model_shape['nu']), mocap_pos=(rows, model_shape['nmocap'], 3),
                  mocap_quat=(rows, model_shape['nmocap'], 4))
    for key, shape in shapes.items():
        values = np.asarray(trace[key])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Recorded {key} differs from actual model shape {shape}')
    addresses = np.asarray(key_addresses)
    limits = np.asarray(key_limits, dtype=float)
    if (addresses.shape != (88,) or not np.issubdtype(addresses.dtype, np.integer)
            or len(np.unique(addresses)) != 88 or np.any(addresses < 0)
            or np.any(addresses >= model_shape['nq'])):
        raise ValueError('Invalid actual-model piano key addresses')
    if limits.shape != (88, 2) or not np.isfinite(limits).all():
        raise ValueError('Invalid actual-model piano key limits')
    fraction = validate_fraction(float(fraction))
    if fraction != 1.:
        raise ValueError('This preparation delivery requires 100% physical activation')
    keys = np.asarray(trace['qpos'])[:, addresses]
    actual = key_active(keys, limits, fraction)
    if not np.array_equal(actual, trace['key_activation']):
        raise ValueError('Saved key_activation differs from actual MJB key qpos/limits')
    return dict(time=time, duration_s=float(time[-1]), physics_dt=dt,
                key_qpos=keys, key_limits=limits, key_qpos_addresses=addresses,
                key_activation_fraction=fraction, actual=actual,
                activated_key_samples=int(np.count_nonzero(actual.any(axis=1))),
                activated_key_cell_count=int(np.count_nonzero(actual)))


def make_preparation_comparison(checked, original_events):
    events = np.asarray(original_events, dtype=float)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or np.any(events[:, 0] < 0)
            or np.any(events[:, 1] <= events[:, 0])):
        raise ValueError('Invalid original score events')
    end = checked['duration_s']
    original = [[float(s), float(e), int(k)+21, int(f)] for s, e, k, f in events]
    return dict(kind='preparation_only_comparison_v1', original_events=original,
                reference_events_on_playback_axis=[[s+end, e+end, k, f] for s, e, k, f in original],
                actual_events=actual_intervals(checked['actual'], checked['time']),
                recorded_interval_s=[0., end], original_midi_start_at_playback_s=end,
                preparation_duration_s=end, performance_recorded=False,
                metrics=None, physical_sample_micro_f1=None,
                metric_reason='Preparation precedes original MIDI t=0; no performance was recorded or scored.',
                time_origin='Playback 0 is preparation start; original MIDI t=0 maps to playback T.',
                actual_sampling_hz=1/checked['physics_dt'],
                context_after_recording='Original MIDI only, shaded as unrecorded; no actual states or audio fabricated.')


def write_preparation_audio(checked, path):
    edges = audio_events(checked['actual'], checked['time'])
    duration = checked['duration_s']
    if edges:
        result = render_audio(edges, duration, path)
        result.update(silent=False, event_count=len(edges))
    else:
        # With no actual key activation, emit exact PCM silence, not synthesized score audio.
        rate = 44100
        frames = round(duration*rate)
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(rate)
            stream.writeframes(bytes(frames*4))
        result = dict(silent=True, event_count=0, duration_s=frames/rate, sample_rate=rate,
                      rms=0., source='No actual key activation in this preparation; exact PCM silence.')
    with wave.open(str(path), 'rb') as stream:
        if abs(stream.getnframes()/stream.getframerate()-duration) > 1/stream.getframerate():
            raise ValueError('Preparation audio duration mismatch')
    return result


def plot_preparation_comparison(output, checked, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    end = checked['duration_s']
    context_end = end+1.
    original = [row for row in comparison['reference_events_on_playback_axis'] if row[0] < context_end]
    actual = comparison['actual_events']
    pitches = [row[2] for row in original+actual] or [21, 108]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True, layout='constrained')
    for axis, rows, color in ((axes[0], original, '#4282bb'), (axes[1], actual, '#259779')):
        axis.add_collection(PolyCollection([[(s, k-.4), (e, k-.4), (e, k+.4), (s, k+.4)]
                                            for s, e, k, *_ in rows], facecolors=color, edgecolors='none'))
        axis.axvspan(end, context_end, color='#b7bac0', alpha=.35)
        axis.axvline(end, color='#555555', linestyle='--')
        axis.set_xlim(0, context_end); axis.set_ylim(max(20, min(pitches)-2), min(109, max(pitches)+2))
        axis.set_ylabel('MIDI pitch'); axis.grid(axis='x', alpha=.15)
    axes[0].set_title('Original MIDI context begins after preparation; gray region is not recorded', loc='left')
    axes[1].set_title('Actual preparation key activation only; no performance data after T', loc='left')
    if not actual:
        axes[1].text(end/2, np.mean(axes[1].get_ylim()), 'No actual key activation — silent audio', ha='center')
    axes[1].set_xlabel(f'Playback seconds; preparation T={end:g}s; original MIDI t=0 at playback T')
    fig.suptitle(f'Preparation only | {1/checked["physics_dt"]:g} Hz physical samples | F1 not applicable\n'
                 'No checkpoint or policy; this clip does not validate performance from the new start pose')
    fig.savefig(output/'actual_vs_original_midi.png', dpi=160); plt.close(fig)


def package(recording, original_midi, score, source_archive, output, model_python=None):
    if not os.environ.get('SSH_CONNECTION'):
        raise RuntimeError('Preparation MJB validation and packaging must run on the remote CPU')
    recording, original_midi, score, source_archive, output = (
        Path(p).resolve() for p in (recording, original_midi, score, source_archive, output))
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((recording/'report.json').read_text())
    manifest = json.loads((recording/'manifest.json').read_text())
    recipe = json.loads((recording/'source_recipe.json').read_text())
    verified = verify_score(score, original_midi)
    if recipe['external_score']['sha256'] != verified['score_sha256']:
        raise ValueError('Recorded source recipe differs from supplied score')
    model_path, trace_path = recording/'model.mjb', recording/'preparation.npz'
    if report['model_sha256'] != sha256(model_path) or report['trace_sha256'] != sha256(trace_path):
        raise ValueError('Preparation model/trace SHA differs from source report')
    with np.load(recording/'original_score.npz', allow_pickle=False) as data:
        events = np.asarray(sorted(data['events'].tolist()))
    if events.shape != np.asarray(verified['events']).shape or not np.allclose(events, verified['events'], rtol=0, atol=1e-9):
        raise ValueError('Recorded original score differs from original MIDI/score')
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {k: data[k] for k in data.files}
    # Composer owns MIDI/plot dependencies, while the recorded MJB requires the
    # learner's MuJoCo version. Keep both environments intact and load only in
    # the explicitly selected model process, without mj_step or XML compilation.
    model_python = Path(model_python or ROOT/'training_mjlab/.venv/bin/python').absolute()
    probe = '''import json,sys,mujoco
from pathlib import Path
if mujoco.__version__ != sys.argv[3]: raise RuntimeError('Recorded MJB version mismatch')
model=mujoco.MjModel.from_binary_path(sys.argv[1])
manifest=json.loads(Path(sys.argv[2]).read_text())
joints=[model.joint(name).id for name in manifest['piano_joints']]
print(json.dumps(dict(shape={k:int(getattr(model,k)) for k in ('nq','nv','nu','nmocap')},
    key_addresses=model.jnt_qposadr[joints].tolist(),key_limits=model.jnt_range[joints].tolist(),
    physics_dt=float(model.opt.timestep),mujoco_version=mujoco.__version__)))
'''
    model_command = [str(model_python), '-c', probe, str(model_path), str(recording/'manifest.json'), report['mujoco_version']]
    model_result = subprocess.run(model_command, check=True, text=True, capture_output=True)
    contract = json.loads(model_result.stdout)
    shape = contract['shape']
    checked = validate_preparation_trace(trace, np.asarray(contract['key_addresses']),
        np.asarray(contract['key_limits']), contract['physics_dt'], manifest['key_activation_fraction'], shape)
    if not np.isclose(report['preparation']['duration'], checked['duration_s'], rtol=0, atol=1e-9):
        raise ValueError('Preparation report duration differs from trace')
    if report['preparation']['report']['activated_key_samples'] != checked['activated_key_samples']:
        raise ValueError('Preparation report activation count differs from trace')
    source_check = verify_source_archive(source_archive, recipe['source_sha256'], report.get('source_archive_sha256'))
    archive_hash = source_check['archive_sha256']
    output.mkdir(parents=True)
    inputs = output/'inputs'; inputs.mkdir()
    for name in ('report.json', 'manifest.json', 'source_recipe.json', 'config.json', 'preparation.npz', 'original_score.npz'):
        shutil.copy2(recording/name, inputs/name)
    for name in ('measurements.npz', 'coordinate_audit.json', 'run.json', 'exit_status'):
        if (recording/name).is_file():
            shutil.copy2(recording/name, inputs/name)
    shutil.copy2(original_midi, inputs/'original.mid'); shutil.copy2(score, inputs/'score.proto')
    shutil.copy2(model_path, output/'model.mjb')
    shutil.copy2(__file__, inputs/'package_preparation_replay.py')
    write_json(inputs/'model_cpu_validation.json', dict(command=model_command, exit_code=model_result.returncode,
        stderr=model_result.stderr, contract=contract, model_sha256=report['model_sha256'],
        scope='CPU binary model load and key mapping only; no physics steps, rendering, or XML compilation.'))
    for name in ('package_planned_replay.py', 'package_interactive_replay.py'):
        shutil.copy2(ROOT/'scripts'/name, inputs/name)
    write_json(inputs/'source_archive_reference.json', dict(path=str(source_archive), sha256=archive_hash,
        verification=source_check,
        note='Source archive reference; packaging does not modify or merge this archive.'))
    viewer_trace = dict(physics_dt=checked['physics_dt'])
    for key in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat'):
        viewer_trace['initial_'+key] = trace[key][0]; viewer_trace[key] = trace[key][1:]
    np.savez_compressed(output/'physics_trace.npz', **viewer_trace)
    np.savez_compressed(output/'score_annotations.npz', time=np.array([checked['duration_s']]),
                        fingers=np.zeros((1, 10, 88), dtype=np.uint8))
    audio = write_preparation_audio(checked, output/'played.wav')
    write_json(output/'audio_events.json', audio_events(checked['actual'], checked['time']))
    comparison = make_preparation_comparison(checked, verified['events'])
    comparison.update(source_trace_sha256=report['trace_sha256'], original_midi_sha256=verified['original_midi_sha256'],
                      score_sha256=verified['score_sha256'], model_sha256=report['model_sha256'])
    write_json(output/'comparison_events.json', comparison)
    np.savez_compressed(output/'key_comparison.npz', time=checked['time'], actual=checked['actual'],
        key_qpos=checked['key_qpos'], key_limits=checked['key_limits'], key_qpos_addresses=checked['key_qpos_addresses'],
        original_events=events, original_midi_start_at_playback_s=checked['duration_s'],
        key_activation_fraction=checked['key_activation_fraction'])
    plot_preparation_comparison(output, checked, comparison)
    meta = dict(display_contract='piano_key_colors_only_v1', checkpoint=None, preparation_only=True,
        preparation_prefix_seconds=0, preparation_duration_s=checked['duration_s'],
        original_midi_start_at_playback_s=checked['duration_s'], native_arms=False,
        dependency_versions=dict(mujoco=report['mujoco_version']), key_activation_fraction=1.,
        archive_sha256=archive_hash, model_sha256=report['model_sha256'], actual_model_shape=shape,
        source_recording=str(recording), source_physics_trace_sha256=report['trace_sha256'],
        physics_trace_sha256=sha256(output/'physics_trace.npz'),
        preparation_prefix_schema_note='All saved frames are preparation; no extra prefix is concatenated.')
    write_json(output/'replay.json', meta)
    install_replay_display(output)
    viewer = output/'viewer.py'
    source = viewer.read_text()
    old = "phase='PREPARATION' if at<meta['preparation_prefix_seconds'] else 'PERFORMANCE'"
    if source.count(old) != 1:
        raise ValueError('Shared viewer phase statement changed; review preparation-only adapter')
    viewer.write_text(source.replace(old, "phase='PREPARATION ONLY (F1 N/A)'"))
    for name in ('preview_keys.py', 'key_activation.py'):
        shutil.copy2(ROOT/'tsensei_v3_2_pianist'/name, output/name)
    (output/'replay.py').write_text('from viewer import main\nif __name__ == "__main__": main()\n')
    install_local_launchers(output)
    with (output/'README.txt').open('a') as stream:
        stream.write('\n本包只含一次首音准备段；不是原谱演奏，F1不适用，没有训练检查点。\n'
            '播放0为准备开始，原谱t=0映射到准备结束T。对比图灰区仅是未录制的原谱上下文，实际层不会延伸过去。\n'
            '若实际琴键没有激活，played.wav为真实静音，不宣称存在弹奏音。\n'
            '完整准备报告在inputs/report.json，包含瞬态越界等局限。不能据此宣称现有整曲验证了新的初始位置。\n')
        if 'preparation_control_update_hz' in report:
            stream.write(f'本次准备目标更新 {report["preparation_control_update_hz"]:g} Hz；策略控制合同 '
                f'{report["policy_control_hz"]:g} Hz，但本片段没有运行策略。\n')
        if 'native_contact_categories' in report:
            stream.write('原始全段原生接触诊断（没有因hover到达通过而排除）：'+
                         json.dumps(report['native_contact_categories'], ensure_ascii=False)+'\n')
            stream.write('存在掌壳/手指接触时不能称为无碰撞或自碰撞验收通过；全段峰值与末段稳定值不可混用。\n')
    result = dict(kind='preparation_only_recorded_replay_v1', checkpoint=None, metrics=None,
        physical_sample_micro_f1=None, performance_recorded=False, duration_s=checked['duration_s'],
        physics_hz=1/checked['physics_dt'], original_midi_start_at_playback_s=checked['duration_s'],
        activated_key_samples=checked['activated_key_samples'],
        activated_key_cell_count=checked['activated_key_cell_count'], source_report=report,
        activation_count_definition='activated_key_samples counts frames with at least one active key; activated_key_cell_count counts key-by-frame cells.',
        actual_model_shape=shape, source_archive_sha256=archive_hash, audio=audio,
        source_archive_verification=source_check,
        model_validation='MJB version/shape/key addresses validated remotely; viewer validation pending',
        limitations='Preparation-only recorded states, no new physics or policy. Preserve source transient joint excursions; '
                    'neither a dynamic safety certificate nor a full-score performance test from the changed initial pose.')
    write_json(output/'report.json', result)
    write_json(output/'manifest.json', dict(files={str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'original-midi', 'score', 'source-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model-python', type=Path, default=ROOT/'training_mjlab/.venv/bin/python')
    args = parser.parse_args()
    result = package(args.recording, args.original_midi, args.score, args.source_archive, args.output, args.model_python)
    print(json.dumps({k: result[k] for k in ('kind', 'duration_s', 'activated_key_samples', 'metrics', 'audio')}, indent=2))


if __name__ == '__main__':
    main()
