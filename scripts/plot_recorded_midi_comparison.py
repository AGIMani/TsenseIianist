#!/usr/bin/env python3
"""CPU-only comparison of one exported recording with its embedded original score."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
from note_seq.protobuf import music_pb2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(mask, times, end):
    result = []
    for key in range(88):
        changes = np.diff(np.r_[False, mask[:, key], False].astype(np.int8))
        for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            result.append([float(times[first]), float(times[last]) if last < len(times) else end, key+21])
    return result


def draw(ax, events, color):
    vertices = [[(s, p-.4), (e, p-.4), (e, p+.4), (s, p+.4)] for s, e, p in events]
    ax.add_collection(PolyCollection(vertices, facecolors=color, edgecolors='none'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    refs = parser.add_mutually_exclusive_group(required=True)
    refs.add_argument('--original-midi', type=Path)
    refs.add_argument('--reference-events', type=Path, help='Verified original events and config SHA provenance JSON')
    parser.add_argument('--label', default='Checkpoint replay')
    args = parser.parse_args()
    output = args.export/'midi_comparison'
    output.mkdir(exist_ok=False)
    config_path = args.export/'inputs/config.json'
    config = json.loads(config_path.read_text())
    report_path = args.recording/'report.json'
    report = json.loads(report_path.read_text())
    exported = json.loads((args.export/'report.json').read_text())
    assert report['checkpoint_sha256'] == exported['checkpoint_sha256']
    assert report['source_runtime'] == config['source_runtime']
    if args.reference_events:
        provenance = json.loads(args.reference_events.read_text())
        assert provenance['config_sha256'] == sha(config_path)
        source = Path(provenance['midi_path'])
        assert sha(source) == provenance['midi_sha256']
        original_events = np.asarray(provenance['events'], dtype=float)
        assert original_events.ndim == 2 and original_events.shape[1] == 3
        assert np.isfinite(original_events).all() and (original_events[:, 1] > original_events[:, 0]).all()
        (output/'reference_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        raw = None
        if provenance.get('reference_kind') == 'registered_note_sequence':
            raw = source.read_bytes()
            embedded = config['source_runtime']['external_score']
            assert raw == base64.b64decode(embedded['base64'])
            assert sha(source) == embedded['sha256']
            seq = music_pb2.NoteSequence.FromString(raw)
            assert np.array_equal(original_events, [[n.start_time,n.end_time,n.pitch] for n in seq.notes])
            (output/'original_score.proto').write_bytes(raw)
        else:
            (output/'original.mid').write_bytes(source.read_bytes())
    else:
        original_events, raw = verify_embedded_midi(config, args.original_midi, output)
        provenance = dict(midi_sha256=sha(args.original_midi), original_note_attributes_verified=True)
    offset = float(report['music_start_policy_seconds']) + float(report['preparation_prefix_seconds'])
    reference, trim_origin = align_external_events(original_events, raw, offset)
    provenance = dict(provenance, source_trim_origin_seconds=trim_origin,
                      timeline='Source onset/end minus source trim origin plus music/preparation offset')
    plot_comparison(args, output, config_path, report_path, report, exported, reference, offset, raw, provenance)


def align_external_events(events, raw, offset):
    # Source runtime trims leading silence before creating its score. Keep the
    # unmodified source bytes for provenance and apply the same time origin.
    origin = 0.
    if raw is not None:
        sequence = music_pb2.NoteSequence.FromString(raw)
        origin = float(sequence.notes[0].start_time)
    return [[float(s)-origin+offset, float(e)-origin+offset, int(p)]
            for s, e, p in events], origin


def verify_embedded_midi(config, midi_path, output):
    raw = base64.b64decode(config['source_runtime']['external_score']['base64'])
    assert hashlib.sha256(raw).hexdigest() == config['source_runtime']['external_score']['sha256']
    sequence = music_pb2.NoteSequence.FromString(raw)
    # Independently verify that assigning fingering did not change original MIDI notes.
    from note_seq.midi_io import midi_file_to_note_sequence
    original = midi_file_to_note_sequence(str(midi_path))
    def notes(seq):
        return sorted((n.pitch, n.start_time, n.end_time, n.velocity, n.instrument) for n in seq.notes)
    assert np.allclose(notes(sequence), notes(original), rtol=0, atol=1e-9), 'Original MIDI differs from checkpoint score'
    (output/'original_score.proto').write_bytes(raw)
    (output/'original.mid').write_bytes(midi_path.read_bytes())
    return [[n.start_time, n.end_time, n.pitch] for n in sequence.notes], raw


def plot_comparison(args, output, config_path, report_path, report, exported, reference, offset, raw, provenance):
    audio_path = args.recording/'audio_substeps.npz'
    with np.load(audio_path, allow_pickle=False) as data:
        q = data['key_qpos']; limits = data['key_limits']; threshold = float(data['key_activation_fraction'])
        dt = float(data['physics_dt'])
        actual = np.clip(q, limits[:, 0], limits[:, 1]) >= limits[:, 1] * threshold
    time = np.arange(1, len(actual)+1)*dt
    end = float(time[-1])
    assert np.isclose(end, exported['recorded_duration'], atol=1e-7)
    assert np.isclose(1/dt, exported['physics_hz'])
    expected = np.zeros_like(actual)
    for start, stop, pitch in reference:
        expected[(time >= start-1e-10) & (time < stop-1e-10), pitch-21] = True
    tp = int((actual & expected).sum()); fp = int((actual & ~expected).sum()); fn = int((~actual & expected).sum())
    event_sets = {'original': reference, 'actual': intervals(actual, time, end),
                  'correct': intervals(actual & expected, time, end),
                  'extra': intervals(actual & ~expected, time, end),
                  'missing': intervals(~actual & expected, time, end)}
    for name, rows in event_sets.items():
        np.savetxt(output/(name+'_events.csv'), np.asarray(rows).reshape(-1, 3),
                   delimiter=',', header='start_s,end_s,midi_pitch', comments='', fmt='%.10g')
    np.savez_compressed(output/'key_comparison.npz', time=time, actual=actual, original=expected,
                        physics_dt=dt, original_events=np.asarray(reference))
    pitches = [row[2] for row in reference + event_sets['actual']]
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True, sharey=True)
    draw(axes[0], reference, '#2563eb')
    draw(axes[1], event_sets['correct'], '#159f73'); draw(axes[1], event_sets['extra'], '#ea6d24')
    draw(axes[2], event_sets['missing'], '#617fe5'); draw(axes[2], event_sets['extra'], '#ea6d24')
    reference_label = 'Registered score' if provenance.get('reference_kind') == 'registered_note_sequence' else 'Original MIDI'
    for ax, title in zip(axes, (reference_label+' (unquantized note times)', 'Actual keys (physical activation threshold)', 'Timing errors vs '+reference_label)):
        ax.set_title(title, loc='left'); ax.set_ylabel('MIDI pitch')
        ax.set_ylim(min(pitches)-2, max(pitches)+2); ax.set_xlim(0, end)
        ax.axvspan(0, offset, color='#d5dae1', alpha=.4)
        ax.axvline(offset, color='#64748b', ls='--', lw=.8)
        ax.grid(alpha=.15)
    axes[1].legend(handles=[Patch(color='#159f73', label='Expected key active'), Patch(color='#ea6d24', label='Extra / early / late key')], loc='upper right')
    axes[2].legend(handles=[Patch(color='#617fe5', label='Missing / late onset / early release'), Patch(color='#ea6d24', label='Unexpected activation')], loc='upper right')
    axes[-1].set_xlabel('Original replay time (seconds), including preparation')
    f1 = 2*tp/max(2*tp+fp+fn, 1)
    fig.suptitle(f'{args.label} | same single replay | {1/dt:g} Hz actual keys | original MIDI offset +{offset:g} s\n'
                 f'Physical-sample micro-F1 = {f1:.4f} | source leading silence {provenance.get("source_trim_origin_seconds",0):g} s removed | video 1x, no crop', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output/'piano_roll_comparison.png', dpi=160)
    plt.close(fig)
    metadata = dict(checkpoint_sha256=report['checkpoint_sha256'], original_midi_sha256=provenance['midi_sha256'],
                    config_sha256=sha(config_path), embedded_proto_sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                    recording_report_sha256=sha(report_path), audio_substeps_sha256=sha(audio_path),
                    reference_provenance=provenance, original_note_count=len(reference),
                    sample_hz=1/dt, samples=len(time), duration_seconds=end, music_offset_seconds=offset,
                    actual_source='Recorded physical key positions with model limits and activation threshold; no policy rerun',
                    sample_time='Post-physics-step: (i+1)*dt; intervals use sample-and-hold to next sample',
                    interval_resolution_seconds=dt, key_activation_fraction=threshold,
                    final_sample_active_keys=int(actual[-1].sum()),
                    terminal_interval_note='Any final active key is right-censored at recording end, not a measured release',
                    physical_sample_micro_f1=f1, tp=tp, fp=fp, fn=fn,
                    metric_scope='This full single replay at physics sample times vs unquantized original MIDI',
                    control_frame_phase_metrics=report['phase_metrics'])
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in ('sample_hz','duration_seconds','music_offset_seconds','physical_sample_micro_f1','tp','fp','fn')}, indent=2))


if __name__ == '__main__':
    main()
