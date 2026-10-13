#!/usr/bin/env python3
"""Offline candidate fingering for explicitly selected MIDI instrument/hand mapping.

Engineering heuristic, not a published fingering model or calibrated reach solver.
Keeps MIDI notes intact; writes NoteSequence .proto because ordinary MIDI loses part.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.music.midi_file import MidiFile, NoteTrajectory


def assign_hand(notes, indices, *, left, dt, origin):
    spans = {i: (int((notes[i].start_time-origin)/dt),
                 max(int((notes[i].start_time-origin)/dt)+1,
                     math.ceil((notes[i].end_time-origin)/dt))) for i in indices}
    events = sorted({s for s, _ in spans.values()})
    order = [9, 8, 7, 6, 5] if left else [0, 1, 2, 3, 4]
    offsets = dict(zip(order, [-7., -5., -4., -2., 0.] if left else [0., 2., 4., 5., 7.]))
    layers = []
    for k, frame in enumerate(events):
        active = [i for i in indices if spans[i][0] <= frame < spans[i][1]]
        pitches = sorted({notes[i].pitch for i in active})
        if len(pitches) > 5:
            raise ValueError(f'More than five active pitches at frame {frame}')
        current = []
        for fingers in itertools.permutations(order, len(pitches)):
            by_pitch = dict(zip(pitches, fingers))
            mapping = {i: by_pitch[notes[i].pitch] for i in active}
            anchors = [pitch-offsets[finger] for pitch, finger in by_pitch.items()]
            center = sum(anchors)/len(anchors)
            posture = sum((a-center)**2 for a in anchors)/len(anchors)
            # Allow finger crossing/turning where strict order cannot bridge a phrase.
            # Penalize reversed order; do not discard the entire score.
            posture += 12*sum(offsets[fingers[a]] > offsets[fingers[b]]
                              for a in range(len(fingers)) for b in range(a+1, len(fingers)))
            # Mild preference against thumb on a black key; no hard prohibition.
            posture += sum(.4 for pitch, finger in by_pitch.items()
                           if pitch % 12 in (1, 3, 6, 8, 10) and finger in (0, 5))
            best, parent = float('inf'), None
            if k == 0:
                best = posture
            else:
                for j, prev in enumerate(layers[-1]):
                    if any(i in prev['mapping'] and prev['mapping'][i] != f
                           for i, f in mapping.items()):
                        continue
                    elapsed = max(dt, (frame-events[k-1])*dt)
                    movement = .35*(center-prev['center'])**2/max(.25, elapsed)
                    score = prev['cost']+posture+movement
                    if score < best:
                        best, parent = score, j
            if math.isfinite(best):
                current.append(dict(mapping=mapping, center=center, cost=best, parent=parent))
        if not current:
            raise ValueError(f'No consistent candidate at frame {frame}; manual fingering needed')
        layers.append(current)
    chosen = min(range(len(layers[-1])), key=lambda j: layers[-1][j]['cost'])
    assignments = {}
    cost = layers[-1][chosen]['cost']
    for layer in reversed(layers):
        state = layer[chosen]
        for i, finger in state['mapping'].items():
            if i in assignments and assignments[i] != finger:
                raise AssertionError('Sustained note changed fingers')
            assignments[i] = finger
        chosen = state['parent']
    return assignments, cost


def signature(note):
    return (note.pitch, note.start_time, note.end_time, note.velocity,
            note.instrument, note.program, note.is_drum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--midi', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--right-instrument', type=int, required=True)
    parser.add_argument('--left-instrument', type=int, required=True)
    parser.add_argument('--dt', type=float, default=.025)
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error('dt must be finite and positive')
    if args.right_instrument == args.left_instrument:
        parser.error('Hands must use different instrument indices')
    midi = MidiFile.from_file(args.midi)
    notes = midi.seq.notes
    if not notes or any(n.is_drum for n in notes):
        raise ValueError('Expected nonempty piano notes')
    if {n.instrument for n in notes} != {args.right_instrument, args.left_instrument}:
        raise ValueError('Every instrument must be explicitly assigned to a hand')
    original = sorted(signature(n) for n in notes)
    origin = min(n.start_time for n in notes)
    assignments, costs = {}, {}
    for left, instrument in ((False, args.right_instrument), (True, args.left_instrument)):
        ids = [i for i, n in enumerate(notes) if n.instrument == instrument]
        mapping, cost = assign_hand(notes, ids, left=left, dt=args.dt, origin=origin)
        assignments.update(mapping)
        costs['left' if left else 'right'] = cost
    rows = []
    names = ['拇指', '食指', '中指', '无名指', '小指']
    for i, n in enumerate(notes):
        n.part = assignments[i]
        name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][n.pitch % 12]+str(n.pitch//12-1)
        rows.append(dict(note_index=i, pitch=n.pitch, note_name=name,
                         start_seconds=n.start_time, end_seconds=n.end_time,
                         velocity=n.velocity, instrument=n.instrument, finger_id=n.part,
                         hand='left' if n.part >= 5 else 'right',
                         finger_number=n.part % 5+1, finger_name=names[n.part % 5]))
    ordered = sorted(notes, key=lambda n: (n.start_time, n.end_time, n.pitch, n.instrument))
    copies = [type(n)().FromString(n.SerializeToString()) for n in ordered]
    del notes[:]
    notes.extend(copies)
    assert sorted(signature(n) for n in notes) == original
    if abs(notes[-1].end_time-max(n.end_time for n in notes)) > 1e-9:
        raise ValueError('Latest-onset note is not latest-ending: source trim needs review')
    trimmed = midi.trim_silence()
    assert len(trimmed.seq.notes) == len(notes), 'Trim lost notes'
    trajectory = NoteTrajectory.from_midi(trimmed, args.dt)
    for frame, chord in enumerate(trajectory.notes):
        pairs = [(n.fingering, n.key) for n in chord]
        if len({finger for finger, _ in pairs}) != len(pairs):
            raise ValueError(f'One finger assigned multiple simultaneous keys at frame {frame}')
        if any(not 0 <= finger < 10 for finger, _ in pairs):
            raise ValueError('Incomplete assignment')
    args.output.mkdir(parents=True, exist_ok=False)
    proto = args.output/'clair-de-lune-beginner_fingered.proto'
    midi.save(proto)
    loaded = MidiFile.from_file(proto)
    assert loaded.has_fingering() and loaded.seq == midi.seq
    with (args.output/'fingering.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(method='dynamic programming over hand assignments with a soft finger-order preference; sustained-note constraints and posture/shift costs',
                  source_sha256=hashlib.sha256(args.midi.read_bytes()).hexdigest(),
                  proto_sha256=hashlib.sha256(proto.read_bytes()).hexdigest(),
                  instrument_to_hand={str(args.right_instrument): 'right', str(args.left_instrument): 'left'},
                  note_count=len(notes), all_note_attributes_preserved=True, dt=args.dt,
                  trim_origin_seconds=origin, trimmed_note_count=len(trimmed.seq.notes),
                  trajectory_frames=len(trajectory.notes), simultaneous_finger_conflicts=0,
                  counts_by_finger={str(f): sum(n.part == f for n in notes) for f in range(10)},
                  costs=costs, limitations='Candidate only. Hand split inferred from supplied tracks. Offsets/costs are engineering heuristics, not calibrated TsenseI IK or expert fingering. No claim of collision-free reachability.')
    (args.output/'fingering_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
