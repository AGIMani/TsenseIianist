#!/usr/bin/env python3
"""Propose four-finger assignments from model offsets and score continuity.

This symbolic search is a candidate, never a physical reachability certificate.
The original MIDI and original registered score are not overwritten.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from note_seq.protobuf import music_pb2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--beam', type=int, default=64)
    p.add_argument('--release-guard', type=float, default=.08)
    a = p.parse_args()
    seq = music_pb2.NoteSequence.FromString(a.score.read_bytes())
    original = music_pb2.NoteSequence.FromString(a.score.read_bytes())
    geo = json.loads(a.geometry.read_text())
    offsets = np.asarray(geo['rest_tips'])[:, 1]
    tips = np.asarray(geo['rest_tips'])
    key_centers = np.asarray(geo['key_centers'])
    key_sizes = np.asarray(geo['key_half_sizes'])
    keys = np.asarray(geo['key_centers'])[:, 1]
    notes = seq.notes
    assigned = {}
    schedules = {}
    for hand in (0, 1):
        groups = {}
        for i, note in enumerate(notes):
            if note.part//5 == hand:
                groups.setdefault(note.start_time, []).append(i)
        # Cost, assignments, previous hand translation, previous onset, path.
        beam = [(0., {}, 0., -2., [])]
        for t, group in sorted(groups.items()):
            options = []
            for cost, mapping, previous, last_t, path in beam:
                held = [i for i in mapping if notes[i].end_time > t]
                occupied = {mapping[i] for i in held}
                available = [hand*5+f for f in range(1, 5) if hand*5+f not in occupied]
                for choice in itertools.permutations(available, len(group)):
                    trial = dict(mapping)
                    trial.update(zip(group, choice))
                    current = held + group
                    # Keep anatomical ordering while old fingers are releasing.
                    nearby = [i for i in mapping if notes[i].end_time > t-a.release_guard] + group
                    direction = 1 if hand == 0 else -1
                    if any(trial[i] == trial[j] and notes[i].pitch != notes[j].pitch
                           for i, j in itertools.combinations(nearby, 2)):
                        continue
                    if any((trial[i]-trial[j])*(notes[i].pitch-notes[j].pitch)*direction < 0
                           for i, j in itertools.combinations(current, 2)):
                        continue
                    crossing = sum((trial[i]-trial[j])*(notes[i].pitch-notes[j].pitch)*direction < 0
                                   for i, j in itertools.combinations(nearby, 2))
                    centers = np.array([keys[notes[i].pitch-21]-offsets[trial[i]] for i in current])
                    dt = max(t-last_t, .05)
                    # Allow finger lateral swing; movement cost favors continuity.
                    weight = .08 / max(dt, .12)**2
                    center = (float(centers.sum()) + weight*previous)/(len(centers)+weight)
                    spread = np.sum(((centers-center)/.025)**2)
                    # Black keys end behind white-key front landing regions.
                    # Model-derived finger lengths must accommodate both
                    # landing rectangles under one shared wrist translation.
                    x_intervals=[]
                    for i in current:
                        key=notes[i].pitch-21
                        xmin=key_centers[key,0]-.8*key_sizes[key,0]
                        xmax=key_centers[key,0]+.8*key_sizes[key,0]
                        if key_sizes[key,1]>.008:xmin=max(xmin,.022)
                        x_intervals.append((xmin-tips[trial[i],0],xmax-tips[trial[i],0]))
                    depth_gap=max(0.,max(x[0] for x in x_intervals)-min(x[1] for x in x_intervals))
                    depth_cost=3.*(depth_gap/.01)**2
                    movement = .08*((center-previous)/(.025*max(dt, .12)))**2
                    edit = .025*sum(trial[i] != notes[i].part for i in group)
                    candidate = (cost+spread+depth_cost+movement+edit+2.*crossing, trial, center, t, path+[(t, center)])
                    options.append(candidate)
            if not options:
                raise RuntimeError(f'No noncrossing four-finger assignment at {t}, hand {hand}')
            options.sort(key=lambda row: row[0])
            # Deduplicate equivalent current/recent assignments and translation.
            unique = {}
            for row in options:
                recent = tuple(sorted((i, f) for i, f in row[1].items() if notes[i].end_time > t-.25))
                key = (recent, round(row[2], 3))
                unique.setdefault(key, row)
                if len(unique) >= a.beam:
                    break
            beam = list(unique.values())
        best = min(beam, key=lambda row: row[0])
        assigned.update(best[1])
        schedules[str(hand)] = best[4]
    changes = []
    for i, note in enumerate(notes):
        part = assigned[i]
        if part != note.part:
            changes.append(dict(note_index=i, start_s=note.start_time, end_s=note.end_time,
                                pitch=note.pitch, old_part=note.part, new_part=part))
        note.part = part
    # Verify every protobuf field except the explicitly authorized part edit.
    check = music_pb2.NoteSequence.FromString(seq.SerializeToString())
    for x, y in zip(check.notes, original.notes):
        x.part = y.part
    if check.SerializeToString() != original.SerializeToString():
        raise AssertionError('Unexpected score mutation beyond fingering')
    for i, n in enumerate(notes):
        for m in notes[i+1:]:
            if n.part == m.part and max(n.start_time, m.start_time) < min(n.end_time, m.end_time):
                raise AssertionError('Overlapping notes assigned to one finger')
    a.output.mkdir(parents=True, exist_ok=False)
    proto = a.output/'015-5_fingered.proto'
    proto.write_bytes(seq.SerializeToString())
    text = ['// IK revision: signed 1..5 right, -1..-5 left; original MIDI times unchanged.']
    import pretty_midi
    for i, n in enumerate(notes):
        finger = (n.part % 5 + 1)*(1 if n.part < 5 else -1)
        text.append(f'{i}\t{n.start_time:.12g}\t{n.end_time:.12g}\t'
                    f'{pretty_midi.note_number_to_name(n.pitch)}\t{n.velocity}\t80\t{n.part//5}\t{finger}')
    (a.output/'015-5_fingering.txt').write_text('\n'.join(text)+'\n')
    report = dict(kind='model_offset_four_finger_candidate', user_authorized_revision=True,
                  notes=len(notes), changed_notes=len(changes), changes=changes,
                  original_score=str(a.score.resolve()), original_score_sha256=sha(a.score),
                  geometry_sha256=sha(a.geometry), revised_score_sha256=sha(proto),
                  source_sha256=sha(__file__), thumb_assignments=0, physical_validation='pending',
                  release_guard_s=a.release_guard,
                  chord_depth_cost='3*(gap between model-offset longitudinal landing intervals / 0.01m)^2',
                  preserved='Every original score protobuf field except note.part',
                  planner_translation_m=schedules)
    (a.output/'revision_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('changes', 'planner_translation_m')}, indent=2))


if __name__ == '__main__':
    main()
