#!/usr/bin/env python3
"""Repair non-thumb ordering conflicts without changing note events.

This is a symbolic candidate generator, not a collision-free certification.
Physical A3.2 validation must precede replacing an active score.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conflict_edges(notes, dt, clearance_time):
    """Pairs overlapping on the control grid or during a release guard.

    The guard is a planner constraint, not measured release time. It never
    changes the MIDI start/end times. Return exact durations for auditing.
    """
    origin = min(n.start_time for n in notes)
    rows = []
    for hand in (0, 1):
        ids = sorted((i for i, n in enumerate(notes) if n.part // 5 == hand),
                     key=lambda i: (notes[i].start_time, i))
        for offset, i in enumerate(ids):
            a = notes[i]
            last = max(a.end_time + clearance_time,
                       np.ceil((a.end_time-origin)/dt)*dt+origin)
            for j in ids[offset+1:]:
                b = notes[j]
                if b.start_time >= last:
                    break
                grid = (int((b.start_time-origin)/dt) <
                        max(int((a.start_time-origin)/dt)+1,
                            int(np.ceil((a.end_time-origin)/dt))))
                gap = b.start_time-a.end_time
                if grid or gap < clearance_time:
                    rows.append((i, j, bool(grid), float(gap)))
    return rows


def violations(notes, parts, edges):
    result = []
    for i, j, grid, gap in edges:
        f, g = int(parts[i]) % 5, int(parts[j]) % 5
        direction = 1 if int(parts[i]) < 5 else -1
        duplicate = grid and f == g and notes[i].pitch != notes[j].pitch
        reverse = (f > 0 and g > 0 and
                   (f-g)*(notes[i].pitch-notes[j].pitch)*direction < 0)
        if duplicate or reverse:
            result.append(dict(note_indices=[i, j], pitches=[notes[i].pitch, notes[j].pitch],
                               fingers=[int(parts[i]), int(parts[j])],
                               start_s=max(notes[i].start_time, notes[j].start_time),
                               gap_s=gap, control_grid_overlap=grid,
                               same_finger_conflict=duplicate, nonthumb_reverse=reverse,
                               index_middle_reverse=reverse and {f,g} == {1,2}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-dt', type=float, default=.05)
    p.add_argument('--release-guard', type=float, default=.05)
    a = p.parse_args()
    if a.control_dt <= 0 or a.release_guard < 0:
        p.error('Invalid control dt/release guard')
    from note_seq.protobuf import music_pb2
    seq = music_pb2.NoteSequence.FromString(a.score.read_bytes())
    baseline = music_pb2.NoteSequence.FromString(a.score.read_bytes())
    notes = seq.notes
    original = np.array([n.part for n in notes], dtype=int)
    if not len(notes) or np.any((original < 0) | (original > 9)):
        raise ValueError('Expected complete original fingering')
    edges = conflict_edges(notes, a.control_dt, a.release_guard)
    nvars = len(notes)*5
    rr, cc, vv, low, high = [], [], [], [], []

    def row(columns, lo, hi):
        r = len(low)
        rr.extend([r]*len(columns)); cc.extend(columns); vv.extend([1.]*len(columns))
        low.append(lo); high.append(hi)

    for i in range(len(notes)):
        row([5*i+f for f in range(5)], 1., 1.)
    for i, j, grid, gap in edges:
        direction = 1 if original[i] < 5 else -1
        if grid and notes[i].pitch != notes[j].pitch:
            for f in range(5):
                row([5*i+f, 5*j+f], -np.inf, 1.)
        # Do not "repair" an index/middle crossing merely by assigning the
        # middle note to ring/pinky and moving the crossing to another pair.
        for f in range(1, 5):
            for g in range(1, 5):
                if (f-g)*(notes[i].pitch-notes[j].pitch)*direction < 0:
                    row([5*i+f, 5*j+g], -np.inf, 1.)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(low), nvars)).tocsc()
    constraint = LinearConstraint(matrix, low, high)
    changes = np.ones((len(notes), 5))
    changes[np.arange(len(notes)), original % 5] = 0
    options = dict(time_limit=120., mip_rel_gap=0.)
    result = milp(changes.ravel(), integrality=np.ones(nvars), bounds=Bounds(0., 1.),
                  constraints=constraint, options=options)
    if not result.success:
        raise RuntimeError(f'Fingering repair not solved: {result.message}')
    minimum = int(round(result.fun))
    # Preserve the minimum edit count; model-derived neutral translation is
    # only a tie-breaker, never a claim about actual collision clearance.
    geo = json.loads(a.geometry.read_text())
    tip = np.array(geo['tip_xyz'])
    costs = np.zeros((len(notes), 5))
    for i, old in enumerate(original):
        for f in range(5):
            costs[i, f] = np.sum(((tip[old//5*5+f, :2]-tip[old, :2])/.0235)**2)
    second = milp(costs.ravel(), integrality=np.ones(nvars), bounds=Bounds(0., 1.),
                  constraints=[constraint, LinearConstraint(changes.reshape(1, -1), minimum, minimum)],
                  options=options)
    if not second.success:
        raise RuntimeError(f'Fingering tie-break not solved: {second.message}')
    parts = original//5*5 + second.x.reshape(-1, 5).argmax(axis=1)
    assert len(violations(notes, parts, edges)) == 0
    assert int(np.sum(parts != original)) == minimum
    changed = []
    for i, (note, part) in enumerate(zip(notes, parts)):
        if part != original[i]:
            changed.append(dict(note_index=i, pitch=note.pitch, start_s=note.start_time,
                                end_s=note.end_time, old_finger=int(original[i]), new_finger=int(part)))
        note.part = int(part)
    check = music_pb2.NoteSequence.FromString(seq.SerializeToString())
    for x, y in zip(check.notes, baseline.notes):
        x.part = y.part
    assert check == baseline, 'Something besides fingering changed'
    a.output.mkdir(parents=True, exist_ok=False)
    output = a.output/'turkish-march-advanced_fingered.proto'
    output.write_bytes(seq.SerializeToString())
    with (a.output/'fingering.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['note_index', 'pitch', 'start_seconds', 'end_seconds', 'hand', 'finger_number'])
        writer.writerows((i, n.pitch, n.start_time, n.end_time, 'left' if n.part >= 5 else 'right',
                          n.part%5+1) for i, n in enumerate(notes))
    report = dict(status='candidate_pending_physical_validation', source_sha256=sha(a.score),
                  output_sha256=sha(output), geometry_sha256=sha(a.geometry),
                  runner_sha256=sha(__file__), control_dt=a.control_dt,
                  release_guard_s=a.release_guard, release_guard_kind='planner assumption, not measured',
                  note_count=len(notes), changes=changed, minimum_changed_notes=minimum,
                  before=violations(baseline.notes, original, edges),
                  after=violations(notes, parts, edges), all_other_sequence_fields_preserved=True,
                  physical_collisions_verified=False, physical_f1=None,
                  limitations='Symbolic non-thumb ordering only. Thumb crossing, idle fingers, '
                              'actual timing, reachability and physical contact need A3.2 dynamics.')
    (a.output/'repair_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), changed_notes=minimum,
                          before=len(report['before']), after=len(report['after']),
                          physical_validation=False), indent=2))


if __name__ == '__main__':
    main()
