#!/usr/bin/env python3
"""Revise fingering from recorded key activation; no policy/model inference.

Empirical coverage is associated with the assigned finger, not measured contact
attribution. Alternative fingers are hypotheses supported by other score notes.
This does not establish reachability or predict retrained performance.
"""
import argparse
import base64
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from note_seq.protobuf import music_pb2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def name(pitch):
    return ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[pitch % 12] + str(pitch // 12 - 1)


def black(pitch):
    return pitch % 12 in (1, 3, 6, 8, 10)


def analyze(seq, samples, episodes, offset):
    times, actual = samples['time'], samples['actual']
    assert actual.shape == (len(times), 88)
    episode_keys = []
    for ep in episodes:
        played = [r for r in ep['note_timing_records'] if r['status'] == 'matched'] + ep['note_timing_extra_records']
        assert len(played) == ep['note_timing_played_count']
        keys = np.zeros_like(actual, dtype=bool)
        for event in played:
            keys[(times >= event['played_start_s']-1e-10) &
                 (times < event['played_end_s']-1e-10), event['pitch']-21] = True
        episode_keys.append(keys)
    rows = []
    for i, note in enumerate(seq.notes):
        start, end = note.start_time + offset, note.end_time + offset
        mask = (times >= start-1e-10) & (times < end-1e-10)
        active = actual[mask, note.pitch - 21]
        eval_hold = [float(keys[mask, note.pitch-21].mean()) for keys in episode_keys]
        assert len(active) > 0
        records = []
        for ep in episodes:
            matches = [r for r in ep['note_timing_records']
                       if r['pitch'] == note.pitch and r['finger'] == note.part
                       and abs(r['target_start_s'] - start) < 1e-7
                       and abs(r['target_end_s'] - end) < 1e-7]
            assert len(matches) == 1, (i, len(matches))
            records.append(matches[0])
        live = np.flatnonzero(active)
        rows.append(dict(note_index=i, pitch=note.pitch, note_name=name(note.pitch),
                         start_seconds=note.start_time, end_seconds=note.end_time,
                         duration_seconds=note.end_time-note.start_time,
                         finger_id=note.part, hand='left' if note.part >= 5 else 'right',
                         coverage=float(active.mean()), active_samples=int(active.sum()),
                         evaluation_hold_coverage=float(np.mean(eval_hold)),
                         evidence_coverage=float(np.mean([float(active.mean()), *eval_hold])),
                         target_samples=len(active),
                         onset_gap_ms=(float(live[0]) * 5 if len(live) else None),
                         release_gap_ms=(float(len(active)-1-live[-1]) * 5 if len(live) else None),
                         evaluation_onset_misses=sum(r['status'] == 'missed' for r in records),
                         evaluation_episodes=len(records)))
    return rows


def alternative(rows, row, finger):
    # Same-pitch evidence first. Local pitch/color fallback has an explicit
    # uncertainty deduction; neither is a calibrated physical reach estimate.
    pool = [r for r in rows if r['finger_id'] == finger and r['pitch'] == row['pitch']]
    kind = 'same_pitch'
    if len(pool) < 3:
        pool = [r for r in rows if r['finger_id'] == finger
                and abs(r['pitch']-row['pitch']) <= 4
                and black(r['pitch']) == black(row['pitch'])]
        kind = 'nearby_pitch_same_key_color'
    if len(pool) < 3:
        return None
    mean = float(np.mean([r['evidence_coverage'] for r in pool]))
    discounted = mean - (0.10 if kind == 'same_pitch' else 0.20)
    return dict(mean_coverage=mean, discounted_score=discounted, count=len(pool),
                kind=kind, supporting_note_indices=[r['note_index'] for r in pool])


def optimize(seq, rows, dt):
    assignments = {i: int(n.part) for i, n in enumerate(seq.notes)}
    evidence = {}
    options = {}
    for i, row in enumerate(rows):
        original = row['finger_id']
        options[i] = [original]
        # Leave well-held notes alone even if the onset matcher marks them
        # missed: a late-onset sustained note may still have high hold coverage.
        if max(row['coverage'], row['evaluation_hold_coverage']) >= .65:
            continue
        for finger in range(5 if original >= 5 else 0, 10 if original >= 5 else 5):
            if finger == original:
                continue
            support = alternative(rows, row, finger)
            if support and support['mean_coverage'] >= .65 and support['discounted_score'] >= row['evidence_coverage'] + .12:
                options[i].append(finger)
                evidence[(i, finger)] = support
    for left in (False, True):
        ids = [i for i,n in enumerate(seq.notes) if (n.part >= 5) == left]
        spans = {i: (int(seq.notes[i].start_time/dt),
                     max(int(seq.notes[i].start_time/dt)+1, math.ceil(seq.notes[i].end_time/dt))) for i in ids}
        events = sorted({spans[i][0] for i in ids})
        layers = []
        # These offsets are the original candidate generator's heuristic, not
        # inferred/calibrated robot geometry. They only regularize reassignment.
        offsets = {5: 0., 6: -2., 7: -4., 8: -5., 9: -7.} if left else {0: 0., 1: 2., 2: 4., 3: 5., 4: 7.}
        for k, frame in enumerate(events):
            active = sorted([i for i in ids if spans[i][0] <= frame < spans[i][1]], key=lambda i: seq.notes[i].pitch)
            current = []
            old_fingers = [int(seq.notes[i].part) for i in active]
            def crossings(ff):
                return sum(offsets[ff[a]] > offsets[ff[b]] for a in range(len(ff)) for b in range(a+1, len(ff)))
            old_cross = crossings(old_fingers)
            for fingers in itertools.product(*(options[i] for i in active)):
                if len(set(fingers)) != len(fingers) or crossings(fingers) > old_cross:
                    continue
                mapping = dict(zip(active, fingers))
                anchors = [seq.notes[i].pitch-offsets[f] for i,f in mapping.items()]
                center = float(np.mean(anchors))
                posture = .06 * float(np.var(anchors))
                onset_cost = 0.
                for i,f in mapping.items():
                    if spans[i][0] == frame:
                        if f == seq.notes[i].part:
                            coverage = rows[i]['evidence_coverage']
                        else:
                            coverage = evidence[(i,f)]['discounted_score']
                            onset_cost += .65
                        onset_cost += 12*(1-coverage)
                cost, parent = float('inf'), None
                if k == 0:
                    cost = posture + onset_cost
                else:
                    elapsed = max(dt, (frame-events[k-1])*dt)
                    for j, prev in enumerate(layers[-1]):
                        if any(i in prev['mapping'] and prev['mapping'][i] != f for i,f in mapping.items()):
                            continue
                        movement = .04*(center-prev['center'])**2/max(.1, elapsed)
                        # Penalize rapid same-finger jumps and rapid pitch/finger
                        # order reversals; allow baseline crossings/turnovers.
                        for i,f in mapping.items():
                            if spans[i][0] != frame:
                                continue
                            for pi,pf in prev['mapping'].items():
                                if pi in mapping or elapsed > .25:
                                    continue
                                jump = seq.notes[i].pitch-seq.notes[pi].pitch
                                if pf == f:
                                    movement += .25 * max(0, abs(jump)-2)**2
                                if jump*(offsets[f]-offsets[pf]) < 0:
                                    movement += .30*abs(jump)
                        total = prev['cost'] + movement + posture + onset_cost
                        if total < cost:
                            cost, parent = total, j
                if math.isfinite(cost):
                    current.append(dict(mapping=mapping, center=center, cost=cost, parent=parent))
            if not current:
                raise ValueError(f'No feasible assignment at frame {frame}; no output written')
            layers.append(current)
        chosen = min(range(len(layers[-1])), key=lambda j: layers[-1][j]['cost'])
        for layer in reversed(layers):
            state = layer[chosen]
            assignments.update(state['mapping'])
            chosen = state['parent']
    return assignments, evidence


def validate(original, revised):
    # Byte-equivalent entire protobuf after resetting only NoteSequence.part.
    assert len(original.notes) == len(revised.notes)
    assert all(0 <= n.part < 10 for n in revised.notes)
    restored = music_pb2.NoteSequence.FromString(revised.SerializeToString())
    for before, after in zip(original.notes, restored.notes):
        assert (before.part >= 5) == (after.part >= 5)
        after.part = before.part
    assert restored == original
    conflicts = {}
    for hz in (None, 20, 40):
        count = 0
        for i,a in enumerate(revised.notes):
            sa,ea = (a.start_time,a.end_time) if hz is None else (int(a.start_time*hz),math.ceil(a.end_time*hz))
            for b in revised.notes[i+1:]:
                sb,eb = (b.start_time,b.end_time) if hz is None else (int(b.start_time*hz),math.ceil(b.end_time*hz))
                if sb >= ea:
                    continue
                if max(sa,sb)<min(ea,eb) and a.part==b.part and a.pitch!=b.pitch:
                    count += 1
        conflicts['continuous' if hz is None else str(hz)+'hz'] = count
    assert not any(conflicts.values()), conflicts
    return dict(all_fields_except_part_identical=True, hand_assignments_preserved=True,
                simultaneous_finger_conflicts=conflicts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args();exp=args.experiment
    config_path=exp/'config.json';summary_path=exp/'summary.json'
    cfg=json.loads(config_path.read_text());summary=json.loads(summary_path.read_text())
    frozen=base64.b64decode(cfg['source_runtime']['external_score']['base64'])
    assert hashlib.sha256(frozen).hexdigest()==cfg['source_runtime']['external_score']['sha256']
    assert args.source.read_bytes()==frozen, 'Selected score must exactly match the trained score'
    seq=music_pb2.NoteSequence.FromString(frozen)
    comparison_path=exp/'video_best/midi_comparison/comparison.json'
    comparison=json.loads(comparison_path.read_text())
    assert comparison['config_sha256']==sha(config_path)
    assert comparison['checkpoint_sha256']==sha(exp/'best.pt')
    assert comparison['audio_substeps_sha256']==sha(exp/'video_best/replay_data/audio_substeps.npz')
    sample_path=exp/'video_best/midi_comparison/key_comparison.npz'
    data=np.load(sample_path);times=data['time']
    assert np.allclose(np.diff(times),.005) and comparison['sample_hz']==200
    with np.load(exp/'video_best/replay_data/audio_substeps.npz') as physical:
        actual = np.clip(physical['key_qpos'], physical['key_limits'][:,0],
                               physical['key_limits'][:,1]) >= physical['key_limits'][:,1]*float(physical['key_activation_fraction'])
        assert np.array_equal(actual, data['actual']), 'Actual keys differ from physical replay'
    embedded_events = np.array([[n.start_time+comparison['music_offset_seconds'],
                                 n.end_time+comparison['music_offset_seconds'], n.pitch] for n in seq.notes])
    assert np.allclose(embedded_events, data['original_events'], rtol=0, atol=1e-9)
    best=summary['evaluation_selection_state']['best']
    rows=analyze(seq,data,best['evaluation']['episode_statistics'],comparison['music_offset_seconds'])
    mapping,evidence=optimize(seq,rows,1/cfg['source_runtime']['environment']['control_hz'])
    revised=music_pb2.NoteSequence.FromString(frozen)
    changes=[]
    for i,n in enumerate(revised.notes):
        if n.part!=mapping[i]:
            e=evidence[(i,mapping[i])]
            changes.append(dict(**rows[i],new_finger_id=mapping[i],new_finger_number=mapping[i]%5+1,
                                alternative_mean_coverage=e['mean_coverage'],evidence_kind=e['kind'],
                                supporting_notes=e['count'],supporting_note_indices=' '.join(map(str,e['supporting_note_indices']))))
        n.part=mapping[i]
    validation=validate(seq,revised)
    args.output.mkdir(parents=True,exist_ok=False)
    old=args.output/'original_fingered.proto';old.write_bytes(frozen)
    dest=args.output/'prelude-bwv-847_fingered.proto';dest.write_bytes(revised.SerializeToString())
    fnames=['拇指','食指','中指','无名指','小指']
    full=[]
    for i,n in enumerate(revised.notes):
        full.append(dict(note_index=i,pitch=n.pitch,note_name=name(n.pitch),start_seconds=n.start_time,
                         end_seconds=n.end_time,velocity=n.velocity,instrument=n.instrument,finger_id=n.part,
                         hand='left' if n.part>=5 else 'right',finger_number=n.part%5+1,finger_name=fnames[n.part%5]))
    write_csv(args.output/'fingering.csv',full)
    write_csv(args.output/'changes.csv',changes)
    write_csv(args.output/'note_diagnostics.csv',rows)
    groups=[]
    for pitch,f in sorted({(r['pitch'],r['finger_id']) for r in rows}):
        rr=[r for r in rows if r['pitch']==pitch and r['finger_id']==f]
        groups.append(dict(pitch=pitch,note_name=name(pitch),finger_id=f,note_count=len(rr),
                           mean_coverage=float(np.mean([r['coverage'] for r in rr])),
                           evaluation_mean_coverage=float(np.mean([r['evaluation_hold_coverage'] for r in rr])),
                           hold_coverage=sum(r['active_samples'] for r in rr)/sum(r['target_samples'] for r in rr),
                           missing_seconds=.005*sum(r['target_samples']-r['active_samples'] for r in rr),
                           changed_count=sum(c['pitch']==pitch and c['finger_id']==f for c in changes)))
    write_csv(args.output/'pitch_finger_diagnostics.csv',sorted(groups,key=lambda g:g['mean_coverage']))
    report=dict(method='Empirical hold-coverage costs + constrained per-hand dynamic programming; engineering heuristic',
                source_experiment=str(exp.resolve()),source_pose=cfg['source_runtime']['pose'],
                best_vector_step=best['vector_step'],training_transitions=summary['transitions'],
                source_sha256=sha(old),proto_sha256=sha(dest),note_count=len(seq.notes),changed_notes=len(changes),
                sample_hz=200,preparation_offset_seconds=comparison['music_offset_seconds'],
                physical_replay_micro_f1=comparison['physical_sample_micro_f1'],
                validation=validation,counts_by_finger={str(f):sum(n.part==f for n in revised.notes) for f in range(10)},
                selection=dict(max_original_coverage=.65,min_alternative_samples=3,min_alternative_mean=.65,
                               min_discounted_gain=.12,same_pitch_discount=.10,nearby_pitch_discount=.20,
                               nearby_semitones=4,keep_same_hand=True,freeze_well_held_notes=True),
                inputs={str(x.resolve()):sha(x) for x in [config_path,summary_path,comparison_path,sample_path,exp/'best.pt',exp/'video_best/replay_data/audio_substeps.npz']},
                limitations=['Observed key activation is not finger-contact attribution or proof of unreachability.',
                             'Evidence was collected with yaw=0.15rad and the old ranges, not yaw=30deg.',
                             'Alternative-finger scores use other notes in the replay and three best-checkpoint evaluation episodes; no counterfactual physical validation.',
                             'Evaluation onset misses are reported separately and are not equated to missing hold time.',
                             'Pitch, velocity, time, instrument, metadata and sustain are unchanged. No retraining was launched.'])
    (args.output/'fingering_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['source_pose','note_count','changed_notes','counts_by_finger','validation']},indent=2))
    for g in sorted(groups,key=lambda g:g['mean_coverage']):
        if g['note_count']>=5 and g['mean_coverage']<.4: print(g)


if __name__=='__main__':main()
