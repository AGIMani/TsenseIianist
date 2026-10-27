#!/usr/bin/env python3
"""Validate a constrained full-score candidate against raw MIDI and its baseline."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def without_parts(sequence):
    out=type(sequence)();out.CopyFrom(sequence)
    for note in out.notes:note.ClearField('part')
    return out

def gaps(sequence):
    result=[]
    for finger in range(10):
        ordered=sorted([(i,n) for i,n in enumerate(sequence.notes) if n.part==finger],key=lambda x:(x[1].start_time,x[0]))
        for (i,a),(j,b) in zip(ordered,ordered[1:]):
            gap=b.start_time-a.end_time
            if gap<0:raise AssertionError(f'Overlapping finger {finger}: {i}, {j}')
            if gap<.05:
                result.append(dict(finger=finger,previous_note=i,next_note=j,
                    previous_pitch=a.pitch,next_pitch=b.pitch,gap_seconds=gap,
                    onset_seconds=b.start_time,same_pitch=a.pitch==b.pitch))
    return sorted(result,key=lambda g:(g['onset_seconds'],g['finger']))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--midi',type=Path,required=True);p.add_argument('--bank-report',type=Path,required=True)
    args=p.parse_args()
    from note_seq import midi_io
    from note_seq.protobuf import music_pb2
    raw=midi_io.midi_file_to_note_sequence(str(args.midi));baseline=music_pb2.NoteSequence();candidate=music_pb2.NoteSequence()
    baseline.ParseFromString((args.baseline/'turkish-march-advanced_fingered.proto').read_bytes())
    candidate.ParseFromString((args.plan/'turkish-march-advanced_fingered.proto').read_bytes())
    assert len(raw.notes)==len(candidate.notes)==len(baseline.notes)==2544
    assert without_parts(candidate)==without_parts(raw)==without_parts(baseline)
    rows=list(csv.DictReader((args.plan/'fingering.csv').open()))
    for i,(row,note) in enumerate(zip(rows,candidate.notes)):
        assert int(row['note_index'])==i and int(row['finger_id'])==note.part
        assert int(row['pitch'])==note.pitch and int(row['instrument'])==note.instrument
        assert float(row['start_seconds'])==note.start_time and float(row['end_seconds'])==note.end_time
        assert int(row['velocity'])==note.velocity and note.part//5==note.instrument
    constraints=json.loads((args.plan/'fixed_fingering.json').read_text())
    assert len(constraints['assignments'])==28
    assert all(candidate.notes[int(i)].part==f for i,f in constraints['assignments'].items())
    changed=[dict(note_index=i,pitch=n.pitch,start_seconds=n.start_time,end_seconds=n.end_time,
        previous_finger=old.part,new_finger=n.part) for i,(old,n) in enumerate(zip(baseline.notes,candidate.notes)) if old.part!=n.part]
    old_gaps,new_gaps=gaps(baseline),gaps(candidate)
    old_keys={(x['finger'],x['previous_note'],x['next_note']) for x in old_gaps}
    new_keys={(x['finger'],x['previous_note'],x['next_note']) for x in new_gaps}
    old_pairs={(x['previous_note'],x['next_note']) for x in old_gaps}
    new_pairs={(x['previous_note'],x['next_note']) for x in new_gaps}
    new_only=[x for x in new_gaps if (x['finger'],x['previous_note'],x['next_note']) not in old_keys]
    removed=[x for x in old_gaps if (x['finger'],x['previous_note'],x['next_note']) not in new_keys]
    bank=json.loads(args.bank_report.read_text())
    assert bank['source_fingering_sha256']==sha(args.baseline/'fingering.csv')
    bank_changes=[dict(time_seconds=pose['time_seconds'],note_index=n['index'],bank_finger=n['finger'],candidate_finger=candidate.notes[n['index']].part)
        for pose in bank['poses'] for n in pose['notes'] if candidate.notes[n['index']].part!=n['finger']]
    count=lambda sequence:sum(n.part in (0,5) for n in sequence.notes)
    report=dict(schema='a32_contact_refined_input_validation_v1',pid=os.getpid(),validator_sha256=sha(__file__),
        note_count=2544,original_midi_sha256=sha(args.midi),source_baseline_proto_sha256=sha(args.baseline/'turkish-march-advanced_fingered.proto'),
        final_proto_sha256=sha(args.plan/'turkish-march-advanced_fingered.proto'),
        all_note_fields_except_part_equal_raw=True,all_sequence_metadata_equal_raw=True,
        simultaneous_finger_conflicts=0,fixed_note_count=28,all_fixed_assignments_satisfied=True,
        baseline_thumb_count=count(baseline),thumb_count=count(candidate),thumb_fraction=count(candidate)/2544,
        counts_by_finger={str(f):sum(n.part==f for n in candidate.notes) for f in range(10)},
        changed_note_count=len(changed),changed_notes=changed,
        first_3s_fingering_equals_baseline=not any(x['start_seconds']<3 for x in changed),
        first_20s_fingering_equals_baseline=not any(x['start_seconds']<20 for x in changed),
        bank_v6_all_21_onset_assignments_equal=not bank_changes,bank_v6_assignment_differences=bank_changes,
        recovery_preference_seconds=.05,baseline_short_reuse_count=len(old_gaps),short_reuse_count=len(new_gaps),
        net_short_reuse_change=len(new_gaps)-len(old_gaps),new_short_reuse_count=len(new_only),removed_short_reuse_count=len(removed),
        first20_short_reuse_count=sum(x['onset_seconds']<20 for x in new_gaps),
        new_short_reuse_note_pair_count_ignoring_finger_relabel=len(new_pairs-old_pairs),
        removed_short_reuse_note_pair_count_ignoring_finger_relabel=len(old_pairs-new_pairs),
        short_reuse_forced_by_fixed_note_pair=[x for x in new_gaps if str(x['previous_note']) in constraints['assignments'] and str(x['next_note']) in constraints['assignments']],
        fixed_repeat_limitation='Listed fixed repeated C#6 note pairs have 1.04166666668ms release gaps and identical fixed pinky assignment. These reuses follow from these specific fixed assignments; they do not prove no alternative hand distribution or physically feasible approach exists.',
        baseline_short_reuse=old_gaps,final_short_reuse=new_gaps,new_short_reuse=new_only,removed_short_reuse=removed,
        limitations=['All 2544 non-part note fields and all sequence metadata compare equal to raw MIDI import. No onset/end quantization.',
            'Two finger/key shapes have static finite-top-contact evidence; repeated occurrences transfer that same geometry only. Neighboring DP changes and full-song dynamics are unvalidated.',
            '50ms is a soft scheduling preference, not a calibrated physical bound. Short reuse remains and is reported.',
            'Current physical playback has not passed full activation and collision requirements; this is a candidate, not an executable or safe trajectory.'])
    (args.plan/'input_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    summary_keys=['note_count','thumb_count','changed_note_count','first_3s_fingering_equals_baseline','first_20s_fingering_equals_baseline','bank_v6_all_21_onset_assignments_equal','baseline_short_reuse_count','short_reuse_count','net_short_reuse_change','new_short_reuse_count','removed_short_reuse_count','first20_short_reuse_count']
    planning=json.loads((args.plan/'report.json').read_text())
    planning['contact_refinement']={k:report[k] for k in summary_keys}
    planning['contact_refinement']['evidence']=constraints['evidence']
    planning['contact_refinement']['dynamic_validation_passed']=False
    planning['contact_refinement']['short_reuse_forced_by_fixed_note_pair_count']=len(report['short_reuse_forced_by_fixed_note_pair'])
    planning['contact_refinement']['new_short_reuse_note_pairs_ignoring_finger_relabel']=len(new_pairs-old_pairs)
    planning['contact_refinement']['removed_short_reuse_note_pairs_ignoring_finger_relabel']=len(old_pairs-new_pairs)
    planning['limitations'].extend(x for x in report['limitations'][1:] if x not in planning['limitations'])
    planning['outputs']['input_validation.json']=sha(args.plan/'input_validation.json')
    planning['outputs']['fixed_fingering.json']=sha(args.plan/'fixed_fingering.json')
    (args.plan/'report.json').write_text(json.dumps(planning,indent=2)+'\n')
    shutil.copy2(args.midi,args.plan/'original.mid')
    print(json.dumps({k:report[k] for k in summary_keys},indent=2))

if __name__=='__main__':main()
