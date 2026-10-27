#!/usr/bin/env python3
"""Independent full-score preservation, context, and repeated-pattern checks."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[0]))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def without_part(sequence):
    clone=type(sequence)();clone.CopyFrom(sequence)
    for n in clone.notes:n.ClearField('part')
    return clone

def main():
    p=argparse.ArgumentParser();p.add_argument('--midi',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--audit-script',type=Path,required=True);p.add_argument('--bank-report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import importlib.util
    spec=importlib.util.spec_from_file_location('repeat_audit_validation',a.audit_script);audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
    from note_seq import midi_io
    from note_seq.protobuf import music_pb2
    raw=midi_io.midi_file_to_note_sequence(str(a.midi))
    def read(folder):
        seq=music_pb2.NoteSequence();seq.ParseFromString((folder/'turkish-march-advanced_fingered.proto').read_bytes());return seq
    baseline,candidate=read(a.baseline),read(a.plan)
    assert len(raw.notes)==len(candidate.notes)==len(baseline.notes)
    assert without_part(raw)==without_part(candidate)==without_part(baseline)
    notes,bars,metadata,seconds=audit.parse_midi(a.midi,a.plan/'fingering.csv')
    assert len(notes)==len(candidate.notes)
    for n,proto in zip(notes,candidate.notes):
        assert n.index==notes.index(n) and (n.hand,n.pitch,n.start_seconds,n.end_seconds,n.finger,n.velocity)==(proto.instrument,proto.pitch,proto.start_time,proto.end_time,proto.part,proto.velocity)
    constraints=json.loads((a.plan/'fixed_fingering.json').read_text())
    old_fixed=json.loads((a.baseline/'fixed_fingering.json').read_text())['assignments']
    mapping={n.index:n.finger for n in notes};fixed={int(i):f for i,f in constraints['assignments'].items()}
    assert all(fixed[int(i)]==f for i,f in old_fixed.items())
    assert audit.assignment_issue(notes,mapping,fixed) is None
    old_mapping={i:n.part for i,n in enumerate(baseline.notes)}
    old_gaps=audit.short_reuses(notes,old_mapping);new_gaps=audit.short_reuses(notes,mapping)
    key=lambda g:(g['finger'],g['previous_note'],g['next_note'])
    old_keys={key(g) for g in old_gaps};new_keys={key(g) for g in new_gaps}
    changed=[dict(index=i,pitch=n.pitch,time_seconds=n.start_time,old_finger=baseline.notes[i].part,new_finger=n.part) for i,n in enumerate(candidate.notes) if n.part!=baseline.notes[i].part]
    windows=audit.windows_for(notes,bars);groups=audit.group_windows(windows,'exact')
    inconsistent=lambda mp:[audit.pattern_record(k,v,mp,[],seconds)['id'] for k,v in groups.items() if len({tuple(mp[n.index] for n in w.notes) for w in v})>1]
    old_patterns,new_patterns=set(inconsistent(old_mapping)),set(inconsistent(mapping))
    bank_changes=[]
    if a.bank_report:
        bank=json.loads(a.bank_report.read_text())
        bank_changes=[dict(time_seconds=pose['time_seconds'],index=n['index'],old_finger=n['finger'],new_finger=mapping[n['index']]) for pose in bank['poses'] for n in pose['notes'] if mapping[n['index']]!=n['finger']]
    old_report=json.loads((a.baseline/'report.json').read_text());new_report=json.loads((a.plan/'report.json').read_text())
    old_reach=json.loads((a.baseline/'reach_diagnostics.json').read_text())['onsets'];new_reach=json.loads((a.plan/'reach_diagnostics.json').read_text())['onsets']
    old_lookup={(r['hand'],r['time_seconds']):r for r in old_reach}
    new_reach_crossings=[dict(hand=r['hand'],time_seconds=r['time_seconds'],old_selected_mm=old_lookup[(r['hand'],r['time_seconds'])]['selected_max_residual_m']*1000,new_selected_mm=r['selected_max_residual_m']*1000,fingers=r['selected_fingers']) for r in new_reach if r['selected_max_residual_m']>.01 and old_lookup[(r['hand'],r['time_seconds'])]['selected_max_residual_m']<=.01]
    out=dict(schema='fullscore_repetition_validation_v1',pid=os.getpid(),validator_sha256=sha(__file__),audit_script_sha256=sha(a.audit_script),
        source_midi_sha256=sha(a.midi),baseline_proto_sha256=sha(a.baseline/'turkish-march-advanced_fingered.proto'),candidate_proto_sha256=sha(a.plan/'turkish-march-advanced_fingered.proto'),candidate_csv_sha256=sha(a.plan/'fingering.csv'),
        note_count=len(notes),duration_seconds=max(n.end_seconds for n in notes),all_non_part_note_and_sequence_fields_equal_raw=True,exact_tick_alignment=True,
        zero_finger_occupancy_overlap=True,natural_order_and_held_note_checks_passed=True,fixed_assignments_satisfied=len(fixed),old_static_constraints_retained=len(old_fixed),
        thumb_count=sum(n.finger%5==0 for n in notes),baseline_thumb_count=sum(f%5==0 for f in old_mapping.values()),counts_by_finger={str(f):sum(n.finger==f for n in notes) for f in range(10)},
        changed_note_count=len(changed),changed_notes=changed,first3_identical=not any(r['time_seconds']<3 for r in changed),first20_identical=not any(r['time_seconds']<20 for r in changed),
        bank_v6_assignments_equal=not bank_changes if a.bank_report else None,bank_v6_assignment_changes=bank_changes,
        baseline_short_reuses=len(old_gaps),short_reuses=len(new_gaps),new_short_reuses=[g for g in new_gaps if key(g) not in old_keys],removed_short_reuses=[g for g in old_gaps if key(g) not in new_keys],
        short_reuse_fixed_wide_pairs=[g for g in new_gaps if str(g['previous_note']) in old_fixed and str(g['next_note']) in old_fixed],
        baseline_inconsistent_groups=len(old_patterns),inconsistent_groups=len(new_patterns),new_inconsistent_pattern_ids=sorted(new_patterns-old_patterns),resolved_inconsistent_pattern_ids=sorted(old_patterns-new_patterns),
        selected_residual_new_crossings_above10mm=new_reach_crossings,
        baseline_hands=old_report['hands'],candidate_hands=new_report['hands'],
        physical_certification=False,limitations=['The repeat definition preserves exact MIDI ticks and carry-over notes. Velocity remains unchanged but does not define a fingering pattern.',
            'DP reach uses a sampled fixed reward-target proxy, not true key-top contact or collision validation. Its residuals and soft movement costs cannot certify dynamics.',
            'Short reuse remains; 50ms is a soft preference. Exact repeated fingering is only reused when whole-score held-note and fixed constraints permit it.',
            'New assignments and references need fresh physical playback. The previous first20s recording is not a recording of this candidate.'])
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['note_count','thumb_count','changed_note_count','first3_identical','first20_identical','bank_v6_assignments_equal','short_reuses','inconsistent_groups','fixed_assignments_satisfied']},indent=2))
if __name__=='__main__':main()
