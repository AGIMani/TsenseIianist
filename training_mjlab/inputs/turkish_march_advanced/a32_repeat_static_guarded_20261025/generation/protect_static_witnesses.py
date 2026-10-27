#!/usr/bin/env python3
"""Protect known baseline static witnesses after paired failed new searches."""
import argparse,csv,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-bank',type=Path,required=True);p.add_argument('--paired-report',type=Path,required=True);p.add_argument('--constraints',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--additional-baseline-onset',type=float,action='append',default=[])
a=p.parse_args();pairs=json.loads(a.paired_report.read_text());v5=json.loads(a.constraints.read_text());old=json.loads((a.baseline/'fixed_fingering.json').read_text())
mapping={r['note_index']:int(r['finger_id']) for r in csv.DictReader((a.baseline/'fingering.csv').open())}
rows=[json.loads(x) for x in (a.baseline_bank/'occurrences.jsonl').read_text().splitlines()];bytime={r['time_seconds']:r for r in rows}
restore={};times=[]
for pair in pairs['results']:
    if not pair['results']['candidate']['static_passed']:
        for t in pair['occurrence_seconds']:
            assert bytime[t]['static_passed'];times.append(t)
            for n in bytime[t]['notes']:restore[str(n['index'])]=n['finger']
for t in a.additional_baseline_onset:
    assert bytime[t]['static_passed'];times.append(t)
    for n in bytime[t]['notes']:restore[str(n['index'])]=n['finger']
conflicts={i for i,f in restore.items() if i in v5['assignments'] and v5['assignments'][i]!=f}
retained=[];dropped=[]
for pattern in v5['selected_patterns']:
    (dropped if any(str(i) in conflicts for i in pattern['fixed_note_indices']) else retained).append(pattern)
assignments=dict(old['assignments'])
for pattern in retained:
    for i in pattern['fixed_note_indices']:assignments[str(i)]=v5['assignments'][str(i)]
for guard in v5['selected_reach_guards']:
    for i in guard['note_indices']:assignments[str(i)]=mapping[str(i)]
for i,f in restore.items():
    assert i not in assignments or assignments[i]==f
    assignments[i]=f
out={**v5,'assignments':assignments,'selected_patterns':retained,'released_repetition_families_for_static_protection':[p['pattern_id'] for p in dropped],
    'static_witness_protection':dict(baseline_bank=str(a.baseline_bank),paired_report_sha256=hashlib.sha256(a.paired_report.read_bytes()).hexdigest(),
        protected_occurrence_seconds=sorted(set(times)),protected_active_notes=restore,additional_baseline_onsets=a.additional_baseline_onset,scope='Baseline full bilateral pose was revalidated; new assignment failed equal-source/equal-budget paired search, or the later full-bank pass exposed another occurrence of that same failed combination. Restore known witness fingers without claiming global infeasibility or dynamic success.')}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'fixed':len(assignments),'restored_active':len(restore),'occurrences':len(times),'released_families':out['released_repetition_families_for_static_protection']}))
