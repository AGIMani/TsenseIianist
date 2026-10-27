#!/usr/bin/env python3
"""Filter exact-repeat proposals using the independently recorded reach review."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',type=int,choices=(0,1,2),default=0);p.add_argument('--exclude',action='append',default=[])
a=p.parse_args();proposal=json.loads((a.audit/'repeat_constraints.json').read_text());audit=json.loads((a.audit/'audit.json').read_text());baseline=json.loads((a.baseline/'fixed_fingering.json').read_text())
assignments=dict(baseline['assignments']);accepted=[]
for group in proposal['selected_patterns']:
    if group['pattern_id']=='0c382d025498' or group['pattern_id'] in a.exclude:continue
    for i in group['fixed_note_indices']:assignments[str(i)]=proposal['assignments'][str(i)]
    accepted.append(group)
if a.variant:
    pattern=next(r for r in audit['strict_groups'] if r['id']=='0c382d025498')
    variant=pattern['fingering_variants'][a.variant]
    for ids in pattern['note_indices_by_occurrence']:
        assignments.update({str(i):f for i,f in zip(ids,variant['fingers'])})
    accepted.append(dict(pattern_id=pattern['id'],alternative_existing_template=variant))
out=dict(schema='a32_reach_reviewed_repetition_constraints_v1',source_midi_sha256=proposal['source_midi_sha256'],assignments=assignments,
    extra_excluded_pattern_ids=a.exclude,
    selected_patterns=accepted,excluded_patterns=[dict(id='0c382d025498',rejected_fingers=[9,8,5,7,6,8,5],
    reason='First proposal replicated the ring/index C#4/E4 mapping to four additional onsets, raising selected sampled residual from 2.790 to 10.418mm. Keep this proposal separate; do not automatically accept its fewer thumbs/short gaps.')],
    scope='Exact MIDI tick patterns only, old 28 static-contact constraints retained; full DP reoptimizes unfixed neighbors.')
a.output.write_text(json.dumps(out,indent=2)+'\n')
