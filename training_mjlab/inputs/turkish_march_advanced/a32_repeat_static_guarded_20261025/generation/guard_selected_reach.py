#!/usr/bin/env python3
"""Preserve baseline fingers where a trial creates a new >10mm selected miss."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--constraints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();old=json.loads((a.baseline/'reach_diagnostics.json').read_text())['onsets'];new=json.loads((a.trial/'reach_diagnostics.json').read_text())['onsets'];lookup={(r['hand'],r['time_seconds']):r for r in old}
mapping={r['note_index']:int(r['finger_id']) for r in csv.DictReader((a.baseline/'fingering.csv').open())};out=json.loads(a.constraints.read_text());guards=[]
for r in new:
    previous=lookup[(r['hand'],r['time_seconds'])]
    if previous['selected_max_residual_m']<=.01<r['selected_max_residual_m']:
        for i in r['note_indices']:
            if str(i) in out['assignments'] and out['assignments'][str(i)]!=mapping[str(i)]:raise ValueError('Reach guard conflicts with an accepted repeat constraint')
            out['assignments'][str(i)]=mapping[str(i)]
        guards.append(dict(hand=r['hand'],time_seconds=r['time_seconds'],note_indices=r['note_indices'],baseline_fingers=previous['selected_fingers'],rejected_fingers=r['selected_fingers'],baseline_selected_residual_m=previous['selected_max_residual_m'],rejected_selected_residual_m=r['selected_max_residual_m']))
out['selected_reach_guards']=out.get('selected_reach_guards',[])+guards
out['reach_guard_limitation']='This preserves baseline assignments at newly worsened sampled target residuals; it is not a certified physical reach threshold.'
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'guarded_onsets':len(guards),'fixed_notes':len(out['assignments'])}))
