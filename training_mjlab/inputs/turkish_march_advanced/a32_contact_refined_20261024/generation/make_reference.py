"""Make scratch seeds and a smoothed wrist request; does not integrate physics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
g=json.loads(a.geometry.read_text());plan=json.loads((a.plan/'candidate_wrist_anchors.json').read_text());report=json.loads((a.plan/'report.json').read_text())
if hashlib.sha256(a.geometry.read_bytes()).hexdigest()!=plan['geometry_sha256']:raise ValueError('Geometry differs from selected plan')
anchors=plan['anchors'];duration=max(r['time_seconds'] for r in anchors)+.25
times=np.arange(int(np.ceil(duration/.05))+1)*.05;actions=np.tile(np.asarray(g['neutral_action']),(len(times),1))
for side in (0,1):
 rows=[r for r in anchors if r['hand']==('right','left')[side]];at=np.array([r['time_seconds'] for r in rows]);ids=np.maximum(np.searchsorted(at,times+1e-10,side='right')-1,0)
 for step,j in enumerate(ids):
  r=rows[j];off=side*25;actions[step,off+20:off+22]=r['wrist_tx_ty_m'];actions[step,off+23]=r['wrist_pitch_rad']
  for note,finger in r['note_fingers'].items():
   sample=g['reachable_samples'][finger][r['sample_indices'][note]]['action'];ix=off+(finger%5)*4
   actions[step,ix:ix+4]=sample[ix:ix+4]
commands=actions.copy();cols=[20,21,23,45,46,48];commands[:,cols]=gaussian_filter1d(commands[:,cols],3,axis=0,mode='nearest')
np.savez_compressed(a.output,time=times,actions=actions,all_command_times=times,all_commands=commands)
meta=dict(score_sha256=report['outputs']['turkish-march-advanced_fingered.proto'] if 'outputs' in report else hashlib.sha256((a.plan/'turkish-march-advanced_fingered.proto').read_bytes()).hexdigest(),geometry_sha256=plan['geometry_sha256'],source_anchors_sha256=hashlib.sha256((a.plan/'candidate_wrist_anchors.json').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),dt=.05,wrist_smoothing_sigma_s=.15,description='Unsmoothed sampled finger actions are scratch IK initial guesses only. Smoothed wrists are requests; current runtime target limiter still applies. Actual key state is never assigned.',duration_s=duration)
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
