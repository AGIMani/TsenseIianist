"""CPU-only validation of captured defaults; no model loading or simulation."""
import argparse,json,sys
from pathlib import Path
root=Path('/home/hj/pianist');sys.path.insert(0,str(root))
import torch
from tsensei_v3_2_pianist.mjlab_env.action_mapping import _direct_contract,active_indices
from tsensei_v3_2_pianist.mjlab_env.control_limits import add_rate_arguments,rate_overrides,resolve
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter
folder=Path(__file__).resolve().parent
command=json.loads((folder/'final_launch_command.json').read_text())
p=argparse.ArgumentParser();add_rate_arguments(p)
args,_=p.parse_known_args(command[command.index('train')+1:]);args.disable_wrist_tz=args.disable_wrist_yaw=True
options=rate_overrides(None,args)
rows=_direct_contract()['channels'];names=[r['name'] for r in rows];reset=torch.zeros(53)
for side in ('R','L'):reset[names.index(side+'_Palm_Base_Ty')]=.02
channels=[side+'_Palm_Base_'+axis for side in ('R','L') for axis in ('Tx','Ty')]
for name in channels:assert options['channels'][name]==dict(max_velocity=1.99,max_acceleration=5.)
resolved=resolve(options,action_names=names,active_indices=active_indices(33,'direct_joint_v1'),physical_low=[r['physical_min'] for r in rows],physical_high=[r['physical_max'] for r in rows],reset_action=reset,control_dt=.02)
limiter=TargetMotionLimiter(resolved,num_envs=2,device='cpu',reset_action=reset)
indices=[names.index(n) for n in channels];previous=reset[indices].double().expand(2,-1).clone();velocity=torch.zeros_like(previous)
peak_a=torch.zeros(4);peak_v=torch.zeros(4)
for step in range(400):
 request=reset.expand(2,-1).clone()
 for i,index in enumerate(indices):
  request[:,index]=rows[index]['physical_max'] if (step//40+i)%2 else rows[index]['physical_min']
 issued,_=limiter.apply(request);q=issued[:,indices].double();v=(q-previous)/.02;a=(v-velocity)/.02
 assert v.abs().max()<=1.99+1e-10 and a.abs().max()<=5+1e-10
 peak_a=torch.maximum(peak_a,a.abs().amax(0));peak_v=torch.maximum(peak_v,v.abs().amax(0))
 previous=q.clone();velocity=v.clone()
assert (peak_a>4.99).all(),peak_a
other=limiter.previous[1].clone();limiter.reset(torch.tensor([0]));assert torch.equal(limiter.previous[1],other)
report=dict(passed=True,formal_training=False,physical_simulation=False,control_hz=50,num_envs=2,target_steps_per_env=400,channels=channels,max_acceleration=5.,max_velocity=1.99,observed_target_acceleration_max=peak_a.tolist(),observed_target_speed_max=peak_v.tolist(),control_limits=resolved,partial_reset_passed=True)
(folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('Bilateral 50Hz target reversals and partial reset passed:',peak_a.tolist())
