"""Validate actual CLI reward overrides with CPU reward fixtures only."""
import json,sys
from copy import deepcopy
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import torch
from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tests.test_paper_rewards import scene_fixture
r=Path(__file__).parent;cmd=json.loads((r/'final_launch_command.json').read_text())
a,_=parser().parse_known_args(cmd[cmd.index('train')+1:]);c=effective_config(a)
term='joint_motion_limit_penalty';p=c['reward_config']['terms'][term]['params']
assert p['tcp_acceleration_limit']==4.75 and p['tcp_velocity_limit']==1.90
rows=c['control_limits']['channels'];names=[s+'_Palm_Base_'+axis for s in ('R','L') for axis in ('Tx','Ty')]
for name in names:assert rows[name]['max_acceleration']==5 and rows[name]['max_velocity']==1.99
assert 4.75==.95*5
for name in names:
 for sign in (-1,1):
  scene=scene_fixture();scene['control_dt']=.02
  isolated=deepcopy(c['reward_config'])
  for k,v in isolated['terms'].items():v['enabled']=k==term
  manager=RewardManager(isolated,scene,{},num_envs=4,device='cpu')
  action=torch.tensor(scene['neutral_action']).repeat(4,1);requested=action.clone()
  requested[:,manager.actions.index(name)]+=sign*4.75*torch.tensor([.99,1.,1.05,1.1])*.02**2
  value=manager({},action,dict(requested_physical_action=requested),torch.zeros(4,89),torch.zeros(4,10,88))[term]
  torch.testing.assert_close(value,torch.tensor([0.,0.,-2.5,-5.]),atol=4e-5,rtol=1e-5)
report=dict(passed=True,formal_training=False,physical_simulation=False,only_parameter_change={'tcp_acceleration_limit':[1.90,4.75]},hard_acceleration=5.,hard_velocity=1.99,soft_velocity=1.90,ratio=.95,channels=names,reward_threshold_checks=8,reward_config=c['reward_config'],control_limits=c['control_limits'])
(r/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('Actual defaults and both signs of all four TCP reward channels passed.')
