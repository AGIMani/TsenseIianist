#!/usr/bin/env python3
"""CPU/GPU regression: hand overlap has no force but still loses forearm bonus."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco,torch, numpy as np, xml.etree.ElementTree as ET
from tsensei_v3_2_pianist.detection_only_collision import apply_to_xml
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
xml='<mujoco><worldbody><body name="right_/a"><freejoint/><geom name="a" type="sphere" size=".1"/></body><body name="left_/b" pos=".15 0 0"><freejoint/><geom name="b" type="sphere" size=".1"/></body><body name="piano/key" pos="-.15 0 0"><freejoint/><geom name="key" type="sphere" size=".1"/></body></worldbody></mujoco>'
r=ET.fromstring(xml);m=mujoco.MjModel.from_xml_string(xml);assert apply_to_xml(r,m,include_cross_hand=True)==1
m=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
for c in d.contact:
 names={m.geom(i).name for i in c.geom}
 if names=={'a','b'}: assert c.efc_address==-1
 if names=={'a','key'}: assert c.efc_address>=0
print('PASS CPU cross hand inactive / piano active',flush=True)
for graph in (False,True):
 base=str(ROOT/'training_mjlab/outputs')+'/'
 e=PianoTaskEnv(base+'scene_twinkle_h12_20hz200hz_20260924',base+'task_twinkle_h12_20hz200hz_20260924',num_envs=2,disable_wrist_tz=True,hand_collision_detection_only=True,action_profile='mixed_v1',physics_graph=graph,self_collision_weight=0)
 s=e.scene;m=s.model
 physical=s.bundle.neutral_action.copy();physical[20]=-.15;physical[45]=.145
 act=s.policy_action(physical).cpu().numpy()
 cross=False;piano=False
 for _ in range(35):
  z=e.step(s.tensor(act).expand(2,-1));assert z.observation.isfinite().all()
  cross |= bool(e.snapshot()['palm_collision'].any())
  geoms=s.data.contact.geom[:].long().clamp_min(0);valid=torch.arange(len(geoms),device=s.device)<s.data.nacon[0]
  hands=torch.tensor([m.body(int(b)).name.startswith(('right_/','left_/')) for b in m.geom_bodyid],device=s.device)
  assert not ((hands[geoms[:,0]]&hands[geoms[:,1]])&valid).any()
  native=s.cpu_snapshot();mujoco.mj_forward(m,native)
  piano |= any(c.efc_address>=0 for c in native.contact)
 assert cross, 'Lost cross-hand reward detection'
 print('PASS GPU graph',graph,'cross reward detected',cross,'last forearm',z.info['reward_terms']['forearm_reward'].tolist(),flush=True)
