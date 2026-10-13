"""Detection-only hand contacts must keep penalties without physical constraints."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from tsensei_v3_2_pianist.detection_only_collision import apply_to_xml, disable_same_hand_response

XML = '''<mujoco><worldbody>
<body name="right_/a"><freejoint/><geom name="a" type="sphere" size=".1"/></body>
<body name="right_/b" pos=".15 0 0"><freejoint/><geom name="b" type="sphere" size=".1"/></body>
<body name="piano/key" pos="-.15 0 0"><freejoint/><geom name="key" type="sphere" size=".1"/></body>
</worldbody></mujoco>'''


class DetectionChecks(unittest.TestCase):
    def test_inactive_pair_keeps_detection_and_piano_force(self):
        root=ET.fromstring(XML);original=mujoco.MjModel.from_xml_string(XML)
        self.assertEqual(apply_to_xml(root,original),1)
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        self_contact=piano_contact=False
        for i,c in enumerate(data.contact):
            names={model.geom(g).name for g in c.geom}
            if names=={'a','b'}:
                self_contact=True;self.assertLess(c.dist,0);self.assertEqual(c.efc_address,-1)
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                np.testing.assert_array_equal(force,0)
            if names=={'a','key'}:
                piano_contact=True;self.assertGreaterEqual(c.efc_address,0)
        self.assertTrue(self_contact and piano_contact)

    def test_masks_only_remove_same_hand(self):
        model=mujoco.MjModel.from_xml_string(XML);disable_same_hand_response(model)
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        names=[{model.geom(g).name for g in c.geom} for c in data.contact]
        self.assertNotIn({'a','b'},names);self.assertIn({'a','key'},names)

    def test_gpu_query_penalizes_without_main_contacts(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA required')
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        root=Path(__file__).resolve().parents[1]/'training_mjlab/outputs'
        if not (root/'scene_twinkle_h12_20hz200hz_20260924').exists():self.skipTest('Frozen scene required')
        for graph in (False,True):
            env=PianoTaskEnv(root/'scene_twinkle_h12_20hz200hz_20260924',root/'task_twinkle_h12_20hz200hz_20260924',
                num_envs=2,disable_wrist_tz=True,physics_graph=graph,action_profile='mixed_v1',
                self_collision_weight=.2,self_collision_detection_only=True)
            s=env.scene;total=0
            gen=torch.Generator(device='cuda:0').manual_seed(42)
            sides=torch.tensor([1 if s.model.body(int(b)).name.startswith('right_/') else
                                2 if s.model.body(int(b)).name.startswith('left_/') else 0
                                for b in s.model.geom_bodyid],device=s.device)
            for _ in range(20):
                result=env.step(torch.rand((2,49),device=s.device,generator=gen)*2-1+s.action_shift)
                self.assertTrue(result.observation.isfinite().all())
                total+=result.info['reward_terms']['self_collision_reward'].sum().item()
                geoms=s.data.contact.geom[:].long().clamp_min(0)
                a,b=sides[geoms[:,0]],sides[geoms[:,1]]
                valid=torch.arange(len(a),device=s.device)<s.data.nacon[0]
                self.assertFalse(((a==b)&(a>0)&valid).any())
            self.assertLess(total,0)
            env.reset([0])


if __name__=='__main__':unittest.main()
