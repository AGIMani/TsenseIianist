"""Physical clipping/slew contracts and applied-action history, CPU plus optional GPU."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical,resolve,TargetLimiter
from tsensei_v3_2_pianist.mjlab_env.reward_config import template
from tests.test_paper_rewards import scene_fixture


def config(**channels):
    return dict(schema=1,mode='target_clip_slew_v1',channels=channels)


class ControlLimitsTest(unittest.TestCase):
    def setUp(self):
        self.scene=scene_fixture()
        self.scene['action_min'][20]=-.7605;self.scene['action_max'][20]=.4605
        self.scene['action_min'][21]=0.;self.scene['action_max'][21]=.15
        self.scene['neutral_action'][21]=.04
        self.active=[i for i in range(50) if i not in (22,24,47,49)]

    def resolve(self,value):
        return resolve(value,action_names=self.scene['action_names'],active_indices=self.active,
            physical_low=self.scene['action_min'],physical_high=self.scene['action_max'],
            reset_action=self.scene['neutral_action'],control_dt=.05)

    def test_none_identity_and_canonical_determinism(self):
        self.assertIsNone(canonical(None));self.assertIsNone(self.resolve(None))
        a=config(R_Palm_Base_Tx={'max_velocity':1,'max':.1,'min':-.1})
        self.assertEqual(canonical(a),canonical(json.loads(json.dumps(a))))
        self.assertEqual(a['channels']['R_Palm_Base_Tx']['max_velocity'],1)
        resolved=self.resolve(a)
        self.assertEqual(resolved['channels']['R_Palm_Base_Tx']['velocity_unit'],'m/s')
        self.assertEqual(resolved['channels']['R_Palm_Base_Tx']['original_min'],-.7605)
        self.assertEqual(resolved['channels']['R_Palm_Base_Tx']['reset_target'],0.)

    def test_invalid_fields_names_fixed_channels_and_ranges(self):
        values=[{},dict(schema=True,mode='target_clip_slew_v1',channels={'R_Palm_Base_Tx':{'min':0}}),
            config(sustain={'max_velocity':1}),config(R_Palm_Base_Tz={'max_velocity':1}),
            config(R_Palm_Base_Yaw={'max_velocity':1}),config(unknown={'max_velocity':1}),
            config(R_Palm_Base_Tx={'min':.01}),config(R_Palm_Base_Ty={'max':.03}),
            config(R_Palm_Base_Tx={'max':2.}),config(R_Palm_Base_Tx={'min':0.,'max':0.}),
            config(R_Palm_Base_Tx={'max_velocity':0.}),config(R_Palm_Base_Tx={'max_velocity':float('nan')}),
            config(R_Palm_Base_Tx={'min':True}),config(R_Palm_Base_Tx={'typo':1}),
            config(R_Palm_Base_Tx={})]
        for value in values:
            with self.subTest(value=value),self.assertRaises(ValueError):self.resolve(value)

    def test_clip_then_rate_limit_from_previous_issued_target(self):
        limits=self.resolve(config(R_Palm_Base_Tx={'min':-.1,'max':.1,'max_velocity':.2},
                                  R_Palm_Base_Ty={'max':.08}))
        limiter=TargetLimiter(limits,device='cpu')
        previous=torch.tensor(self.scene['neutral_action']).expand(2,-1).clone()
        previous[1,20]=.04
        requested=previous.clone();requested[:,20]=.5;requested[:,21]=.12;requested[:,0]=.7
        applied,clip,slew=limiter.apply(requested,previous)
        torch.testing.assert_close(applied[:,20],torch.tensor([.01,.05]))
        torch.testing.assert_close(applied[:,21],torch.full((2,),.08))
        torch.testing.assert_close(applied[:,0],torch.full((2,),.7))
        self.assertTrue(clip[:,20:22].all());self.assertTrue(slew[:,20].all())
        self.assertFalse(slew[:,21].any());self.assertFalse(clip[:,:20].any())
        self.assertEqual(requested[0,20].item(),.5)
        # apply itself never advances hidden state. A second evaluation is identical.
        torch.testing.assert_close(limiter.apply(requested,previous)[0],applied)
        next_applied=limiter.apply(requested,applied)[0]
        torch.testing.assert_close(next_applied[:,20],torch.tensor([.02,.06]))

    def test_partial_reset_uses_actual_reset_and_leaves_other_world(self):
        limiter=TargetLimiter(self.resolve(config(R_Palm_Base_Tx={'max_velocity':.2})),device='cpu')
        previous=torch.tensor(self.scene['neutral_action']).expand(2,-1).clone()
        requested=previous.clone();requested[:,20]=.4
        previous=limiter.apply(requested,previous)[0]
        previous[0]=torch.tensor(self.scene['neutral_action'])
        applied=limiter.apply(requested,previous)[0]
        torch.testing.assert_close(applied[:,20],torch.tensor([.01,.02]))


class GPUControlLimitsTest(unittest.TestCase):
    def test_graph_applied_action_observation_and_reset_isolation(self):
        if not torch.cuda.is_available():self.skipTest('Requires scheduled CUDA window')
        sys.path.insert(0,str(Path(__file__).parent))
        from test_preparation import write_fixture
        from test_preparation_policy import write_task,POLICY
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        with tempfile.TemporaryDirectory() as temporary:
            scene=Path(temporary)/'scene';scene.mkdir();write_fixture(scene)
            task=Path(temporary)/'task';write_task(scene,task)
            options=dict(num_envs=2,disable_wrist_tz=True,disable_wrist_yaw=True,preparation=POLICY)
            control=config(R_Thumb_Rot={'min':0.,'max':.3,'max_velocity':1.})
            reward = template()
            reward['terms']['key_press_reward']['enabled'] = False
            # Simulate a historical task whose reset action observation differs
            # from real ctrl; the limiter must expose true previous applied state.
            tm=json.loads((task/'task.json').read_text());tm['reset_action']=[-.25]*51
            (task/'task.json').write_text(json.dumps(tm))
            for graph in (False,True):
                env=PianoTaskEnv(scene,task,physics_graph=graph,control_limits=control,
                                 reward_config=reward,**options)
                obs=env.reset()
                self.assertAlmostEqual(env.logic.last_action[0,0].item(),.1,places=6)
                self.assertAlmostEqual(obs[0,0].item(),.1,places=6)
                action=env.scene.policy_action(env.scene.bundle.neutral_action).expand(2,-1).clone();action[:,0]=1.
                one=env.step(action)
                torch.testing.assert_close(one.info['applied_physical_action'][:,0],torch.full((2,),.15,device='cuda'))
                torch.testing.assert_close(one.info['requested_physical_action'][:,0],torch.full((2,),.5,device='cuda'))
                torch.testing.assert_close(env.logic.last_action,one.info['applied_physical_action'])
                torch.testing.assert_close(one.observation[:,0],torch.full((2,),.15,device='cuda'))
                self.assertTrue(one.info['control_clip_mask'][:,0].all())
                self.assertTrue(one.info['control_slew_mask'][:,0].all())
                previous=env.scene.last_applied_physical_action[1].clone()
                qpos=env.scene.data.qpos[1].clone()
                history=env.logic.history[1].clone()
                env.reset([0])
                torch.testing.assert_close(env.scene.last_applied_physical_action[1],previous,atol=0,rtol=0)
                torch.testing.assert_close(env.scene.data.qpos[1],qpos,atol=0,rtol=0)
                torch.testing.assert_close(env.logic.history[1],history,atol=0,rtol=0)
                two=env.step(action)
                torch.testing.assert_close(two.info['applied_physical_action'][:,0],torch.tensor([.15,.2],device='cuda'))
                # Step one's returned diagnostic owns its tensor despite later reset/step.
                torch.testing.assert_close(one.info['applied_physical_action'][:,0],torch.full((2,),.15,device='cuda'))
                if graph:self.assertIsNotNone(env.scene.control_graph)


if __name__=='__main__':unittest.main()
