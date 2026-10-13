"""Task-gated pose shaping cannot reward silence or substitute for pressing notes."""
from copy import deepcopy
import math
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.reward_config import (
    TASK_CONDITIONED_GESTURE, template, validate,
)
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, gesture_protection_mask
from tests.test_paper_rewards import scene_fixture


class ConditionedGestureTest(unittest.TestCase):
    def fixture(self, *, legacy=False, key_reward=False, count=1):
        scene=scene_fixture()
        names=scene['action_names'][:-1]
        scene.update(joint_names=names,qpos=[0.]*50)
        task={'hand_joints':[names[:25],names[25:]],
              'contract':{'environment':{'n_steps_lookahead':10}}}
        config=template()
        for name,term in config['terms'].items():
            term['enabled']=name=='humanoid_finger_gesture_reward' or (key_reward and name=='key_press_reward')
        if not legacy:
            config['terms']['humanoid_finger_gesture_reward']['params']=deepcopy(TASK_CONDITIONED_GESTURE)
        manager=RewardManager(config,scene,task,num_envs=count,device='cpu')
        snapshot={'right_qpos':torch.zeros(count,25),'left_qpos':torch.zeros(count,25),
                  'key_activation':torch.zeros(count,88,dtype=torch.bool),
                  'gesture_protected_fingers':torch.zeros(count,10,dtype=torch.bool)}
        target=torch.zeros(count,89);target[:,10]=1
        fingers=torch.zeros(count,10,88);fingers[:,1,10]=1
        snapshot['key_activation'][:,10]=True
        return manager,snapshot,target,fingers,config,scene,task

    def reward(self, data, base=None):
        manager,snapshot,target,fingers,*_=data
        return manager({} if base is None else base,torch.zeros(len(target),51),snapshot,target,fingers)

    def test_prepare_and_rest_have_no_positive_gesture_reward(self):
        data=self.fixture(count=2)
        data[2].zero_();data[3].zero_();data[1]['key_activation'].zero_()
        # The second silent world mispresses a key; neither can farm pose points.
        data[1]['key_activation'][1,20]=True
        torch.testing.assert_close(self.reward(data)['humanoid_finger_gesture_reward'],torch.zeros(2))

    def test_missing_or_wrong_notes_disable_pose_reward(self):
        data=self.fixture(count=3)
        data[1]['key_activation'][1,10]=False
        data[1]['key_activation'][2,20]=True
        torch.testing.assert_close(self.reward(data)['humanoid_finger_gesture_reward'],
                                   torch.tensor([.045,0.,0.]))
        # A partially played chord is also a failure, not a scaled pose bonus.
        data[2][:,20]=1
        torch.testing.assert_close(self.reward(data)['humanoid_finger_gesture_reward'],
                                   torch.tensor([0.,0.,.045]))

    def test_current_and_upcoming_fingers_are_exempt(self):
        data=self.fixture()
        data[1]['gesture_protected_fingers'][:,2]=True
        expected=self.reward(data)['humanoid_finger_gesture_reward'].clone()
        manager=data[0]
        # Large motion by the current index or next middle finger is unpenalized.
        data[1]['right_qpos'][:,manager.gesture_indices[0][1]]=1.2
        data[1]['right_qpos'][:,manager.gesture_indices[0][2]]=-1.2
        torch.testing.assert_close(self.reward(data)['humanoid_finger_gesture_reward'],expected)
        self.assertAlmostEqual(expected.item(),.04,places=7)
        # An unrelated idle finger still gives a small, smooth posture signal.
        data[1]['left_qpos'][:,manager.gesture_indices[1][4]]=.5
        self.assertAlmostEqual(self.reward(data)['humanoid_finger_gesture_reward'].item(),
                               .035+.005*math.exp(-2),places=7)

    def test_lookahead_boundary_zero_padding_and_staggered_worlds(self):
        fingers=torch.zeros(7,10,88)
        fingers[0,0,1]=1;fingers[2,2,2]=1;fingers[3,3,3]=1;fingers[6,9,4]=1
        protected=gesture_protection_mask(fingers,2)
        self.assertEqual(protected[0].nonzero().flatten().tolist(),[0,2])
        self.assertEqual(protected[1].nonzero().flatten().tolist(),[2,3])
        self.assertEqual(protected[6].nonzero().flatten().tolist(),[9])
        self.assertEqual(protected[torch.tensor([0,6])].sum(-1).tolist(),[2,1])
        torch.testing.assert_close(gesture_protection_mask(fingers,0),fingers.bool().any(-1))

    def test_explicit_small_budget_and_invalid_horizon(self):
        data=self.fixture()
        config=data[4]
        for field,value in (('maximum_total',.11),('maximum_total',float('nan')),
                            ('lookahead_seconds',-.01),('sigma',0),('mode','typo')):
            invalid=deepcopy(config)
            invalid['terms']['humanoid_finger_gesture_reward']['params'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate(invalid)
        invalid=deepcopy(config);invalid['terms']['humanoid_finger_gesture_reward']['weight']=1.01
        with self.assertRaises(ValueError):validate(invalid)
        config['terms']['humanoid_finger_gesture_reward']['params']['lookahead_seconds']=.55
        with self.assertRaisesRegex(ValueError,'observation lookahead'):
            RewardManager(config,data[5],data[6],num_envs=1,device='cpu')
        manager=data[0]
        self.assertEqual(manager.gesture_lookahead_steps,5)
        value=self.reward(data)['humanoid_finger_gesture_reward']
        self.assertLessEqual(value.item(),.05)
        torch.testing.assert_close(value,manager.raw_terms['humanoid_finger_gesture_reward'])
        torch.testing.assert_close(value,manager.weighted_terms['humanoid_finger_gesture_reward'])

    def test_legacy_contract_and_silence_formula_remain_unchanged(self):
        config=template()
        self.assertEqual(config['terms']['humanoid_finger_gesture_reward']['params'],
                         {'maximum_per_finger':.2,'sigma':.25})
        self.assertEqual(validate(config),config)
        data=self.fixture(legacy=True)
        data[2].zero_();data[3].zero_()
        self.assertAlmostEqual(self.reward(data)['humanoid_finger_gesture_reward'].item(),2.,places=6)

    def test_counterfactual_static_miss_cannot_win_via_new_pose_bonus(self):
        # Analytic counterexample: all other reward terms equal. This is not a
        # claim that these fixtures constitute a physically realized trajectory.
        totals=[]
        for legacy in (True,False):
            data=self.fixture(legacy=legacy,key_reward=True,count=2)
            data[1]['key_activation'][0,10]=False  # stationary missed note
            for side,key in enumerate(('right_qpos','left_qpos')):
                data[1][key][1,data[0].gesture_indices[side]]=.5
            terms=self.reward(data,{'key_press_reward':torch.tensor([.5,1.])})
            totals.append(terms['key_press_reward']+terms['humanoid_finger_gesture_reward'])
        self.assertGreater(totals[0][0].item(),totals[0][1].item())
        self.assertLess(totals[1][0].item(),totals[1][1].item())
        self.assertAlmostEqual(totals[0][0].item(),2.3,places=6)
        self.assertAlmostEqual(totals[1][0].item(),.5,places=6)


if __name__=='__main__':
    unittest.main()
