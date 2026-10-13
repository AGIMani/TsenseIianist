"""Final-step posture costs and collision baselines, evaluated on CPU."""
from copy import deepcopy
import unittest

import torch

from tests.test_paper_rewards import scene_fixture, only
from tsensei_v3_2_pianist.mjlab_env.reward_config import (
    IDLE_POSTURE_GESTURE, FOREARM_COLLISION_PENALTY, validate)
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager


class IdlePostureRewardsTest(unittest.TestCase):
    def fixture(self, count=1, dt=.025, anticipation=False):
        scene=scene_fixture();names=scene['action_names'][:-1]
        scene.update(joint_names=names,qpos=[0.]*50,control_dt=dt)
        task=dict(hand_joints=[names[:25],names[25:]],finger_tolerance=.01)
        cfg=only('humanoid_finger_gesture_reward','forearm_reward','self_collision_reward')
        cfg.update(scale_by_dt=True,reference_dt=.05,performance={})
        for term in cfg['terms'].values():term['weight']=1.
        cfg['terms']['humanoid_finger_gesture_reward']['params']=deepcopy(IDLE_POSTURE_GESTURE)
        cfg['terms']['forearm_reward']['params']=deepcopy(FOREARM_COLLISION_PENALTY)
        if anticipation:
            cfg['performance']['anticipatory_fingering_time']=.15
            cfg['terms']['fingering_reward']['enabled']=True
        snapshot=dict(right_qpos=torch.zeros(count,25),left_qpos=torch.zeros(count,25),
            palm_collision=torch.zeros(count,dtype=torch.bool),
            palm_collision_any=torch.zeros(count,dtype=torch.bool),
            self_collision_pairs=torch.zeros(count),
            anticipatory_mask=torch.zeros(count,10,dtype=torch.bool),
            anticipatory_targets=torch.zeros(count,10,3),fingertips=torch.zeros(count,10,3),
            key_targets=torch.zeros(count,88,3))
        return cfg,scene,task,snapshot,torch.zeros(count,89),torch.zeros(count,10,88)

    def evaluate(self, data, angle=.25):
        cfg,scene,task,snapshot,target,fingers=data
        manager=RewardManager(cfg,scene,task,num_envs=len(target),device='cpu')
        for side,key in enumerate(('right_qpos','left_qpos')):
            snapshot[key][:,manager.gesture_indices[side]]=angle
        values=manager({'forearm_reward':.5*(~snapshot['palm_collision']).float()},
                       torch.zeros(len(target),51),snapshot,target,fingers)
        return values,manager

    def test_posture_final_units_independent_of_dt(self):
        for dt in (.01,.025,.05):
            with self.subTest(dt=dt):
                values,manager=self.evaluate(self.fixture(dt=dt))
                torch.testing.assert_close(values['humanoid_finger_gesture_reward'],torch.tensor([-.05]))
                torch.testing.assert_close(values['humanoid_finger_gesture_reward'],
                                           manager.raw_terms['humanoid_finger_gesture_reward'])

    def test_fixed_divisor_active_exemption_and_chord_independence(self):
        data=self.fixture(3);data[5][1,:2,10]=1;data[5][2,:,10]=1
        data[4][1,:20]=1
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['humanoid_finger_gesture_reward'],torch.tensor([-.05,-.04,0.]))

    def test_upcoming_fingers_also_exempt(self):
        data=self.fixture(anticipation=True);data[3]['anticipatory_mask'][:,:3]=True
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['humanoid_finger_gesture_reward'],torch.tensor([-.035]))

    def test_origin_zero_and_bounded_large_deviation(self):
        values,_=self.evaluate(self.fixture(),angle=0.)
        self.assertEqual(values['humanoid_finger_gesture_reward'].item(),0.)
        values,_=self.evaluate(self.fixture(),angle=100.)
        value=values['humanoid_finger_gesture_reward'].item()
        self.assertGreaterEqual(value,-.1);self.assertLess(value,-.09999)

    def test_geometry_and_feasibility_do_not_change_posture(self):
        data=self.fixture(2)
        data[3]['finger_clearance_deficit_m']=torch.tensor([[0.]*10,[100.]*10])
        data[3]['finger_pose_reference_feasible']=torch.tensor([[True]*10,[False]*10])
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['humanoid_finger_gesture_reward'],torch.tensor([-.05,-.05]))

    def test_external_weight_applies_once(self):
        data=self.fixture();data[0]['terms']['humanoid_finger_gesture_reward']['weight']=.4
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['humanoid_finger_gesture_reward'],torch.tensor([-.02]))

    def test_collision_endpoint_and_self_pair_weight(self):
        data=self.fixture(3)
        data[3]['palm_collision_any'][1:]=True
        data[3]['palm_collision'][1:]=True
        data[3]['self_collision_pairs'][:]=torch.tensor([0.,1.,3.])
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['forearm_reward'],torch.tensor([0.,-.25,-.25]))
        torch.testing.assert_close(values['self_collision_reward'],torch.tensor([0.,-.5,-1.5]))
        data[3]['palm_collision'].zero_()
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['forearm_reward'],torch.zeros(3))

    def test_legacy_forearm_baseline_is_preserved(self):
        data=self.fixture(2)
        data[0]['terms']['forearm_reward']['params']={}
        data[3]['palm_collision'][1]=True
        values,_=self.evaluate(data)
        torch.testing.assert_close(values['forearm_reward'],torch.tensor([.25,0.]))

    def test_invalid_contracts_and_multiplier_rejected(self):
        for field,value in (('maximum_penalty_per_step',-.1),('maximum_penalty_per_step',True),
                            ('maximum_penalty_per_step',float('nan')),('sigma',0),('score_multiplier',10)):
            cfg=self.fixture()[0]
            cfg['terms']['humanoid_finger_gesture_reward']['params'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate(cfg)
        cfg=self.fixture()[0];cfg['terms']['humanoid_finger_gesture_reward']['weight']=1.01
        with self.assertRaises(ValueError):validate(cfg)

    def test_idle_posture_config_rejects_geometry_only_permission(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
        with self.assertRaisesRegex(ValueError,'geometry_clearance_v1'):
            effective_config(parser().parse_args(['--reward-config',
                'configs/rewards/default_20261006.json',
                '--finger-clearance-config', 'configs/finger_clearance/control_disabled_20260927.json',
                '--upcoming-press-permission-time','.05']))
