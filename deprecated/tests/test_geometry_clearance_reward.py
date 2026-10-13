"""Geometric shaping remains bounded and informative during errors and silence."""
from copy import deepcopy
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.reward_config import GEOMETRY_CLEARANCE_GESTURE, template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, geometry_clearance_penalty
from tests.test_paper_rewards import scene_fixture


class GeometryClearanceRewardTest(unittest.TestCase):
    def fixture(self, count=1):
        scene=scene_fixture();names=scene['action_names'][:-1]
        scene.update(joint_names=names,qpos=[0.]*50)
        task={'hand_joints':[names[:25],names[25:]]}
        config=template()
        for name, term in config['terms'].items():
            term['enabled']=name in ('key_press_reward','humanoid_finger_gesture_reward')
        config['terms']['humanoid_finger_gesture_reward']['params']=deepcopy(GEOMETRY_CLEARANCE_GESTURE)
        manager=RewardManager(config,scene,task,num_envs=count,device='cpu')
        snapshot={'right_qpos':torch.zeros(count,25),'left_qpos':torch.zeros(count,25),
                  'key_activation':torch.zeros(count,88,dtype=torch.bool),
                  'finger_clearance_deficit_m':torch.zeros(count,10),
                  'finger_pose_reference_feasible':torch.ones(count,10,dtype=torch.bool)}
        target=torch.zeros(count,89);target[:,10]=1
        fingers=torch.zeros(count,10,88);fingers[:,1,10]=1
        return manager,snapshot,target,fingers,config

    def reward(self,data):
        manager,snapshot,target,fingers,*_=data
        return manager({'key_press_reward':torch.ones(len(target))},torch.zeros(len(target),51),
                       snapshot,target,fingers)['humanoid_finger_gesture_reward']

    def test_danger_signal_during_correct_wrong_missing_preparation_and_rest(self):
        data=self.fixture(5)
        data[1]['finger_clearance_deficit_m'][:,5]=.005
        data[1]['key_activation'][0,10]=True
        data[1]['key_activation'][1,20]=True
        data[2][3:].zero_();data[3][3:].zero_()  # silence, with/without a wrong press
        data[1]['key_activation'][4,20]=True
        torch.testing.assert_close(self.reward(data),torch.full((5,),-.04))

    def test_single_bad_thumb_not_diluted_and_active_finger_geometry_not_exempt(self):
        data=self.fixture(2)
        data[1]['finger_clearance_deficit_m'][0,1]=.005  # currently assigned finger
        data[1]['finger_clearance_deficit_m'][1,:]=.005
        torch.testing.assert_close(self.reward(data),torch.full((2,),-.04))

    def test_pose_only_idle_safe_reference_and_fixed_ten_divisor(self):
        data=self.fixture(3);manager=data[0]
        for side,key in enumerate(('right_qpos','left_qpos')):
            data[1][key][:,manager.gesture_indices[side]]=.25
        data[1]['finger_pose_reference_feasible'][1].zero_()
        # Nine idle fingers at MSE .0625: .5 each, /10, * .2 * .1.
        torch.testing.assert_close(self.reward(data),torch.tensor([-.009,0.,-.009]))
        data[3][2]=1  # all currently assigned, no idle pose cost
        self.assertEqual(self.reward(data)[2].item(),0.)
        self.assertNotIn('gesture_protected_fingers',data[1])  # no hidden future mask required

    def test_safe_origin_zero_reward_and_no_positive_idle_points(self):
        data=self.fixture(2);data[2][1].zero_();data[3][1].zero_()
        torch.testing.assert_close(self.reward(data),torch.zeros(2))
        data[1]['left_qpos'][:]=1.
        self.assertTrue((self.reward(data)<=0).all())

    def test_budget_scales_with_chord_and_actual_key_weight(self):
        p=deepcopy(GEOMETRY_CLEARANCE_GESTURE)
        target=torch.zeros(6,89)
        for i,n in enumerate((0,1,2,3,10,88)):target[i,:n]=1
        deficit=torch.full((6,10),1e6)
        mse=torch.full((6,10),1e6)
        idle=torch.ones(6,10,dtype=torch.bool)
        values=geometry_clearance_penalty(deficit,mse,idle,idle,target,p,1.)
        cap=torch.tensor([.1,.1,.0625,.125/3,.0125,.125/88])
        torch.testing.assert_close(values,-cap)
        values=geometry_clearance_penalty(deficit,mse,idle,idle,target,p,.1)
        torch.testing.assert_close(values,-torch.tensor([.0125,.0125,.00625,.0125/3,.00125,.0125/88]))

    def test_full_single_note_miss_cannot_be_paid_for_by_shaping_budget_alone(self):
        # Analytic per-frame comparison with all OTHER terms held equal; not a
        # guarantee for partial depth errors, future return or physical policy.
        for chord in range(1,11):
            cap=min(.1,.125/chord)
            self.assertGreater(1.-cap,1.-.5/chord)

    def test_geometric_local_sensitivity_exists_without_success_gate(self):
        p=deepcopy(GEOMETRY_CLEARANCE_GESTURE)
        deficit=torch.zeros(1,10,dtype=torch.float64);deficit[0,5]=.01;deficit.requires_grad_()
        pose=torch.zeros_like(deficit)
        mask=torch.ones(1,10,dtype=torch.bool)
        value=geometry_clearance_penalty(deficit,pose,mask,mask,torch.zeros(1,89),p,1.)
        value.sum().backward()
        self.assertAlmostEqual(deficit.grad[0,5].item(),-.1*.8*.005/(.01+.005)**2,places=6)
        self.assertEqual(deficit.grad[0,0].item(),0.)

    def test_missing_or_wrong_geometry_inputs_fail_clearly(self):
        for field in ('finger_clearance_deficit_m','finger_pose_reference_feasible'):
            data=self.fixture();del data[1][field]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):self.reward(data)
        data=self.fixture();data[1]['finger_pose_reference_feasible']=torch.ones(1,10)
        with self.assertRaisesRegex(ValueError,'boolean'):self.reward(data)
        data=self.fixture();data[1]['finger_clearance_deficit_m']=torch.zeros(1,9)
        with self.assertRaisesRegex(ValueError,'num_envs, 10'):self.reward(data)

    def test_configuration_has_explicit_bounded_budget_and_legacy_key_requirement(self):
        config=self.fixture()[4]
        for field,value in (('maximum_total',.1001),('missed_key_fraction',.2501),
                            ('posture_fraction',.251),('posture_fraction',-.1),
                            ('clearance_scale',0),('sigma',float('nan')),('maximum_total',True)):
            invalid=deepcopy(config);invalid['terms']['humanoid_finger_gesture_reward']['params'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(invalid)
        for change in ({'enabled':False},{'weight':0},{'params':{'mode':'paper','chord_bonus':False,'chord_scope':None}}):
            invalid=deepcopy(config);invalid['terms']['key_press_reward'].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'legacy key_press_reward'):validate(invalid)
        invalid=deepcopy(config);invalid['terms']['humanoid_finger_gesture_reward']['weight']=1.01
        with self.assertRaises(ValueError):validate(invalid)


if __name__=='__main__':unittest.main()
