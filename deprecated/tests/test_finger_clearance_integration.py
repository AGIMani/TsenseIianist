"""Real V3.2 GPU geometry graph parity, applied-action history and local reset."""
import json
from pathlib import Path
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for physical integration')
class FingerClearanceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        root=Path(__file__).resolve().parents[1]
        recipe=make_recipe(task_profile='tsensei_collision_v1',drive_profile='shadow_v1',initial_buffer_time=0.,
                           preparation_time=1.,preparation_mode='policy_v1')
        scene,task,_=build(recipe)
        scene,task=without_arms(scene,task,scene.parent/'hand_only')
        cls.scene_path,cls.task_path,cls.recipe=scene,task,recipe
        cls.env=PianoTaskEnv(scene,task,num_envs=2,physics_graph=True,action_profile='direct_joint_v1',
            disable_wrist_tz=True,disable_wrist_yaw=True,preparation=recipe['preparation'],
            finger_clearance=json.loads((root/'configs/finger_clearance/generic_20260927.json').read_text()),
            reward_config=json.loads((root/'configs/rewards/generic_finger_clearance_20260927.json').read_text()))

    def test_graph_matches_eager_with_changed_finger_roles_and_inputs(self):
        env=self.env;s=env.scene;g=s.finger_clearance_guard
        env.reset()
        request=s.control_reset_action.expand(2,-1).clone()
        request[:,g.finger_indices]=(g.low+g.high)/2
        assigned=env.logic.fingers[env.logic.index].clone()
        for active in (False,True):
            if active:
                assigned[:,0,40]=1
                request[:,g.finger_indices]+=0.01
                request[:,20]+=0.05
            with s.scope():
                expected,expected_metrics=g.filter(request,actual_qpos=s.data.qpos[:],
                    mocap_pos=s.data.mocap_pos[:],assigned_keys=assigned)
                actual,actual_metrics=s._filter_finger_targets(request,assigned)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
            for name,value in expected_metrics.items():
                torch.testing.assert_close(actual_metrics[name],value,atol=1e-6,rtol=1e-6)

    def test_applied_history_reward_and_subset_reset_are_consistent(self):
        env=self.env;s=env.scene;g=s.finger_clearance_guard
        env.reset()
        request=s.control_reset_action.expand(2,-1).clone()
        request[:,g.finger_indices]=g.high
        action=s.policy_action(request)
        result=env.step(action)
        self.assertTrue(result.info['finger_projection_mask'].any())
        torch.testing.assert_close(env.logic.last_action,result.info['applied_physical_action'])
        torch.testing.assert_close(s.data.ctrl[:,s.action_ids],
                                   result.info['applied_physical_action'][:,s.policy_ids[:-1]])
        auxiliary=result.info['reward_weighted_terms']['humanoid_finger_gesture_reward']
        self.assertTrue((auxiliary<0).all())
        self.assertTrue((auxiliary>=-.1-1e-6).all())
        saved={name:getattr(s.data,name)[1].clone() for name in ('qpos','qvel','ctrl','mocap_pos')}
        history=env.logic.history[1].clone()
        metric=s.finger_clearance_metrics['finger_requested_deficit_m'][1].clone()
        old_result=result.info['applied_physical_action'].clone()
        env.reset([0])
        for name,value in saved.items():
            torch.testing.assert_close(getattr(s.data,name)[1],value,atol=0,rtol=0)
        torch.testing.assert_close(env.logic.history[1],history,atol=0,rtol=0)
        torch.testing.assert_close(s.finger_clearance_metrics['finger_requested_deficit_m'][1],metric,atol=0,rtol=0)
        torch.testing.assert_close(env.logic.last_action[0],s.control_reset_action,atol=0,rtol=0)
        env.step(action)
        torch.testing.assert_close(result.info['applied_physical_action'],old_result,atol=0,rtol=0)

    def test_default_disabled_controls_keep_reward_and_physics_targets(self):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        root=Path(__file__).resolve().parents[1]
        env=PianoTaskEnv(self.scene_path,self.task_path,num_envs=2,physics_graph=True,
            action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=self.recipe['preparation'],collision_diagnostics=True,
            finger_clearance=json.loads((root/'configs/finger_clearance/control_disabled_20260927.json').read_text()),
            reward_config=json.loads((root/'configs/rewards/generic_finger_clearance_20260927.json').read_text()))
        s=env.scene;guard=s.finger_clearance_guard
        request=s.control_reset_action.expand(2,-1).clone()
        request[:,guard.finger_indices]=guard.high
        action=s.policy_action(request)
        for _ in range(2):
            result=env.step(action)
            torch.testing.assert_close(result.info['applied_physical_action'],
                                       result.info['requested_physical_action'],atol=0,rtol=0)
            torch.testing.assert_close(s.data.ctrl[:,s.action_ids],
                result.info['requested_physical_action'][:,s.policy_ids[:-1]],atol=0,rtol=0)
            self.assertFalse(result.info['finger_projection_mask'].any())
            self.assertFalse(result.info['finger_fallback_mask'].any())
            penalty=result.info['reward_weighted_terms']['humanoid_finger_gesture_reward']
            self.assertTrue((penalty<0).all())
            self.assertTrue((penalty>=-.100001).all())
        self.assertIsNotNone(s.key_contact_diagnostics)


if __name__=='__main__':unittest.main()
