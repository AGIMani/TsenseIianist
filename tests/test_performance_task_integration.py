"""Real GPU task: optional rewards, observable future fingers and local reset."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

import torch


@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required')
class PerformanceTaskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        root = Path(__file__).resolve().parents[1]
        cls.recipe = make_recipe(task_profile='tsensei_collision_v1',drive_profile='shadow_v1',
            initial_buffer_time=0,preparation_time=1,preparation_lift=.025,preparation_mode='policy_v1')
        scene, task, _ = build(cls.recipe)
        cls.scene, cls.task = without_arms(scene,task,scene.parent/'hands')
        from tsensei_v3_2_pianist.mjlab_env.reward_config import template
        cls.reward = template()
        cls.guard = json.loads((root/'configs/finger_clearance/control_disabled_20260927.json').read_text())

    def environment(self, performance, *, collision_diagnostics=True):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        reward = deepcopy(self.reward)
        if performance is not None: reward['performance'] = performance
        return PianoTaskEnv(self.scene,self.task,num_envs=2,physics_graph=True,
            action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=self.recipe['preparation'],collision_diagnostics=collision_diagnostics,
            reward_config=reward,finger_clearance=self.guard)

    def test_all_options_reward_alignment_observations_and_local_reset(self):
        env = self.environment(dict(anticipatory_fingering_time=.15,
            staged_fingering_reward=True))
        scene=env.scene
        self.assertEqual(env.observation_size,1231)
        obs=env.reset()
        self.assertEqual(obs.shape,(2,1231))
        self.assertFalse(obs[:,-50:].any())  # 1s preparation exceeds .15s horizon
        action=scene.policy_action(scene.control_reset_action.expand(2,-1).clone())
        # Reset worlds at different score times to exercise per-world assignment lookup.
        for _ in range(18): result=env.step(action)
        self.assertTrue(result.observation[:,-50:].any())
        self.assertEqual(result.info['substep_count'].tolist(),[10,10])
        self.assertEqual(result.info['substep_key_counts'].shape,(2,3))
        self.assertEqual(scene.reward_sampling.key_state_history.shape,(2,10,88))
        snap=env.snapshot()
        self.assertEqual(snap['key_surface_height'].shape,(2,10,88))
        self.assertTrue(torch.isfinite(result.reward).all())
        self.assertTrue((result.info['reward_raw_terms']['fingering_reward']<=1.000001).all())
        previous=result.observation.clone()
        saved={name:getattr(scene.data,name)[1].clone() for name in ('qpos','qvel','ctrl','mocap_pos')}
        history=scene.reward_sampling.key_state_history[1].clone()
        reset=env.reset([0])
        self.assertFalse(reset[0,-50:].any())
        torch.testing.assert_close(reset[1],previous[1],atol=0,rtol=0)
        torch.testing.assert_close(history,scene.reward_sampling.key_state_history[1],atol=0,rtol=0)
        for name,value in saved.items():torch.testing.assert_close(value,getattr(scene.data,name)[1],atol=0,rtol=0)
        env.step(action)
        torch.testing.assert_close(result.observation,previous,atol=0,rtol=0)

    def test_baseline_diagnostics_do_not_change_legacy_reward_or_observation_contract(self):
        env=self.environment(dict(anticipatory_fingering_time=0,
            staged_fingering_reward=False))
        historical=self.environment(None)
        self.assertEqual(env.observation_size,1181)
        self.assertIsNotNone(env.scene.reward_sampling)
        self.assertIsNone(historical.scene.reward_sampling)
        action=env.scene.policy_action(env.scene.control_reset_action.expand(2,-1).clone())
        for _ in range(3):
            one,two=env.step(action),historical.step(action)
            torch.testing.assert_close(one.reward,two.reward,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(one.observation,two.observation,atol=1e-6,rtol=1e-6)
            self.assertIn('substep_count',one.info)
            self.assertNotIn('substep_count',two.info)

    def test_endpoint_key_reward_matches_independent_shadow_formula(self):
        env=self.environment(dict(anticipatory_fingering_time=0.), collision_diagnostics=True)
        self.assertIsNotNone(env.scene.reward_sampling)
        action=env.scene.policy_action(env.scene.control_reset_action.expand(2,-1).clone())
        for _ in range(24):
            index=env.logic.index.clone()
            result=env.step(action)
        snapshot=env.snapshot()
        import numpy as np
        on=env.logic.goal[index,:88].cpu().numpy().astype(bool)
        self.assertTrue(on.any())
        travel=snapshot['key_state'].cpu().numpy()
        tol=env.bundle.manifest['key_tolerance']
        error=1-travel
        distance=np.maximum(-error,error-tol).clip(min=0)/(tol*10)
        continuous=np.exp(np.log(.1)*distance**2)
        per_key=continuous*on
        correct=per_key.sum(-1)/on.sum(-1).clip(min=1)
        wrong=(snapshot['key_activation'].cpu().numpy() & ~on).any(-1)
        expected=.5*correct+.5*(~wrong)
        np.testing.assert_allclose(result.info['reward_raw_terms']['key_press_reward'].cpu().numpy(),
                                   expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(snapshot['key_activation_history'][:,-1],
                                   snapshot['key_activation'],atol=0,rtol=0)
        before=env.scene.reward_sampling.key_state_history[1].clone()
        env.reset([0])
        torch.testing.assert_close(env.scene.reward_sampling.key_state_history[1],before,atol=0,rtol=0)
        self.assertEqual(env.scene.reward_sampling.key_state_history[0].count_nonzero().item(),0)


if __name__=='__main__':unittest.main()
