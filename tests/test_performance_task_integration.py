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
        cls.guard = None # current convex-hull runtime has no capsule clearance guard

    def environment(self, performance, *, collision_diagnostics=True):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        reward = deepcopy(self.reward)
        if performance is not None: reward['performance'] = performance
        return PianoTaskEnv(self.scene,self.task,num_envs=2,physics_graph=True,
            action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=self.recipe['preparation'],collision_diagnostics=collision_diagnostics,
            reward_config=reward,finger_clearance=self.guard)

    def test_current_future_observations_and_local_reset(self):
        env=self.environment(dict(anticipatory_fingering_time=.15))
        obs=env.reset()
        assert obs.shape==(2,env.observation_size)
        assert env.future_assignments is env.exact_score.future
        assert env.logic.observation_fingers is env.exact_score.fingers
        assert env.logic.observation_goal is env.exact_score.goal
        action=env.scene.policy_action(env.logic.reset_action).expand(2,-1)
        for _ in range(19):result=env.step(action)
        assert result.observation[:,-226:-176].any() # 50 future fields before event countdowns
        before=result.observation[1].clone()
        history=env.scene.reward_sampling.fingertip_history[1].clone()
        reset=env.reset([0])
        torch.testing.assert_close(reset[1],before,atol=0,rtol=0)
        torch.testing.assert_close(env.scene.reward_sampling.fingertip_history[1],history)
        result=env.step(action)
        assert result.info['score_index'].tolist()==[0,19]
        assert result.info['finger_keys_substeps'].shape==(2,10,10)
        assert torch.isfinite(result.reward).all()

if __name__=='__main__':unittest.main()
