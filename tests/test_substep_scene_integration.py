"""Real GPU scene: read-only configuration, callback samples and graph parity."""
import unittest

import numpy as np
import torch

from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene


@unittest.skipUnless(torch.cuda.is_available(), 'MJLab requires CUDA')
class SubstepSceneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1',
                             initial_buffer_time=0., preparation_mode='policy_v1', preparation_time=1.)
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent / 'hand_only')
        cls.scene = PianoScene(scene, num_envs=2, physics_graph=True, action_profile='direct_joint_v1',
                              disable_wrist_tz=True, disable_wrist_yaw=True,
                              preparation=recipe['preparation'])

    def test_configuration_no_mutation_eager_samples_graph_parity_and_local_reset(self):
        s = self.scene
        s.reset()
        self.assertIsNone(s.reward_sampling)
        action = s.policy_action(s.bundle.neutral_action).expand(2, -1).clone()
        before = [x.clone() for x in (s.data.qpos[:], s.data.qvel[:], s.data.time[:], s.control_steps)]
        pair_mask = np.zeros((s.model.ngeom, s.model.ngeom), dtype=bool)
        s.configure_reward_sampling(pair_mask)
        for actual, expected in zip((s.data.qpos[:], s.data.qvel[:], s.data.time[:], s.control_steps), before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        eager = []
        powers = []
        def capture(scene):
            eager.append((scene.data.qpos[:, scene.key_qadr] / scene.key_limits[:, 1]).clamp(0, 1).clone())
            powers.append((scene.data.actuator_force[:, scene.action_ids] *
                           scene.data.actuator_velocity[:, scene.action_ids]).abs().sum(-1).clone())
        s.step(action, substep_callback=capture)
        torch.testing.assert_close(s.reward_sampling.key_state_history, torch.stack(eager, 1), atol=0, rtol=0)
        torch.testing.assert_close(s.reward_sampling.power_mean, torch.stack(powers, 1).mean(-1))
        eager_qpos = s.data.qpos[:].clone()
        eager_qvel = s.data.qvel[:].clone()
        eager_history = s.reward_sampling.key_state_history.clone()
        s.reset()
        s.step(action)
        self.assertIsNotNone(s.control_graph)
        torch.testing.assert_close(s.data.qpos[:], eager_qpos, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(s.data.qvel[:], eager_qvel, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(s.reward_sampling.key_state_history, eager_history, atol=1e-6, rtol=1e-6)
        other_qpos = s.data.qpos[1].clone()
        other_history = s.reward_sampling.key_state_history[1].clone()
        other_power = s.reward_sampling.power_mean[1].clone()
        s.reset([0])
        torch.testing.assert_close(s.data.qpos[1], other_qpos, atol=0, rtol=0)
        torch.testing.assert_close(s.reward_sampling.key_state_history[1], other_history, atol=0, rtol=0)
        torch.testing.assert_close(s.reward_sampling.power_mean[1], other_power, atol=0, rtol=0)
        self.assertEqual(s.reward_sampling.key_state_history[0].count_nonzero().item(), 0)
        s.configure_reward_sampling(None)
        self.assertIsNone(s.reward_sampling)
        self.assertIsNone(s.control_graph)


if __name__ == '__main__':
    unittest.main()
