"""Shadow transition budgets, including boundaries inside a parallel vector."""
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.rl_torch.train import (
    effective_config, parser, training_action, updates_this_step, warmup_action_count,
)


class ShadowTrainingBudgetChecks(unittest.TestCase):
    def make_args(self, *extra):
        args = parser().parse_args([
            '--num-envs', '512', '--batch-size', '256', '--replay-capacity', '1000000',
            '--warmup-transitions', '4999', '--learning-start-transitions', '5000',
            '--updates-per-transition', '1', '--actor-lr', '.0003',
            '--critic-lr', '.0003', '--temp-lr', '.0003', *extra,
        ])
        effective_config(args)
        return args

    def test_shadow_boundary_and_total_budget(self):
        args = self.make_args()
        self.assertEqual(sum(warmup_action_count(args, s) for s in range(1, 12)), 4999)
        self.assertEqual(warmup_action_count(args, 10), 391)
        self.assertEqual(updates_this_step(args, 9), 0)
        self.assertEqual(updates_this_step(args, 10), 121)
        self.assertEqual(updates_this_step(args, 11), 512)
        self.assertEqual(sum(updates_this_step(args, s) for s in range(1, 9767)), 4995193)
        c = effective_config(args)
        self.assertEqual(c['replay_capacity'], 1000000)
        self.assertEqual(c['warmup_transitions'], 4999)
        self.assertEqual(c['learning_starts_vector_steps'], 10)
        self.assertEqual(c['steady_sample_ratio'], 256)
        self.assertEqual(c['sac']['critic_lr'], .0003)
        self.assertEqual(c['sac']['temp_lr'], .0003)

    def test_fractional_budget_no_roundoff_loss(self):
        args = self.make_args('--num-envs', '3', '--learning-start-transitions', '1',
                              '--updates-per-transition', '.1')
        self.assertEqual(sum(updates_this_step(args, s) for s in range(1, 10001)), 3000)

    def test_partial_boundary_actions_use_correct_observation_rows(self):
        args = self.make_args('--num-envs', '4', '--warmup-transitions', '6')
        observed = []
        class Agent:
            def act(self, obs):
                observed.append(obs.clone())
                return obs[:, :3]
        obs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        scene = SimpleNamespace(action_low=torch.tensor([-1., -2., 0.]),
                                action_high=torch.tensor([1., 2., 1.]))
        rng = torch.Generator().manual_seed(19)
        expected_rng = torch.Generator().manual_seed(19)
        expected_random = scene.action_low + torch.rand((2, 3), generator=expected_rng) * (
            scene.action_high - scene.action_low)
        action = training_action(Agent(), scene, obs, args, 2, rng)
        torch.testing.assert_close(action[:2], expected_random, rtol=0, atol=0)
        torch.testing.assert_close(action[2:], obs[2:, :3], rtol=0, atol=0)
        torch.testing.assert_close(observed[0], obs[2:], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng.get_state(), expected_rng.get_state()))

    def test_legacy_vector_schedule_is_preserved(self):
        args = parser().parse_args([])
        c = effective_config(args)
        self.assertEqual(c['replay_capacity'], 262144)
        self.assertEqual(c['steady_updates_per_transition'], .125)
        self.assertEqual(warmup_action_count(args, 300), 256)
        self.assertEqual(warmup_action_count(args, 301), 0)
        self.assertEqual(updates_this_step(args, 999), 0)
        self.assertEqual(updates_this_step(args, 1000), 32)
        self.assertEqual(updates_this_step(args, 1001), 32)
        args.learning_starts = 0
        self.assertEqual(updates_this_step(args, 1), 32)

    def test_invalid_transition_overrides(self):
        for key, value in [('--replay-capacity', '0'), ('--replay-capacity', '255'),
                           ('--warmup-transitions', '-1'), ('--learning-start-transitions', '0'),
                           ('--updates-per-transition', '0'), ('--updates-per-transition', 'nan'),
                           ('--critic-lr', '0'), ('--temp-lr', 'nan')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.make_args(key, value)


if __name__ == '__main__':
    unittest.main()
