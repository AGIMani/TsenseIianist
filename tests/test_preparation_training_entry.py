"""Preparation changes the episode; acceleration leaves the SAC budget intact."""
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.rl_torch.train import effective_config, evaluate, parser, updates_this_step


class PreparationEntryChecks(unittest.TestCase):
    def test_runtime_acceleration_keeps_sac_and_sampling_budget(self):
        base = ['--num-envs','512','--vector-steps','9766','--batch-size','256',
                '--replay-capacity','1000000','--warmup-transitions','4999',
                '--learning-start-transitions','5000','--updates-per-transition','1',
                '--actor-lr','.0003','--critic-lr','.0003','--temp-lr','.0003',
                '--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
                '--preparation-time','1','--preparation-lift','.02']
        old_args = parser().parse_args(base)
        new_args = parser().parse_args(base + ['--preparation-mode','policy_v1','--learner-graph','--optimizer-layout'])
        old, new = effective_config(old_args), effective_config(new_args)
        for key in old.keys() - {'preparation','acceleration'}:
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(new['preparation']['mode'], 'policy_v1')
        self.assertEqual(sum(updates_this_step(new_args, i) for i in range(1, 9767)), 4995193)
        with self.assertRaisesRegex(ValueError, 'fused'):
            effective_config(parser().parse_args(base + ['--learner-graph','--no-fused-optimizer']))
        with self.assertRaisesRegex(ValueError, 'learner-graph'):
            effective_config(parser().parse_args(base + ['--optimizer-layout']))

    def test_music_metric_does_not_gain_credit_from_quiet_preparation(self):
        class Agent:
            device = torch.device('cpu')
            def act(self, obs, deterministic=False):
                return torch.zeros(1, 1)

        class Env:
            preparation_steps = 2
            logic = SimpleNamespace(length=4, key_counts=torch.tensor([[1,0,1]]))
            def reset(self):
                self.index = 0
                return torch.zeros(1, 1)
            def step(self, action):
                index = self.index
                self.index += 1
                goal, actual = torch.zeros(1,89), torch.zeros(1,88,dtype=torch.bool)
                if index >= 2:
                    goal[0,40] = 1
                if index == 3:
                    actual[0,40] = True
                return SimpleNamespace(observation=torch.zeros(1,1),
                    done=torch.tensor([index==3]), terminated=torch.tensor([index==3]),
                    truncated=torch.tensor([False]), info={
                        'goal_applied': goal, 'key_activation': actual,
                        'score_index': torch.tensor([index]),
                        'episode_metrics': torch.tensor([[1.,.75,.75,1.,1.,1.]]),
                        'episode_return':torch.tensor([3.]),'episode_steps':torch.tensor([4]),
                        'key_frame_micro_f1':torch.tensor([2/3])})

        result = evaluate(Agent(), Env(), episodes=2)
        self.assertEqual(result['f1'], .75)  # original full-episode metric retained
        self.assertEqual(result['music_only_f1'], .5)
        self.assertEqual(result['preparation_frames'], 2)
        self.assertEqual(result['preparation_wrong_key_frames'], 0)
        self.assertEqual(result['music_frames'], 2)
        self.assertAlmostEqual(result['music_only_micro_f1'], 2/3)
        # A separately requested blank buffer is not part of the music metric.
        env = Env()
        env.bundle = SimpleNamespace(manifest={'control_dt': .05,
            'contract': {'environment': {'initial_buffer_time': .15}}})
        extra = evaluate(Agent(), env)
        self.assertEqual(extra['extra_buffer_frames'], 1)
        self.assertEqual(extra['music_frames'], 1)
        self.assertEqual(extra['music_only_f1'], 1)


if __name__ == '__main__':
    unittest.main()
