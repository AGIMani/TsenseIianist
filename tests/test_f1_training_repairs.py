"""Coverage, current-finger priority, and honest release-tail evaluation."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env import performance_options
from tsensei_v3_2_pianist.rl_torch.train import (
    effective_config, evaluate, parser, resolve_replay_prefill, updates_this_step,
    warmup_action_count,
)


class ReplayPrefillTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--num-envs', '512', '--vector-steps', '9766',
            '--batch-size', '1024', '--replay-capacity', '1000000',
            '--warmup-transitions', '4999', '--learning-start-transitions', '5000',
            '--updates-per-transition', '.25', '--replay-prefill-episodes', '1', *extra])

    def test_full_prepared_score_is_collected_before_learning_without_catchup(self):
        args = self.args()
        config = effective_config(args)
        self.assertFalse(config['replay_prefill']['resolved'])
        resolve_replay_prefill(args, config, 183)  # Includes 20 prep and 5 release frames.
        self.assertEqual(config['learning_start_transition_1_based'], 93697)
        self.assertEqual(config['learning_starts_vector_steps'], 184)
        self.assertEqual(config['replay_prefill']['required_transitions'], 93696)
        self.assertEqual(config['requested_learning_start_transition_1_based'], 5000)
        self.assertEqual(sum(updates_this_step(args, i) for i in range(1, 184)), 0)
        self.assertEqual(updates_this_step(args, 184), 128)
        self.assertEqual(sum(updates_this_step(args, i) for i in range(1, 9767)),
                         (5_000_192 - 93_696)//4)
        self.assertEqual(sum(warmup_action_count(args, i) for i in range(1, 184)), 4999)
        self.assertEqual(warmup_action_count(args, 11), 0)
        self.assertEqual(config['requested_transitions'], 5_000_192)

    def test_explicit_later_learning_start_wins_and_full_score_size_is_not_hardcoded(self):
        args = self.args('--learning-start-transitions', '300000', '--replay-prefill-episodes', '2')
        config = effective_config(args)
        resolve_replay_prefill(args, config, 77)
        self.assertEqual(config['learning_start_transition_1_based'], 300000)
        self.assertEqual(config['replay_prefill']['required_vector_steps'], 154)
        self.assertEqual(config['replay_prefill']['required_transitions'], 78848)

    def test_disabled_retains_historical_schedule(self):
        args = self.args('--replay-prefill-episodes', '0')
        config = effective_config(args)
        resolve_replay_prefill(args, config, 183)
        self.assertEqual(config['learning_start_transition_1_based'], 5000)
        self.assertEqual(updates_this_step(args, 10), 30)
        self.assertEqual(updates_this_step(args, 11), 128)

    def test_truncated_or_unretainable_prefill_is_rejected(self):
        for extra, text in ((['--max-episode-steps', '100'], 'complete scores'),
                            (['--replay-capacity', '4096'], 'retain at least one')):
            args = self.args(*extra)
            config = effective_config(args)
            with self.assertRaisesRegex(ValueError, text):
                resolve_replay_prefill(args, config, 183)
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            effective_config(self.args('--replay-prefill-episodes', '-1'))


class ReleaseTailEvaluationTests(unittest.TestCase):
    def test_render_timeline_separates_release_from_music(self):
        path = Path(__file__).resolve().parents[1]/'preview/scripts/preparation_frames.py'
        spec = importlib.util.spec_from_file_location('f1_repair_preview_timing', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bundle = SimpleNamespace(preparation={'mode': 'policy_v1', 'duration': 1.})
        manifest = dict(release_tail_steps=5, music_end_step=178,
                        contract={'environment': {'initial_buffer_time': 1.}})
        timing = module.timeline(bundle, manifest)
        self.assertEqual(module.phase_at(19, .05, timing)[0], 'preparation')
        self.assertEqual(module.phase_at(20, .05, timing), ('music', 0))
        self.assertEqual(module.phase_at(177, .05, timing)[0], 'music')
        self.assertEqual(module.phase_at(178, .05, timing)[0], 'release_tail')
        historical = module.timeline(bundle, {'contract': manifest['contract']})
        self.assertEqual(module.phase_at(178, .05, historical)[0], 'music')

    class Agent:
        device = torch.device('cpu')
        def act(self, obs, deterministic=False):
            return torch.zeros(1, 1)

    class Env:
        num_envs = 1
        preparation_steps = 1
        release_tail_steps = 2
        music_end_step = 3
        bundle = SimpleNamespace(manifest={'control_dt': .05, 'physics_dt': .005})
        def __init__(self, truncate=False, missing=False):
            self.logic = SimpleNamespace(length=5, key_counts=torch.zeros(1, 3, dtype=torch.int64))
            self.truncate, self.missing = truncate, missing
        def reset(self):
            self.index = 0
            self.logic.key_counts.zero_()
            return torch.zeros(1, 1)
        def step(self, action):
            index = self.index
            self.index += 1
            goal, actual = torch.zeros(1, 89), torch.zeros(1, 88, dtype=torch.bool)
            counts = torch.zeros(1, 3, dtype=torch.int64)
            if index in (1, 2):
                goal[0, 40] = 1
                actual[0, 40] = index == 2
                counts[0, 0 if index == 2 else 2] = 1
            elif index == 3:
                actual[0, 40] = True
                counts[0, 1] = 1
            self.logic.key_counts += counts
            info = dict(goal_applied=goal, key_activation=actual, score_index=torch.tensor([index]),
                        episode_metrics=torch.tensor([[1., 1., .6, 1., 1., 1.]]),
                        episode_steps=torch.tensor([self.index]), episode_return=torch.tensor([1.]),
                        key_frame_micro_f1=torch.tensor([.5]))
            if not (self.missing and index == 3):
                info.update(substep_key_counts=counts*10, substep_count=torch.tensor([10]),
                            wrong_key_substeps=counts[:, 1]*10, wrong_key_key_substeps=counts[:, 1]*10)
            done = index == (3 if self.truncate else 4)
            return SimpleNamespace(info=info, observation=torch.zeros(1, 1),
                terminated=torch.tensor([done and not self.truncate]),
                truncated=torch.tensor([done and self.truncate]), done=torch.tensor([done]))

    def test_music_denominators_exclude_tail_but_full_episode_keeps_it(self):
        result = evaluate(self.Agent(), self.Env(), 2)
        self.assertAlmostEqual(result['f1'], .6)
        self.assertNotIn('music_only_f1',result) # current metrics publication contract
        self.assertEqual(result['music_frames'], 2)
        self.assertNotIn('music_only_substep_key_micro_f1',result)
        self.assertEqual(result['release_tail_frames'], 2)
        self.assertEqual(result['release_tail_wrong_key_frames_max'], 1)
        self.assertEqual(result['release_tail_wrong_key_substeps_max'], 10)
        self.assertEqual(result['release_tail_end_active_keys_max'], 0)
        self.assertTrue(result['release_tail_complete'])

    def test_incomplete_or_missing_tail_is_never_reported_as_successful_release(self):
        result = evaluate(self.Agent(), self.Env(truncate=True))
        self.assertFalse(result['release_tail_complete'])
        self.assertIsNone(result['release_tail_end_active_keys_max'])
        result = evaluate(self.Agent(), self.Env(missing=True))
        self.assertIsNone(result['release_tail_wrong_key_substeps_max'])
        self.assertFalse(result['substep_key_diagnostics_valid'])


if __name__ == '__main__':
    unittest.main()
