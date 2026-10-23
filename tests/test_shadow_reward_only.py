"""Substep key scoring with unchanged endpoint semantics for other rewards."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env import performance_options
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
from tests.test_mjlab_task import fixture, snapshot
from tests.test_paper_rewards import scene_fixture


def config():
    result = template()
    result['performance'] = dict(anticipatory_fingering_time=.15,
        exact_note_timing=True)
    return result


def inputs():
    bundle = fixture()
    logic = MusicTaskLogic(bundle, num_envs=1, device='cpu')
    sample = snapshot(1)
    # Earlier physical samples deliberately disagree with the final state and
    # final score: wrong key down, old sustain target on, current keys absent.
    history = torch.zeros(1, 10, 88)
    history[:, :-1, 30] = 1.
    history[:, -1] = sample['key_state']
    goal_history = logic.goal[logic.index][:, None, :].expand(-1, 10, -1).clone()
    goal_history[:, :-1, -1] = 1.
    sample.update(key_state_history=history, key_activation_history=history.bool(),
        exact_goal_history=goal_history, power_mean=torch.tensor([100.]),
        palm_collision_any=torch.tensor([True]), signed_power=torch.zeros(1, 50),
        anticipatory_targets=torch.zeros(1, 10, 3),
        anticipatory_mask=torch.ones(1, 10, dtype=torch.bool))
    base, _ = logic.reward_terms(sample, logic.index)
    return bundle, logic, sample, base


def evaluate(reward, sample, base, bundle, logic):
    manager = RewardManager(reward, scene_fixture(), bundle.manifest, num_envs=1, device='cpu')
    return manager(base, torch.zeros(1, 51), sample,
        logic.goal[logic.index], logic.fingers[logic.index])


class ShadowRewardOnlyTests(unittest.TestCase):
    def test_old_sampling_options_are_rejected_even_when_disabled(self):
        for key, value in dict(substep_rewards=False, target_key_substep_reward=False,
                               target_key_activation_weight=0., reward_sampling='control_endpoint').items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Removed reward sampling'):
                performance_options.canonical({key: value})
        old = template(); old['contract'] = 'shadow_control_frame_legacy'
        with self.assertRaisesRegex(ValueError, 'physical_key_fingering_exact_activation100_v1'):
            validate(old)

    def test_cli_removes_old_options_and_preserves_original_midi_diagnostics(self):
        for args in (['--substep-rewards'], ['--no-substep-rewards'],
                     ['--target-key-substep-reward'], ['--target-key-activation-weight', '0'],
                     ['--reward-sampling', 'control_endpoint'], ['--staged-fingering-reward'],
                     ['--no-staged-fingering-reward'], ['--anticipatory-current-priority'],
                     ['--anticipatory-fingering-reward']):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                parser().parse_args(args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reward.json'
            path.write_text(json.dumps(config()))
            actual = effective_config(parser().parse_args(['--reward-config', str(path)]))['reward_config']
        self.assertTrue(actual['performance']['exact_note_timing'])
        self.assertFalse(performance_options.REMOVED_SAMPLING_OPTIONS.intersection(actual['performance']))

    def test_key_substeps_and_other_endpoint_terms(self):
        bundle, logic, sample, base = inputs()
        result = evaluate(config(), sample, base, bundle, logic)
        self.assertEqual(set(result), {'key_press_reward', 'sustain_reward',
            'energy_reward', 'fingering_reward', 'forearm_reward'})
        for name, expected in base.items():
            torch.testing.assert_close(result[name], expected, atol=0, rtol=0)
        torch.testing.assert_close(result['key_press_reward'], torch.tensor([.1+.9*(.6*10**(-3.61)+.4*__import__('math').exp(-1.25))]))
        torch.testing.assert_close(result['sustain_reward'], torch.tensor([1.]))
        torch.testing.assert_close(result['energy_reward'], torch.tensor([-.01]))
        torch.testing.assert_close(result['fingering_reward'], torch.tensor([1.]))
        torch.testing.assert_close(result['forearm_reward'], torch.tensor([.5]))
        # Diagnostic targets cannot change what this action was asked to play.
        sample['exact_goal_endpoint'] = torch.zeros(1, 89)
        sample['exact_fingers_endpoint'] = torch.zeros(1, 10, 88)
        changed, _ = logic.reward_terms(sample, logic.index)
        value = evaluate(config(), sample, changed, bundle, logic)
        torch.testing.assert_close(value['key_press_reward'], result['key_press_reward'])

    def test_future_targets_never_earn_current_reward(self):
        bundle, logic, sample, base = inputs()
        sample['exact_finger_key_history'].fill_(-1)
        base, _ = logic.reward_terms(sample, logic.index)
        result = evaluate(config(), sample, base, bundle, logic)
        self.assertEqual(result['fingering_reward'].item(), 0.)
        self.assertTrue(sample['anticipatory_mask'].all())
        for key in ('staged_fingering_reward', 'anticipatory_fingering_reward', 'anticipatory_current_priority'):
            for value in (True, False):
                with self.assertRaisesRegex(ValueError, 'Removed reward sampling'):
                    performance_options.canonical({key: value})

    def test_endpoint_false_press_does_not_read_transient_history(self):
        bundle, logic, sample, base = inputs()
        reward = config()
        reward['terms']['false_press_penalty']['enabled'] = True
        endpoint = evaluate(reward, sample, base, bundle, logic)['false_press_penalty']
        self.assertEqual(endpoint.item(), 0.)


if __name__ == '__main__': unittest.main()
