"""Substep key scoring with unchanged endpoint semantics for other rewards."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env import performance_options
from tsensei_v3_2_pianist.mjlab_env.anticipation import future_fingering
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
from tests.test_mjlab_task import fixture, snapshot
from tests.test_paper_rewards import scene_fixture


def config():
    result = template()
    result['performance'] = dict(anticipatory_fingering_time=.15,
        exact_note_timing=True,
        anticipatory_fingering_reward=False)
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
        with self.assertRaisesRegex(ValueError, 'physical_key_activation50_v1'):
            validate(old)

    def test_cli_removes_old_options_and_preserves_original_midi_diagnostics(self):
        for args in (['--substep-rewards'], ['--no-substep-rewards'],
                     ['--target-key-substep-reward'], ['--target-key-activation-weight', '0'],
                     ['--reward-sampling', 'control_endpoint']):
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
        torch.testing.assert_close(result['key_press_reward'], torch.tensor([.1+.9*(.3*10**(-3.61)+.4*__import__('math').exp(-1.25))]))
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

    def test_observed_future_fingers_no_longer_earn_reward_when_disabled(self):
        bundle, logic, sample, base = inputs()
        logic.fingers.zero_()
        logic.goal.zero_()
        base, _ = logic.reward_terms(sample, logic.index)
        disabled = evaluate(config(), sample, base, bundle, logic)
        enabled_config = config()
        enabled_config['performance']['anticipatory_fingering_reward'] = True
        enabled = evaluate(enabled_config, sample, base, bundle, logic)
        self.assertEqual(disabled['fingering_reward'].item(), 0.)
        self.assertEqual(enabled['fingering_reward'].item(), .5)
        self.assertTrue(sample['anticipatory_mask'].all())

    def test_future_observation_does_not_require_fingering_reward_when_disabled(self):
        reward = config()
        reward['terms']['fingering_reward']['enabled'] = False
        self.assertFalse(validate(reward)['terms']['fingering_reward']['enabled'])
        reward['performance']['anticipatory_fingering_reward'] = True
        with self.assertRaisesRegex(ValueError, 'enabled legacy fingering_reward'):
            validate(reward)

    def test_endpoint_false_press_does_not_read_transient_history(self):
        bundle, logic, sample, base = inputs()
        reward = config()
        reward['terms']['false_press_penalty']['enabled'] = True
        endpoint = evaluate(reward, sample, base, bundle, logic)['false_press_penalty']
        self.assertEqual(endpoint.item(), 0.)

    def test_runtime_observation_assembly_keeps_2297_and_future_timing_values(self):
        # Exercise the actual CPU tensor observation methods; physics is not
        # constructed and no GPU is used. Only the two reward flags differ.
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        sequence = torch.zeros(4, 10, 88)
        sequence[2:, 1, 20] = 1.
        assignments, seconds = future_fingering(sequence, .025, .15)
        env = PianoTaskEnv.__new__(PianoTaskEnv)
        env.logic = SimpleNamespace(index=torch.tensor([0]), length=4, observation_size=2070)
        env.scene = SimpleNamespace(policy_preparation=True,
            preparation_state=lambda: torch.tensor([[.75]]),
            data=SimpleNamespace(geom_xpos=torch.zeros(1, 88, 3),
                                 site_xpos=torch.zeros(1, 10, 3)))
        env.future_assignments, env.future_seconds = assignments, seconds
        env.exact_score = SimpleNamespace(observation=torch.arange(4 * 176).reshape(4, 176).float())
        env.coupling_observation_contract = None
        env._coupling_observation_ids = None
        env.key_geoms, env.tip_ids = torch.arange(88), torch.arange(10)
        env.finger_key_offsets = None
        env.key_offsets = torch.zeros(88, 3)
        env.rest_key_top_z = torch.zeros(88)
        env.performance = performance_options.get(config())
        disabled = env._with_preparation(torch.zeros(1, 2070))
        self.assertEqual(env.observation_size, 2297)
        self.assertEqual(disabled.shape, (1, 2297))
        self.assertTrue(disabled[:, 2071:2121].any())
        torch.testing.assert_close(disabled[:, -176:], env.exact_score.observation[[0]])
        env.performance['anticipatory_fingering_reward'] = True
        enabled = env._with_preparation(torch.zeros(1, 2070))
        torch.testing.assert_close(disabled, enabled, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
