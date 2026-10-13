"""Actual shell/CLI configuration acceptance for the five 20/200Hz trials."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from tests.test_hpc_40hz_entry import arguments, ROOT
from tsensei_v3_2_pianist.mjlab_env import performance_options, source_runtime, reward_config
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser

LABELS = ('S0_shadow_endpoint', 'S1_shadow_substep', 'S2_activation05', 'S3_self02', 'S4_false01')
DIRECTORY = ROOT/'configs/experiments/twinkle_20hz_20261004'


def explicit_performance_arguments(reward):
    """Override every option after run.sh defaults, including explicit false."""
    actual = reward['performance']
    if set(actual) != set(performance_options.DEFAULTS):
        raise ValueError('Experiment must explicitly declare every performance field')
    result = []
    for name, value in actual.items():
        option = name.replace('_', '-')
        if type(value) is bool:
            result.append('--'+('' if value else 'no-')+option)
        else:
            result.extend(('--'+option, str(value)))
    return result


def resolved_trial(label, *extra):
    path = DIRECTORY/(label+'.json')
    expected = json.loads(path.read_text())
    overrides = ['--reward-config', str(path), *explicit_performance_arguments(expected)]
    argv = arguments(*overrides, *extra)
    args, _ = parser().parse_known_args(argv)
    return expected, argv, args, effective_config(args)


def resolved_source_fields(argv):
    """Run the real source CLI parser, stopping immediately before construction."""
    class Captured(Exception):
        pass
    captured = {}
    def stop(**kwargs):
        captured.update(kwargs)
        raise Captured
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT/'scripts'))
        main = runpy.run_path(str(ROOT/'scripts/train_from_source.py'))['main']
        with patch.object(source_runtime, 'make_recipe', side_effect=stop):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    main([*argv, '--build-only'])
                except Captured:
                    pass
                else:
                    raise AssertionError('Source entry did not call make_recipe')
    finally:
        sys.path[:] = old_path
    return captured


class Twinkle20HzExperimentsTest(unittest.TestCase):
    def test_all_five_real_wrappers_resolve_exact_saved_rewards(self):
        for label in LABELS:
            with self.subTest(label=label):
                expected, _, _, actual = resolved_trial(label)
                self.assertEqual(actual['reward_config'], reward_config.with_legacy_defaults(expected))
                self.assertFalse(actual['reward_config']['terms']['joint_motion_limit_penalty']['enabled'])

    def test_performance_flags_are_complete_and_do_not_use_shell_defaults(self):
        for label in LABELS:
            reward, argv, _, _ = resolved_trial(label)
            self.assertEqual(set(reward['performance']), set(performance_options.DEFAULTS))
            explicit = explicit_performance_arguments(reward)
            self.assertIn('--no-anticipatory-fingering-reward', explicit)
            self.assertIn('--exact-note-timing', explicit)
            self.assertIn('--no-staged-fingering-reward', explicit)
            self.assertEqual(argv[-len(explicit):], explicit)

    def test_no_nonreward_learning_or_control_differences(self):
        base = None
        for label in LABELS:
            _, _, _, actual = resolved_trial(label)
            actual.pop('reward_config')
            if base is None:
                base = actual
            else:
                self.assertEqual(actual, base)

    def test_sampling_is_only_s1_change(self):
        base = resolved_trial(LABELS[0])[0]
        expected = deepcopy(base)
        expected['performance'].update(reward_sampling='auto', substep_rewards=True,
                                       target_key_substep_reward=True)
        self.assertEqual(resolved_trial(LABELS[1])[0], expected)

    def test_activation_is_only_s2_change(self):
        expected = deepcopy(resolved_trial(LABELS[1])[0])
        expected['performance']['target_key_activation_weight'] = .5
        self.assertEqual(resolved_trial(LABELS[2])[0], expected)

    def test_individual_auxiliary_arms_have_no_hidden_combination(self):
        base = resolved_trial(LABELS[2])[0]
        for label, term, weight in ((LABELS[3], 'self_collision_reward', .2),
                                    (LABELS[4], 'false_press_penalty', .1)):
            expected = deepcopy(base)
            expected['terms'][term].update(enabled=True, weight=weight)
            self.assertEqual(resolved_trial(label)[0], expected)

    def test_common_twenty_hz_source_and_full_observation_contract(self):
        for label in LABELS:
            reward, argv, args, actual = resolved_trial(label)
            fields = resolved_source_fields(argv)
            self.assertEqual((fields['control_hz'], fields['physics_hz'], fields['lookahead']), (20., 200., 10))
            self.assertEqual(fields['midi'], 'TwinkleTwinkleRousseau')
            self.assertEqual(fields['preparation_mode'], 'policy_v1')
            self.assertEqual(fields['preparation_time'], 1.)
            self.assertEqual(fields['preparation_lift'], .025)
            self.assertEqual(fields['release_tail_time'], .25)
            self.assertEqual(fields['wrist_pitch_range_scale'], .8)
            self.assertTrue(fields['exact_note_timing'])
            self.assertEqual(actual['sac']['gamma'], .75)
            self.assertFalse(actual['finger_coupling']['reduce_observation'])
            self.assertEqual(reward['performance']['anticipatory_fingering_time'], .15)
            self.assertTrue(args.disable_wrist_tz and args.disable_wrist_yaw)

    def test_baseline_five_terms_and_original_forearm_are_preserved(self):
        reward = resolved_trial(LABELS[0])[0]
        self.assertEqual({name for name, term in reward['terms'].items() if term['enabled']},
            {'key_press_reward', 'sustain_reward', 'energy_reward', 'fingering_reward', 'forearm_reward'})
        self.assertEqual(reward['terms']['forearm_reward']['params'], {})
        # These candidates inherit original Shadow per-control-step units.
        # At20Hz the earlier dt/.05 normalization would also equal one, but
        # the actual declared configuration must remain unscaled here.
        self.assertFalse(reward['scale_by_dt'])


if __name__ == '__main__':
    unittest.main()
