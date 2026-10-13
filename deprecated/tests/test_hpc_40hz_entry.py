"""Capture the current 20/200 Hz wrapper and its historical filename alias."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FINGER_NAMES, RANGE_MODE

ROOT = Path(__file__).resolve().parents[1]


def arguments(*extra, entry='train_coupled_20hz.sh'):
    with tempfile.TemporaryDirectory() as directory:
        python = Path(directory)/'python3'
        python.write_text('#!/bin/sh\nprintf "%s\\0" "$@"\n')
        python.chmod(0o755)
        result = subprocess.check_output(
            ['bash', str(ROOT/'scripts/hpc'/entry), *extra],
            env={**os.environ, 'PATH': directory+os.pathsep+os.environ['PATH']}, cwd=ROOT)
    return result.decode().rstrip('\0').split('\0')[1:]


class Hpc20HzEntryTest(unittest.TestCase):
    def test_all_training_entry_defaults_use_substep_scoring(self):
        for entry in ('run.sh', 'run_tsensei_a32.sh',
                      'scripts/hpc/train_coupled_20hz.sh', 'scripts/hpc/train_coupled_40hz.sh',
                      'scripts/train_twinkle_h12_no_tz.sh',
                      'scripts/train_twinkle_h12_no_tz_no_yaw.sh',
                      'scripts/train_twinkle_h10_ty04_yaw015_no_tz.sh',
                      'scripts/train_twinkle_h10_ty04_yaw015_no_tz_no_yaw.sh'):
            with self.subTest(entry=entry):
                args, _ = parser().parse_known_args(arguments(entry=str(ROOT/entry)))
                performance = effective_config(args)['reward_config']['performance']
                self.assertEqual(performance['reward_sampling'], 'auto')
                self.assertTrue(performance['substep_rewards'])
                self.assertTrue(performance['target_key_substep_reward'])
                self.assertFalse(performance['staged_fingering_reward'])

    def test_moon_comparison_changes_one_parameter_per_group(self):
        def resolve(*extra):
            args, _ = parser().parse_known_args(arguments('--vector-steps', '10000', '--seed', '42', *extra))
            self.assertEqual(args.vector_steps * args.num_envs, 5120000)
            self.assertEqual(args.seed, 42)
            return effective_config(args)
        base = resolve()
        for filename, term, field, value in (
                ('r1_self02.json', 'self_collision_reward', 'weight', .2),
                ('r2_posture002.json', 'humanoid_finger_gesture_reward', 'maximum_penalty_per_step', .02)):
            expected = deepcopy(base)
            target = expected['reward_config']['terms'][term]
            if field == 'weight':
                target[field] = value
            else:
                target['params'][field] = value
            self.assertEqual(resolve('--reward-config', str(ROOT/'configs/experiments/moon_f1_20261001'/filename)), expected)
        expected = deepcopy(base)
        expected['sac']['gamma'] = .97
        self.assertEqual(resolve('--discount', '.97'), expected)

    def test_time_horizons_budget_and_learning_units(self):
        argv = arguments()
        args, _ = parser().parse_known_args(argv)
        config = effective_config(args)
        source = argparse.ArgumentParser()
        source.add_argument('--midi')
        for name in ('control-hz', 'physics-hz', 'lookahead', 'preparation-time', 'release-tail-time'):
            source.add_argument('--'+name, type=float)
        time, _ = source.parse_known_args(argv)
        self.assertEqual(time.midi, 'TwinkleTwinkleRousseau')
        self.assertEqual((time.control_hz, time.physics_hz, time.lookahead), (20, 200, 10))
        self.assertEqual(time.lookahead/time.control_hz, .5)
        self.assertEqual(time.preparation_time*time.control_hz, 20)
        self.assertEqual(time.release_tail_time*time.control_hz, 5)
        self.assertEqual(config['finger_coupling']['reduce_observation'], False)
        self.assertEqual(config['finger_coupling']['mode'], RANGE_MODE)
        self.assertEqual(config['finger_coupling']['angle_ranges_deg'], {name: [60, 80] for name in FINGER_NAMES})
        reward = config['reward_config']
        self.assertEqual(reward['performance']['anticipatory_fingering_time'], .15)
        self.assertFalse(reward['performance']['anticipatory_current_priority'])
        self.assertFalse(reward['performance']['staged_fingering_reward'])
        self.assertEqual(reward['performance']['upcoming_press_permission_time'], 0.)
        self.assertTrue(reward['performance']['exact_note_timing'])
        self.assertTrue(reward['scale_by_dt'])
        self.assertEqual(reward['reference_dt'], .05)
        gesture = reward['terms']['humanoid_finger_gesture_reward']
        self.assertEqual(gesture['weight'], 1.)
        self.assertEqual(gesture['params'], dict(mode='idle_posture_penalty_v1',
            maximum_penalty_per_step=.1, sigma=.25))
        self.assertEqual(reward['terms']['forearm_reward']['params'],
                         dict(mode='collision_penalty_v1', penalty=.5))
        self.assertEqual(reward['terms']['self_collision_reward']['weight'], 1.)
        self.assertEqual(config['sac']['gamma'], .75)
        self.assertEqual(config['sac']['init_temperature'], .5)
        self.assertEqual(args.vector_steps, 19532)
        self.assertEqual(args.vector_steps*args.num_envs, 10000384)
        self.assertEqual(args.replay_capacity, 2000000)
        self.assertEqual(args.warmup_transitions, 9999)
        self.assertEqual(args.learning_start_transitions, 10000)
        self.assertEqual(args.eval_interval, 1000)
        self.assertEqual(args.save_interval, 0)
        self.assertTrue(args.save_forward)
        self.assertEqual(config['checkpoint_retention']['forward_vector_step'], 1000)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.updates_per_transition, .25)
        self.assertEqual(args.eval_num_envs, 3)
        for name in ('R_Wrist_TX', 'L_Wrist_TX', 'R_Wrist_TY', 'L_Wrist_TY'):
            self.assertEqual(config['control_limits']['channels'][name]['max_velocity'], 1.)

    def test_explicit_overrides_and_single_gpu_entry_remain_available(self):
        args, _ = parser().parse_known_args(arguments('--no-finger-coupling', '--discount', '.9',
            '--vector-steps', '512', '--learner-device', 'cuda:0', '--no-collector-pipeline'))
        config = effective_config(args)
        self.assertIsNone(config['finger_coupling'])
        self.assertEqual(config['sac']['gamma'], .9)
        self.assertEqual(args.vector_steps, 512)
        self.assertFalse(args.collector_pipeline)

    def test_historical_filename_alias_and_explicit_frequency_override(self):
        self.assertEqual(arguments(), arguments(entry='train_coupled_40hz.sh'))
        extra = ('--control-hz', '40', '--physics-hz', '400', '--lookahead', '20')
        self.assertEqual(arguments(*extra), arguments(*extra, entry='train_coupled_40hz.sh'))
        source = argparse.ArgumentParser()
        for name in ('control-hz', 'physics-hz', 'lookahead'):
            source.add_argument('--'+name, type=float)
        actual, _ = source.parse_known_args(arguments(*extra))
        self.assertEqual((actual.control_hz, actual.physics_hz, actual.lookahead), (40, 400, 20))

    def test_default_interval_and_explicit_legacy_fixed_angle_override(self):
        args, _ = parser().parse_known_args(arguments())
        coupling = effective_config(args)['finger_coupling']
        self.assertEqual(coupling, dict(schema=1, mode=RANGE_MODE, reduce_observation=False,
                                      angle_ranges_deg={name: [60, 80] for name in FINGER_NAMES}))
        args, _ = parser().parse_known_args(arguments('--finger-coupling-angle', '60'))
        coupling = effective_config(args)['finger_coupling']
        self.assertEqual(coupling['target_angle_deg'], 60)
        self.assertNotIn('angle_ranges_deg', coupling)
