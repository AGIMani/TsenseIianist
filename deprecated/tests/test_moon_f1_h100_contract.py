"""The 144-note H100 comparison contract; GPU acceptance requires Slurm.

The existing exact-timing integration remains a small regression fixture.
This test explicitly builds the score and idle-posture reward used by R0.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch

from tests.test_hpc_40hz_entry import arguments
from tsensei_v3_2_pianist.rl_torch.train import (
    effective_config, parser, resolve_replay_prefill, updates_this_step)

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / 'training_mjlab/inputs/clair_de_lune_beginner/generated_v1/clair-de-lune-beginner_fingered.proto'
SCORE_SHA256 = 'b3bae36cb7b77a82148674e0c124482c204ad1dee0be3e35ab585f94d02acb3b'
POSTURE = 'humanoid_finger_gesture_reward'


def comparison_config(*extra):
    args, _ = parser().parse_known_args(arguments('--midi', str(SCORE), '--discount', str(.8**.5),
        '--wrist-pitch-range-scale', '1', '--eval-interval', '200', '--save-interval', '1000',
        '--no-save-forward', '--vector-steps', '10000', '--seed', '42', *extra))
    cfg = effective_config(args)
    resolve_replay_prefill(args, cfg, 2637)
    return args, cfg


class MoonComparisonCPU(unittest.TestCase):
    def test_budget_and_single_variable_groups(self):
        args, base = comparison_config()
        self.assertEqual((args.num_envs, args.seed, args.vector_steps), (512, 42, 10000))
        self.assertEqual(base['requested_transitions'], 5120000)
        self.assertEqual(base['replay_prefill']['required_transitions'], 1350144)
        self.assertEqual(base['learning_start_transition_1_based'], 1350145)
        self.assertEqual(base['warmup_transitions'], 9999)
        self.assertEqual(sum(updates_this_step(args, step) for step in range(1, 10001)), 942464)
        self.assertEqual(base['reward_config']['terms'][POSTURE]['params'],
                         dict(mode='idle_posture_penalty_v1', maximum_penalty_per_step=.1, sigma=.25))
        for filename, term, field, value in (
                ('r1_self02.json', 'self_collision_reward', 'weight', .2),
                ('r2_posture002.json', POSTURE, 'maximum_penalty_per_step', .02)):
            expected = deepcopy(base)
            target = expected['reward_config']['terms'][term]
            (target if field == 'weight' else target['params'])[field] = value
            _, actual = comparison_config('--reward-config',
                str(ROOT / 'configs/experiments/moon_f1_20261001' / filename))
            self.assertEqual(actual, expected)
        for gamma in (.97, .8):
            with self.subTest(gamma=gamma):
                expected = deepcopy(base)
                expected['sac']['gamma'] = gamma
                self.assertEqual(comparison_config('--discount', str(gamma))[1], expected)

    def test_committed_score_and_all_physical_note_samples(self):
        self.assertEqual(hashlib.sha256(SCORE.read_bytes()).hexdigest(), SCORE_SHA256)
        # Parse with the installed Composer environment, as source export does.
        code = '''
import json, sys
from tsensei_v3_2_pianist.music.midi_file import MidiFile, NoteTrajectory
from tsensei_v3_2_pianist.mjlab_env.exact_score import sequence_events, validate_events, physical_coverage
midi = MidiFile.from_file(sys.argv[1]).trim_silence()
frames = len(NoteTrajectory.from_midi(midi, .025))
events, sustain = sequence_events(midi.seq, 1., .025, (frames + 40) * .025)
events, sustain = validate_events(events, sustain, (frames + 50) * .025)
counts = physical_coverage(events, length=frames + 50, control_dt=.025, physics_dt=.0025)
print(json.dumps(dict(notes=len(events), music_frames=frames, counts=counts)))
'''
        result = subprocess.check_output(
            [str(ROOT / 'training/.venv/bin/python'), '-c', code, str(SCORE)],
            cwd=ROOT, text=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})
        score = json.loads(result.strip().splitlines()[-1])
        self.assertEqual((score['notes'], score['music_frames']), (144, 2587))
        self.assertEqual(len(score['counts']), 144)
        self.assertGreater(min(score['counts']), 0)


@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required')
class MoonComparisonGPU(unittest.TestCase):
    def test_actual_moon_episode_idle_posture_and_diagnostics(self):
        if not torch.cuda.is_available():
            self.fail('Slurm acceptance requires allocated CUDA GPUs')
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.train import evaluate

        _, cfg = comparison_config()
        recipe = make_recipe(midi=str(SCORE), control_hz=40, physics_hz=400, lookahead=20,
            task_profile='tsensei_collision_v1', drive_profile='shadow_v1', initial_buffer_time=0.,
            preparation_time=1., preparation_lift=.025, preparation_mode='policy_v1',
            release_tail_time=.25, score_validation='strict_v1', exact_note_timing=True)
        self.assertEqual(recipe['external_score']['sha256'], SCORE_SHA256)
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent / 'hands_only')
        env = PianoTaskEnv(scene, task, num_envs=1, physics_graph=True,
            action_profile=cfg['action_profile'], disable_wrist_tz=cfg['disable_wrist_tz'],
            disable_wrist_yaw=cfg['disable_wrist_yaw'], preparation=recipe['preparation'],
            collision_diagnostics=True, reward_config=cfg['reward_config'],
            finger_clearance=cfg['finger_clearance'], finger_coupling=cfg['finger_coupling'],
            control_limits=cfg['control_limits'])
        self.assertEqual((env.action_size, env.observation_size), (31, 2297))
        self.assertEqual((env.preparation_steps, env.music_end_step, env.release_tail_steps), (40, 2627, 10))
        self.assertEqual(env.logic.length, 2637)
        self.assertEqual(len(env.bundle.exact_note_events), 144)
        self.assertEqual(env.reward_manager.gesture_mode, 'idle_posture_penalty_v1')
        self.assertEqual(tuple(env.reset().shape), (1, 2297))
        names = env.scene.bundle.manifest['action_names']
        for side in ('R', 'L'):
            for finger in ('Index', 'Mid', 'Ring', 'Pinky'):
                for joint, upper in (('MCP', 1.2), ('PIP', .9), ('DIP', .8)):
                    index = names.index(f'{side}_{finger}_{joint}')
                    self.assertAlmostEqual(env.scene.physical_high[index].item(), upper, places=6)
            index = names.index(f'{side}_Wrist_TY')
            self.assertAlmostEqual(env.scene.physical_low[index].item(), -.05, places=6)
            self.assertAlmostEqual(env.scene.physical_high[index].item(), .15, places=6)

        original_step = env.step
        posture_values = []

        def checked_step(action):
            transition = original_step(action)
            weighted = transition.info['reward_weighted_terms']
            actual = transition.info['reward_terms'][POSTURE]
            # The posture cap is in final per-step units, without a second dt scale.
            torch.testing.assert_close(actual, weighted[POSTURE], rtol=0, atol=0)
            expected = torch.stack([value * (.5 if name != POSTURE else 1.)
                                    for name, value in weighted.items()], dim=1).sum(1)
            torch.testing.assert_close(transition.reward, expected)
            self.assertTrue(torch.isfinite(transition.observation).all())
            posture_values.append(actual.item())
            return transition

        env.step = checked_step
        action = env.scene.policy_action(env.scene.control_reset_action).expand(1, -1)

        class ConstantAgent:
            device = env.device

            def act(self, observation, deterministic=True):
                return action

        metrics = evaluate(ConstantAgent(), env, episodes=1)
        row = metrics['episode_statistics'][0]
        self.assertEqual(metrics['completed_episodes'], 1)
        self.assertEqual(row['length'], 2637)
        self.assertEqual(metrics['note_timing_reference_count'], 144)
        self.assertTrue(metrics['release_tail_complete'])
        for flag in ('substep_key_diagnostics_valid', 'collision_diagnostics_valid',
                     'key_contact_diagnostics_valid', 'note_timing_available'):
            self.assertTrue(row[flag], flag)
        for coverage in ('substep_key_coverage', 'collision_substep_coverage',
                         'key_contact_substep_coverage'):
            self.assertEqual(row[coverage], 1., coverage)
        expected_samples = int(env.exact_score.goal_history[:, :, :88].sum().item())
        self.assertEqual(row['substep_true_positive_count'] + row['substep_false_negative_count'], expected_samples)
        self.assertEqual(row['finger_clearance_projection_frames'], 0)
        self.assertEqual(len(posture_values), 2637)
        self.assertTrue(np.isfinite(posture_values).all())
        self.assertGreaterEqual(min(posture_values), -.1000001)
        self.assertLessEqual(max(posture_values), 0.)
        self.assertLess(min(posture_values), -1e-10, 'Actual idle-posture reward must be exercised')
        report = dict(score_sha256=SCORE_SHA256, note_count=144, control_frames=2637,
            physics_samples=26370, expected_target_key_samples=expected_samples,
            posture_min=min(posture_values), posture_max=max(posture_values),
            metrics=metrics)
        print('MOON_H100_ACCEPTANCE=' + json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
