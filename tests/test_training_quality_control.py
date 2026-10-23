"""Best weights, conservative stopping and explicit fresh-state finetuning."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.replay import Transition
from tsensei_v3_2_pianist.rl_torch.train import effective_config, evaluate, parser, updates_this_step

torch.set_num_threads(1)


def selection_config(**changes):
    config = dict(save_best=True, best_metric='music_only_f1_min', target_music_f1=.99,
                  target_micro_f1=.99, consecutive_evaluations=3,
                  max_preparation_wrong_frames=0., max_hand_collision_frames=0.)
    return config | changes


def evaluation(**changes):
    return dict(evaluation_episodes=3, completed_episodes=3, truncated_episodes=0,
                music_only_f1=.999, music_only_f1_min=.995,
                music_only_micro_f1=.999, music_only_micro_f1_min=.992,
                preparation_wrong_key_frames_max=0., hand_collision_frames_max=0.,
                collision_diagnostics_coverage=1., collision_substep_coverage=1.,
                collision_physics_sample_hz=200., collision_diagnostics_valid=True) | changes


def exact_evaluation(counts=((995, 0, 5), (995, 0, 5), (980, 0, 20))):
    keys = ['exact_original_music_' + name + '_count' for name in
            ('true_positive', 'false_positive', 'false_negative')]
    rows = []
    for tp, fp, fn in counts:
        rows.append(dict(zip(keys, (tp, fp, fn))) | dict(
            completed=1, truncated=0, note_timing_available=True,
            note_timing_interval='start_inclusive_end_exclusive', note_timing_sample_time='post_step',
            note_timing_music_start_s=1., note_timing_music_end_s=8.,
            substep_key_diagnostics_coverage=1., substep_key_coverage=1.,
            substep_key_diagnostics_valid=True, exact_original_music_micro_f1=2*tp/(2*tp+fp+fn)))
    tp, fp, fn = (sum(row[key] for row in rows) for key in keys)
    return evaluation() | dict(zip(keys, (tp, fp, fn))) | dict(
        episode_statistics=rows, note_timing_available=True, note_timing_episodes=3,
        substep_key_sample_hz=400., substep_key_diagnostics_coverage=1., substep_key_coverage=1.,
        substep_key_diagnostics_valid=True, exact_original_music_micro_f1=2*tp/(2*tp+fp+fn),
        key_underside_substeps_max=0, key_contact_diagnostics_coverage=1.,
        key_contact_substep_coverage=1., key_contact_diagnostics_valid=True,
        key_contact_physics_sample_hz=400.)


@unittest.skip('Auxiliary music-only/exact-clock selection contract was removed')
class ExactQualitySelectionChecks(unittest.TestCase):
    def config(self, **changes):
        return selection_config(best_metric=checkpoint.EXACT_MUSIC_F1_MIN,
            target_exact_micro_f1=.99, max_key_underside_substeps=0) | changes

    def test_exact_worst_episode_not_pool_controls_best_and_additional_stop_gate(self):
        tracker = checkpoint.EvaluationTracker(self.config())
        value = exact_evaluation()
        self.assertGreater(value['exact_original_music_micro_f1'], .99)
        result = tracker.observe(value, vector_step=1, updates=1)
        self.assertTrue(result['improved'])
        self.assertFalse(result['target_passed'])
        self.assertAlmostEqual(tracker.best['score'], 1960/1980)
        self.assertEqual(tracker.best['metric'], checkpoint.EXACT_MUSIC_F1_MIN)
        better = exact_evaluation(((995, 0, 5),) * 3)
        # The original control-rate gates remain independently required.
        self.assertFalse(tracker.observe(better | dict(music_only_f1_min=.98),
            vector_step=2, updates=2)['target_passed'])
        for step in (3, 4, 5):
            result = tracker.observe(better, vector_step=step, updates=step)
        self.assertTrue(result['target_reached'])

    def test_exact_selection_fails_closed_for_missing_partial_or_inconsistent_records(self):
        changes = [lambda e: e.pop('episode_statistics'),
            lambda e: e['episode_statistics'].pop(),
            lambda e: e.update(note_timing_episodes=2),
            lambda e: e.update(substep_key_sample_hz=None),
            lambda e: e.update(substep_key_coverage=.9),
            lambda e: e.update(substep_key_diagnostics_valid=False),
            lambda e: e.update(completed_episodes=2),
            lambda e: e.update(key_underside_substeps_max=1),
            lambda e: e.update(key_contact_substep_coverage=.9),
            lambda e: e.update(exact_original_music_false_negative_count=1),
            lambda e: e.update(exact_original_music_micro_f1=.999),
            lambda e: e['episode_statistics'][1].pop('exact_original_music_micro_f1'),
            lambda e: e['episode_statistics'][1].update(exact_original_music_micro_f1=float('nan')),
            lambda e: e['episode_statistics'][1].update(exact_original_music_micro_f1=True),
            lambda e: e['episode_statistics'][1].update(exact_original_music_false_positive_count=-1),
            lambda e: e['episode_statistics'][1].update(substep_key_coverage=.99),
            lambda e: e['episode_statistics'][1].update(note_timing_available=False),
            lambda e: e['episode_statistics'][1].update(note_timing_sample_time='pre_step'),
            lambda e: e['episode_statistics'][1].update(truncated=1),
            lambda e: e['episode_statistics'][1].update(note_timing_music_end_s=None)]
        for i, change in enumerate(changes):
            with self.subTest(i=i):
                value = exact_evaluation(((995, 0, 5),) * 3)
                change(value)
                tracker = checkpoint.EvaluationTracker(self.config())
                result = tracker.observe(value, vector_step=1, updates=1)
                self.assertFalse(result['best_eligible'])
                self.assertFalse(result['target_passed'])
                self.assertIsNone(tracker.best)

    def test_legacy_records_and_exact_best_without_stopping(self):
        legacy = checkpoint.EvaluationTracker(selection_config())
        self.assertTrue(legacy.observe(evaluation(), vector_step=1, updates=1)['improved'])
        tracker = checkpoint.EvaluationTracker(self.config(target_music_f1=None, target_exact_micro_f1=None))
        result = tracker.observe(exact_evaluation(), vector_step=1, updates=1)
        self.assertTrue(result['improved'])
        self.assertFalse(result['target_enabled'])

    def test_cli_defaults_and_exact_sampling_requirements(self):
        default = effective_config(parser().parse_args([]))['evaluation_selection']
        self.assertEqual(default['best_metric'], 'music_only_f1_min')
        self.assertNotIn('target_exact_micro_f1', default)
        flags = ['--best-f1-metric', checkpoint.EXACT_MUSIC_F1_MIN]
        with self.assertRaisesRegex(ValueError, 'physical key sampling'):
            effective_config(parser().parse_args(flags))
        config = effective_config(parser().parse_args(flags + ['--substep-rewards']))
        self.assertEqual(config['evaluation_selection']['best_metric'], checkpoint.EXACT_MUSIC_F1_MIN)
        for value in ('nan', '1.1', '-.1'):
            with self.assertRaises(ValueError):
                effective_config(parser().parse_args(flags + ['--substep-rewards', '--collision-diagnostics',
                    '--target-music-f1', '.99', '--target-exact-micro-f1', value]))
        config = effective_config(parser().parse_args(flags + ['--substep-rewards', '--collision-diagnostics',
            '--target-music-f1', '.99', '--target-exact-micro-f1', '.99']))
        self.assertEqual(config['evaluation_selection']['target_exact_micro_f1'], .99)


@unittest.skip('Auxiliary music-only stopping and worst-episode contract was removed')
class QualitySelectionChecks(unittest.TestCase):
    def test_release_tail_blocks_target_completion_but_keeps_music_best(self):
        valid = evaluation(release_tail_steps=5, release_tail_complete=True,
                           release_tail_end_active_keys_max=0,
                           release_tail_wrong_key_frames_max=2)
        # Initial release inertia may register pressed keys; only the final state is gated.
        for change in ({'release_tail_complete': False}, {'release_tail_complete': None},
                       {'release_tail_end_active_keys_max': 1},
                       {'release_tail_end_active_keys_max': None},
                       {'release_tail_end_active_keys_max': True},
                       {'release_tail_end_active_keys_max': float('nan')},
                       {'release_tail_steps': -1}, {'release_tail_steps': .5},
                       {'release_tail_steps': None}):
            with self.subTest(change=change):
                tracker = checkpoint.EvaluationTracker(selection_config())
                result = tracker.observe(valid | change, vector_step=1, updates=1)
                self.assertTrue(result['improved'])
                self.assertTrue(result['best_eligible'])
                self.assertFalse(result['target_passed'])
                self.assertEqual(result['consecutive'], 0)
        tracker = checkpoint.EvaluationTracker(selection_config())
        for step in (1, 2, 3):
            result = tracker.observe(valid, vector_step=step, updates=step)
        self.assertTrue(result['target_reached'])

    def test_legacy_and_disabled_release_tail_keep_existing_target_gate(self):
        for changes in ({}, {'release_tail_steps': 0, 'release_tail_complete': None}):
            with self.subTest(changes=changes):
                tracker = checkpoint.EvaluationTracker(selection_config())
                for step in (1, 2, 3):
                    result = tracker.observe(evaluation(**changes), vector_step=step, updates=step)
                self.assertTrue(result['target_reached'])

    def test_initial_evaluation_keeps_best_but_not_target_streak(self):
        tracker = checkpoint.EvaluationTracker(selection_config())
        baseline = tracker.observe(evaluation(), vector_step=0, updates=0, count_for_target=False)
        self.assertTrue(baseline['improved'])
        self.assertEqual(tracker.consecutive, 0)
        self.assertFalse(baseline['target_reached'])
        for step in (20, 40, 60):
            result = tracker.observe(evaluation(), vector_step=step, updates=step)
        self.assertTrue(result['target_reached'])
        self.assertEqual(tracker.best['vector_step'], 0)  # ties do not rewrite weights

    def test_worst_episode_and_every_coverage_gate_fail_closed(self):
        for changes in (
            dict(music_only_f1_min=.98), dict(music_only_micro_f1_min=.98),
            dict(preparation_wrong_key_frames_max=1), dict(hand_collision_frames_max=1),
            dict(hand_collision_frames_max=None), dict(collision_diagnostics_coverage=.99),
            dict(collision_substep_coverage=.9), dict(collision_diagnostics_valid=False),
            dict(collision_physics_sample_hz=None), dict(completed_episodes=2),
            dict(truncated_episodes=1), dict(music_only_f1_min=float('nan')),
        ):
            with self.subTest(changes=changes):
                tracker = checkpoint.EvaluationTracker(selection_config())
                tracker.observe(evaluation(), vector_step=20, updates=20)
                tracker.observe(evaluation(), vector_step=40, updates=40)
                decision = tracker.observe(evaluation(**changes), vector_step=60, updates=60)
                self.assertFalse(decision['target_reached'])
                self.assertEqual(decision['consecutive'], 0)
                self.assertTrue(decision['target_failed_checks'])

    def test_best_uses_music_worst_case_and_ignores_incomplete_peak(self):
        tracker = checkpoint.EvaluationTracker(selection_config(target_music_f1=None))
        tracker.observe(evaluation(f1=1., music_only_f1_min=.8), vector_step=20, updates=1)
        self.assertTrue(tracker.observe(evaluation(f1=.9, music_only_f1_min=.85),
                                        vector_step=40, updates=2)['improved'])
        self.assertFalse(tracker.observe(evaluation(music_only_f1_min=1., truncated_episodes=1),
                                         vector_step=60, updates=3)['improved'])
        self.assertEqual(tracker.best['vector_step'], 40)
        self.assertFalse(tracker.last['target_reached'])

    def test_cli_defaults_target_validation_and_authorized_batch_budget(self):
        default = effective_config(parser().parse_args([]))
        self.assertTrue(default['evaluation_selection']['save_best'])
        self.assertIsNone(default['evaluation_selection']['target_music_f1'])
        self.assertFalse(default['evaluation_selection']['eval_at_start'])
        for args in (['--target-music-f1', '.99'], ['--target-micro-f1', '.99'],
                     ['--target-consecutive-evals', '1'], ['--target-music-f1', 'nan'],
                     ['--target-max-hand-collision-frames', '-1'],
                     ['--target-music-f1', '.99', '--collision-diagnostics', '--eval-interval', '0']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                effective_config(parser().parse_args(args))
        args = parser().parse_args(['--target-music-f1', '.99', '--collision-diagnostics',
                                   '--num-envs', '512', '--batch-size', '1024',
                                   '--updates-per-transition', '.25', '--learning-start-transitions', '91137'])
        config = effective_config(args)
        self.assertEqual(config['evaluation_selection']['target_micro_f1'], .99)
        self.assertEqual(config['steady_sample_ratio'], 256)
        self.assertEqual(sum(updates_this_step(args, i) for i in range(1, 9767)), 1227264)


    def test_optional_key_underside_gate_fails_closed_without_changing_legacy_gate(self):
        valid=evaluation(key_underside_substeps_max=0,key_contact_diagnostics_coverage=1.,
            key_contact_substep_coverage=1.,key_contact_diagnostics_valid=True,key_contact_physics_sample_hz=200.)
        for change in ({'key_underside_substeps_max':1},{'key_underside_substeps_max':None},
                       {'key_contact_diagnostics_coverage':.9},{'key_contact_substep_coverage':.9},
                       {'key_contact_diagnostics_valid':False},{'key_contact_physics_sample_hz':None}):
            tracker=checkpoint.EvaluationTracker(selection_config(max_key_underside_substeps=0))
            for step in (1,2):tracker.observe(valid,vector_step=step,updates=step)
            result=tracker.observe(valid|change,vector_step=3,updates=3)
            self.assertFalse(result['target_reached']);self.assertTrue(result['target_failed_checks'])
        tracker=checkpoint.EvaluationTracker(selection_config(max_key_underside_substeps=0))
        for step in (1,2,3):result=tracker.observe(valid,vector_step=step,updates=step)
        self.assertTrue(result['target_reached'])
        legacy=checkpoint.EvaluationTracker(selection_config())
        for step in (1,2,3):result=legacy.observe(evaluation(),vector_step=step,updates=step)
        self.assertTrue(result['target_reached'])
        self.assertIsNone(effective_config(parser().parse_args([]))['evaluation_selection']['max_key_underside_substeps'])
        for args in (['--target-max-key-underside-substeps','0'],['--target-max-key-underside-substeps','-1']):
            with self.assertRaises(ValueError):effective_config(parser().parse_args(args))
        flags=['--target-music-f1','.99','--collision-diagnostics','--target-max-key-underside-substeps','0']
        with self.assertRaisesRegex(ValueError,'finger-clearance-config'):
            effective_config(parser().parse_args(flags))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json';path.write_text('{"schema":1,"mode":"full_finger_clearance_v1"}')
            enabled=effective_config(parser().parse_args(flags+['--finger-clearance-config',str(path)]))
            self.assertEqual(enabled['evaluation_selection']['max_key_underside_substeps'],0)

    def test_underside_filter_keeps_best_when_higher_score_is_unsafe_or_unobserved(self):
        safe=evaluation(music_only_f1_min=.97,key_underside_substeps_max=0,
            key_contact_diagnostics_coverage=1.,key_contact_substep_coverage=1.,
            key_contact_diagnostics_valid=True,key_contact_physics_sample_hz=200.)
        tracker=checkpoint.EvaluationTracker(selection_config(max_key_underside_substeps=0))
        first=tracker.observe(safe,vector_step=0,updates=0,count_for_target=False)
        self.assertTrue(first['best_eligible']);self.assertTrue(first['improved'])
        self.assertEqual(tracker.best['max_key_underside_substeps'],0)
        for index,changes in enumerate(({'key_underside_substeps_max':1},
                                       {'key_underside_substeps_max':None},
                                       {'key_contact_diagnostics_valid':False},
                                       {'key_contact_substep_coverage':.999},
                                       {'key_contact_diagnostics_coverage':.99},
                                       {'key_contact_physics_sample_hz':None}),1):
            result=tracker.observe(safe|dict(music_only_f1_min=.999)|changes,vector_step=index,updates=index)
            self.assertFalse(result['best_eligible']);self.assertFalse(result['improved'])
            self.assertTrue(result['best_ineligible_reasons']);self.assertEqual(tracker.best['vector_step'],0)
        updated=tracker.observe(safe|dict(music_only_f1_min=.98),vector_step=10,updates=10)
        self.assertTrue(updated['improved']);self.assertEqual(updated['best_ineligible_reasons'],[])
        self.assertEqual(tracker.best['vector_step'],10)
        self.assertTrue(tracker.best['best_eligible'])
        empty=checkpoint.EvaluationTracker(selection_config(max_key_underside_substeps=0))
        empty.observe(evaluation(music_only_f1_min=1.),vector_step=0,updates=0)
        self.assertIsNone(empty.best)
        legacy=checkpoint.EvaluationTracker(selection_config())
        legacy.observe(safe,vector_step=0,updates=0)
        result=legacy.observe(evaluation(music_only_f1_min=.999),vector_step=1,updates=1)
        self.assertTrue(result['improved']);self.assertTrue(result['best_eligible'])


class EvaluationStatisticsChecks(unittest.TestCase):
    class Agent:
        device = torch.device('cpu')
        def act(self, obs, deterministic=False):
            return torch.zeros(1, 1)

    class Env:
        num_envs = 1
        preparation_steps = 2
        bundle = SimpleNamespace(manifest={'control_dt': .05, 'physics_dt': .005,
            'contract': {'environment': {'initial_buffer_time': .15}}})
        def __init__(self, missing=False):
            self.episode = -1
            self.missing = missing
            self.logic = SimpleNamespace(length=6, key_counts=torch.zeros((1, 3), dtype=torch.int64))
        def reset(self):
            self.index = 0
            self.episode += 1
            self.logic.key_counts.zero_()
            return torch.zeros(1, 1)
        def step(self, action):
            index = self.index
            self.index += 1
            goal, actual = torch.zeros(1, 89), torch.zeros(1, 88, dtype=torch.bool)
            if index >= 3:
                goal[0, 40] = 1
                actual[0, 40] = not (self.episode == 1 and index == 3)
                self.logic.key_counts[0, 0 if actual[0, 40] else 2] += 1
            wrong_prep = self.episode == 1 and index == 0
            if wrong_prep:
                actual[0, 41] = True
                self.logic.key_counts[0, 1] += 1
            contact = self.episode == 1 and index == 2
            info = dict(goal_applied=goal, key_activation=actual, score_index=torch.tensor([index]),
                        episode_metrics=torch.ones(1, 6), episode_return=torch.tensor([10.]),
                        episode_steps=torch.tensor([self.index]), key_frame_micro_f1=torch.tensor([1.]))
            if not (self.missing and index == 2):
                info.update(self_collision_pairs=torch.tensor([int(contact)]),
                            inter_hand_collision_pairs=torch.tensor([0]),
                            self_collision_substeps=torch.tensor([3 if contact else 0]),
                            inter_hand_collision_substeps=torch.tensor([0]),
                            hand_collision_substeps=torch.tensor([3 if contact else 0]),
                            collision_physics_substeps=torch.tensor([10]))
            return SimpleNamespace(observation=torch.zeros(1, 1), info=info,
                done=torch.tensor([index == 5]), terminated=torch.tensor([index == 5]), truncated=torch.tensor([False]))

    def test_prefix_excluded_worst_episode_and_physics_coverage(self):
        result = evaluate(self.Agent(), self.Env(), 2)
        self.assertIn('f1', result)
        for key in ('music_only_f1', 'music_only_f1_min', 'music_only_micro_f1_min',
                    'substep_key_micro_f1', 'exact_original_music_micro_f1'):
            self.assertNotIn(key, result)
        self.assertEqual(result['preparation_wrong_key_frames_max'], 1)
        self.assertEqual(result['hand_collision_frames_max'], 1)
        self.assertEqual(result['self_collision_substeps_max'], 3)
        self.assertEqual(result['collision_diagnostics_coverage'], 1)
        self.assertEqual(result['collision_substep_coverage'], 1)
        self.assertEqual(result['collision_physics_sample_hz'], 200)
        self.assertEqual(len(result['episode_statistics']), 2)

    def test_missing_one_collision_frame_is_not_zero_collisions(self):
        result = evaluate(self.Agent(), self.Env(missing=True))
        self.assertIsNone(result['hand_collision_frames_max'])
        self.assertAlmostEqual(result['collision_diagnostics_coverage'], 5/6)
        self.assertLess(result['collision_substep_coverage'], 1)

    def test_control_limiter_statistics_use_limited_channels_and_missing_is_unknown(self):
        for missing in (False, True):
            env = self.Env()
            env.scene = SimpleNamespace(control_limits={'channels': {'TX': {}, 'TY': {}}}, decimation=10)
            original_step = env.step
            def step(action):
                index = env.index
                result = original_step(action)
                if not (missing and index == 2):
                    clip, slew = torch.zeros(1, 51, dtype=torch.bool), torch.zeros(1, 51, dtype=torch.bool)
                    clip[0, 20] = index == 0
                    slew[0, 20:22] = index == 1
                    result.info.update(control_clip_mask=clip, control_slew_mask=slew)
                return result
            env.step = step
            with patch('tsensei_v3_2_pianist.rl_torch.train.MotionMetrics') as motion:
                motion.return_value.result.return_value = {}
                result = evaluate(self.Agent(), env)
            with self.subTest(missing=missing):
                self.assertEqual(result['control_limited_channel_count'], 2)
                self.assertAlmostEqual(result['control_limits_diagnostics_coverage'], 5/6 if missing else 1.)
                if missing:
                    self.assertIsNone(result['control_clipped_fraction'])
                    self.assertIsNone(result['control_slew_limited_frames_max'])
                else:
                    self.assertEqual(result['control_clipped_frames_max'], 1)
                    self.assertEqual(result['control_slew_limited_frames_max'], 1)
                    self.assertAlmostEqual(result['control_clipped_fraction'], 1/12)
                    self.assertAlmostEqual(result['control_slew_limited_fraction'], 2/12)


    def test_clearance_diagnostics_keep_collision_coverage_and_missing_unknown(self):
        for missing, nonfinite in ((False,False),(True,False),(False,True)):
            env=self.Env()
            env.scene=SimpleNamespace(finger_clearance={'mode':'fixture'},decimation=10)
            original_step=env.step
            def step(action):
                index=env.index;result=original_step(action)
                if not (missing and index==2):
                    projection=torch.zeros(1,10,dtype=torch.bool);projection[0,0]=index==0
                    fallback=torch.zeros_like(projection);fallback[0,1]=index==1
                    actual=torch.zeros(1,10);actual[0,0]=.003 if index==2 else 0
                    requested=torch.zeros_like(actual);requested[0,3]=.01 if index==0 else 0
                    applied=torch.zeros_like(actual)
                    if nonfinite and index==2:actual[0,0]=float('nan')
                    result.info.update(finger_projection_mask=projection,finger_fallback_mask=fallback,
                        finger_actual_deficit_m=actual,finger_requested_deficit_m=requested,finger_applied_deficit_m=applied)
                return result
            env.step=step
            with patch('tsensei_v3_2_pianist.rl_torch.train.MotionMetrics') as motion:
                motion.return_value.result.return_value={}
                result=evaluate(self.Agent(),env)
            self.assertEqual(result['collision_diagnostics_coverage'],1)
            self.assertEqual(result['collision_substep_coverage'],1)
            self.assertAlmostEqual(result['finger_clearance_diagnostics_coverage'],5/6 if missing else 1)
            self.assertEqual(result['finger_clearance_diagnostics_valid'],not nonfinite)
            if missing or nonfinite:
                self.assertIsNone(result['finger_clearance_actual_max_deficit_m'])
            else:
                self.assertEqual(result['finger_clearance_projection_frames'],1)
                self.assertEqual(result['finger_clearance_fallback_frames'],1)
                self.assertAlmostEqual(result['finger_clearance_projection_fraction'],1/60)
                self.assertAlmostEqual(result['finger_clearance_actual_max_deficit_m'],.003)
                self.assertAlmostEqual(result['finger_clearance_requested_max_deficit_m'],.01)
                self.assertEqual(result['finger_clearance_applied_max_deficit_m'],0)


    def test_key_contact_statistics_sample_substeps_and_missing_or_invalid_fail_closed(self):
        for missing, invalid in ((False,False),(True,False),(False,True)):
            env=self.Env();original_step=env.step
            def step(action):
                index=env.index;result=original_step(action)
                if not (missing and index==2):
                    result.info.update(key_underside_substeps=torch.tensor([3 if index==2 else 0]),
                        key_side_or_under_substeps=torch.tensor([5 if index==2 else 0]),
                        key_non_distal_substeps=torch.tensor([2 if index==1 else 0]),
                        key_contact_physics_substeps=torch.tensor([9 if invalid and index==2 else 10]))
                return result
            env.step=step;result=evaluate(self.Agent(),env)
            self.assertEqual(result['collision_substep_coverage'],1)
            self.assertEqual(result['key_contact_diagnostics_valid'],not invalid)
            self.assertAlmostEqual(result['key_contact_diagnostics_coverage'],5/6 if missing else 1)
            if missing or invalid:self.assertIsNone(result['key_underside_substeps_max'])
            else:
                self.assertEqual(result['key_underside_substeps_max'],3)
                self.assertEqual(result['key_underside_frames_max'],1)
                self.assertEqual(result['key_side_or_under_substeps_max'],5)
                self.assertEqual(result['key_non_distal_substeps_max'],2)
                self.assertEqual(result['key_contact_substep_coverage'],1)
                self.assertEqual(result['key_contact_physics_sample_hz'],200)


class FinetuneChecks(unittest.TestCase):
    def test_changed_reward_transfers_networks_only_with_fresh_alpha_rng(self):
        config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        source = SAC(7, 3, config, device='cpu', seed=1)
        batch = Transition(torch.ones(4, 7), torch.zeros(4, 3), torch.ones(4),
                           torch.ones(4), torch.ones(4, 7), torch.zeros(4, dtype=torch.bool),
                           torch.zeros(4, dtype=torch.bool))
        source.update(batch)
        metadata = dict(scene_sha256='scene', task_sha256='task', stage=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.pt'
            checkpoint.save(path, source, metadata, torch.Generator().manual_seed(12))
            destination = SAC(7, 3, replace(config, init_temperature=.0017), device='cpu', seed=3)
            action_rng, update_rng = destination.action_rng.get_state(), destination.update_rng.get_state()
            changed = dict(metadata, self_collision_weight=.2)
            with self.assertRaisesRegex(ValueError, 'reward contract'):
                checkpoint.load(path, destination, changed, weights_only=True, transfer_mode='finetune')
            previous = checkpoint.load(path, destination, changed, weights_only=True,
                                       transfer_mode='finetune', allow_reward_change=True)
            self.assertEqual(previous, metadata)
            self.assertEqual(destination.updates, 0)
            self.assertEqual(len(destination.actor_optimizer.state), 0)
            self.assertEqual(len(destination.critic_optimizer.state), 0)
            self.assertEqual(len(destination.temp_optimizer.state), 0)
            self.assertAlmostEqual(destination.temperature.item(), .0017, places=8)
            self.assertTrue(torch.equal(action_rng, destination.action_rng.get_state()))
            self.assertTrue(torch.equal(update_rng, destination.update_rng.get_state()))
            for name in ('actor', 'critic', 'target_critic'):
                for a, b in zip(getattr(source, name).parameters(), getattr(destination, name).parameters()):
                    self.assertTrue(torch.equal(a, b))
            with self.assertRaisesRegex(ValueError, 'Stage1'):
                checkpoint.load(path, destination, metadata, weights_only=True)
            with self.assertRaisesRegex(ValueError, 'task/scene'):
                checkpoint.load(path, destination, dict(metadata, task_sha256='changed'),
                                weights_only=True, transfer_mode='finetune')

    def test_architecture_failure_is_atomic_and_used_learner_is_rejected(self):
        config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        source = SAC(7, 3, config, device='cpu', seed=1)
        metadata = dict(scene_sha256='scene', task_sha256='task', stage=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.pt'
            checkpoint.save(path, source, metadata, torch.Generator())
            destination = SAC(7, 3, replace(config, critic_hidden=(9,)), device='cpu', seed=3)
            before = deepcopy(destination.actor.state_dict())
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                checkpoint.load(path, destination, metadata, weights_only=True, transfer_mode='finetune')
            for key, value in before.items():
                self.assertTrue(torch.equal(value, destination.actor.state_dict()[key]))
            destination.updates = 1
            with self.assertRaisesRegex(ValueError, 'fresh learner'):
                checkpoint.load(path, destination, metadata, weights_only=True, transfer_mode='finetune')


if __name__ == '__main__':
    unittest.main()
