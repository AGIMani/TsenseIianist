"""Batched evaluation preserves each requested episode and its failure gates."""
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.task_logic import binary_metrics
from tsensei_v3_2_pianist.rl_torch.train import evaluate, effective_config, parser
from tsensei_v3_2_pianist.rl_torch.coupling_evaluation import CouplingMetrics


class Agent:
    device = 'cpu'

    def __init__(self):
        self.calls = 0

    def act(self, observation, deterministic=False):
        assert deterministic
        self.calls += 1
        return torch.zeros((len(observation), 1))


class EvaluationWorlds:
    preparation_steps = 1
    release_tail_steps = 1
    music_end_step = 5

    def __init__(self, num_envs, *, invalid_world=None, missing_frame=None, truncate=False, asynchronous=False):
        self.num_envs = num_envs
        self.next_episode = 0
        self.reset_calls = self.step_calls = 0
        self.invalid_world, self.missing_frame = invalid_world, missing_frame
        self.truncate, self.asynchronous = truncate, asynchronous
        self.logic = SimpleNamespace(length=6, key_counts=torch.zeros((num_envs, 3), dtype=torch.int64))
        manifest = dict(control_dt=.05, physics_dt=.005,
                        contract={'environment': {'initial_buffer_time': .1}},
                        action_names=['R_Wrist_TX'])
        self.bundle = SimpleNamespace(manifest=manifest)
        self.scene = SimpleNamespace(device='cpu', decimation=10,
            control_limits={'channels': {'R_Wrist_TX': {}}}, finger_clearance={'mode': 'fixture'},
            bundle=SimpleNamespace(manifest=manifest, active_action_indices=[0], action_ids=[0]),
            model=SimpleNamespace(actuator_trntype=[0], actuator_trnid=torch.tensor([[0, 0]]), jnt_dofadr=[0]),
            data=SimpleNamespace(ctrl=torch.zeros((num_envs, 1)), qvel=torch.zeros((num_envs, 1))))

    def reset(self):
        self.reset_calls += 1
        self.index = 0
        self.ids = torch.arange(self.next_episode, self.next_episode+self.num_envs)
        self.next_episode += self.num_envs
        self.logic.key_counts.zero_()
        self.metric_sum = torch.zeros((self.num_envs, 3))
        self.scene.data.ctrl.zero_()
        return torch.zeros((self.num_envs, 1))

    def step(self, action):
        assert action.shape == (self.num_envs, 1)
        self.step_calls += 1
        index = self.index
        self.index += 1
        goal = torch.zeros((self.num_envs, 89))
        actual = torch.zeros((self.num_envs, 88), dtype=torch.bool)
        if 2 <= index < 5:
            goal[:, 40] = 1
            actual[:, 40] = (self.ids+index) % 3 != 0
        actual[:, 41] = (self.ids % 2 == 1) & (index in (0, 5))
        metrics, counts = binary_metrics(goal[:, :88], actual)
        self.metric_sum += metrics
        self.logic.key_counts += counts
        tp, fp, fn = self.logic.key_counts.unbind(-1)
        own = ((self.ids == 1) & (index == 3)).long()
        cross = ((self.ids == 2) & (index == 2)).long()
        hand = (own | cross)*3
        under = ((self.ids == 3) & (index == 4)).long()*2
        sample_count = torch.full((self.num_envs,), 10)
        info = dict(goal_applied=goal, key_activation=actual,
            score_index=torch.full((self.num_envs,), index),
            episode_metrics=torch.cat((self.metric_sum/self.index, torch.ones((self.num_envs, 3))), dim=-1),
            episode_return=(self.ids+1).float()*self.index,
            episode_steps=torch.full((self.num_envs,), self.index),
            key_frame_micro_f1=2*tp/(2*tp+fp+fn).clamp_min(1),
            self_collision_pairs=own, inter_hand_collision_pairs=cross,
            self_collision_substeps=own*3, inter_hand_collision_substeps=cross*3,
            hand_collision_substeps=hand, collision_physics_substeps=sample_count,
            key_underside_substeps=under, key_side_or_under_substeps=under,
            key_non_distal_substeps=under, key_contact_physics_substeps=sample_count,
            wrong_key_substeps=(counts[:, 1]>0).long()*10,
            wrong_key_key_substeps=counts[:, 1]*10, substep_key_counts=counts*10,
            substep_count=sample_count,
            control_clip_mask=(self.ids[:, None] == index),
            control_slew_mask=(self.ids[:, None] < index))
        projection = torch.zeros((self.num_envs, 10), dtype=torch.bool)
        projection[:, 0] = self.ids == index
        deficits = projection.float()*.003
        info.update(finger_projection_mask=projection, finger_fallback_mask=projection,
                    finger_actual_deficit_m=deficits, finger_requested_deficit_m=deficits*2,
                    finger_applied_deficit_m=deficits*.5)
        if self.invalid_world is not None and index == 3:
            invalid = self.ids == self.invalid_world
            info['collision_physics_substeps'] = torch.where(invalid, 9, sample_count)
            info['key_contact_physics_substeps'] = torch.where(invalid, 9, sample_count)
            info['substep_count'] = torch.where(invalid, 9, sample_count)
        if index == self.missing_frame:
            for name in ('self_collision_substeps', 'key_underside_substeps', 'substep_count'):
                del info[name]
        # Unused worlds have larger maxima, so a partial batch must filter them.
        self.scene.data.ctrl[:] = (self.ids[:, None]+1).float()*self.index*.01
        self.scene.data.qvel[:] = (self.ids[:, None]+1).float()*self.index
        done = torch.full((self.num_envs,), index == (3 if self.truncate else 5))
        if self.asynchronous and index == 2:
            done[0] = True
        return SimpleNamespace(observation=torch.zeros((self.num_envs, 1)), info=info,
            done=done, terminated=done & ~torch.tensor(self.truncate), truncated=done & torch.tensor(self.truncate))


class ControlFramePhaseWorlds:
    """Pre-action control frames with independently enabled original MIDI diagnostics."""
    preparation_steps = 40
    music_end_step = 355
    release_tail_steps = 10

    def __init__(self, *, exact=True, stop_after=365, truncated=False):
        self.num_envs = 2
        self.stop_after, self.truncated = stop_after, truncated
        self.exact_score = SimpleNamespace() if exact else None
        self.logic = SimpleNamespace(length=365, key_counts=torch.zeros((2, 3), dtype=torch.int64))
        self.bundle = SimpleNamespace(manifest=dict(control_dt=.025, physics_dt=.0025,
            contract={'environment': {'initial_buffer_time': 1.}}))

    def reset(self):
        self.index = 0
        self.logic.key_counts.zero_()
        self.metric_sum = torch.zeros((2, 3))
        return torch.zeros((2, 1))

    def step(self, action):
        index = self.index
        self.index += 1
        goal = torch.zeros((2, 89))
        actual = torch.zeros((2, 88), dtype=torch.bool)
        # First music frame misses a note; first release frame holds a wrong key.
        if index == 40:
            goal[:, 40] = 1.
        if index == 355:
            actual[:, 41] = True
        metrics, counts = binary_metrics(goal[:, :88], actual)
        self.logic.key_counts += counts
        self.metric_sum += metrics
        tp, fp, fn = self.logic.key_counts.unbind(-1)
        done = torch.full((2,), self.index >= self.stop_after)
        return SimpleNamespace(observation=torch.zeros((2, 1)), done=done,
            terminated=done & ~torch.tensor(self.truncated),
            truncated=done & torch.tensor(self.truncated), info=dict(
                score_index=torch.full((2,), index), goal_applied=goal, key_activation=actual,
                episode_return=torch.zeros(2), episode_steps=torch.full((2,), self.index),
                episode_metrics=torch.cat((self.metric_sum / self.index, torch.ones((2, 3))), -1),
                key_frame_micro_f1=2*tp/(2*tp+fp+fn).clamp_min(1)))


class BatchedEvaluationTests(unittest.TestCase):
    def assert_metrics_equal(self, serial, batch):
        for key, value in serial.items():
            if key not in ('evaluation_num_envs', 'evaluation_batches'):
                self.assertEqual(value, batch[key], key)

    def test_batched_metrics_match_serial_including_partial_last_batch(self):
        for episodes in (1, 3, 5):
            with self.subTest(episodes=episodes):
                serial_agent, batch_agent = Agent(), Agent()
                serial_env, batch_env = EvaluationWorlds(1), EvaluationWorlds(3)
                serial = evaluate(serial_agent, serial_env, episodes)
                batch = evaluate(batch_agent, batch_env, episodes)
                self.assert_metrics_equal(serial, batch)
                self.assertEqual(len(batch['episode_statistics']), episodes)
                self.assertEqual(batch_env.step_calls, ((episodes+2)//3)*6)
                self.assertEqual(batch_agent.calls, batch_env.step_calls)
                self.assertEqual(serial_agent.calls, episodes*6)
                self.assertEqual(batch['motion_max_wrist_tx_speed_m_s'], episodes*6)
                self.assertEqual(batch['collision_substep_coverage'], 1)
                self.assertEqual(batch['substep_key_coverage'], 1)
                self.assertEqual(batch['key_contact_substep_coverage'], 1)
                self.assertEqual(batch['music_frames'], 3)
                self.assertEqual(batch['release_tail_frames'], 1)

    def test_invalid_counts_in_one_world_fail_closed(self):
        serial = evaluate(Agent(), EvaluationWorlds(1, invalid_world=1), 3)
        batch = evaluate(Agent(), EvaluationWorlds(3, invalid_world=1), 3)
        self.assert_metrics_equal(serial, batch)
        self.assertFalse(batch['collision_diagnostics_valid'])
        self.assertFalse(batch['substep_key_diagnostics_valid'])
        self.assertFalse(batch['key_contact_diagnostics_valid'])
        self.assertIsNone(batch['hand_collision_frames_max'])
        self.assertIsNone(batch['music_only_substep_key_micro_f1_min'])
        self.assertIsNone(batch['key_underside_substeps_max'])
        self.assertTrue(batch['episode_statistics'][0]['collision_diagnostics_valid'])
        self.assertFalse(batch['episode_statistics'][1]['collision_diagnostics_valid'])

    def test_missing_frame_remains_unknown(self):
        serial = evaluate(Agent(), EvaluationWorlds(1, missing_frame=3), 3)
        batch = evaluate(Agent(), EvaluationWorlds(3, missing_frame=3), 3)
        self.assert_metrics_equal(serial, batch)
        self.assertAlmostEqual(batch['collision_diagnostics_coverage'], 5/6)
        self.assertIsNone(batch['hand_collision_substeps_max'])
        self.assertIsNone(batch['key_underside_substeps_max'])
        self.assertIsNone(batch['wrong_key_substeps_max'])

    def test_truncation_preserved_and_cannot_claim_complete_tail(self):
        serial = evaluate(Agent(), EvaluationWorlds(1, truncate=True), 3)
        batch = evaluate(Agent(), EvaluationWorlds(3, truncate=True), 3)
        self.assert_metrics_equal(serial, batch)
        self.assertEqual(batch['truncated_episodes'], 3)
        self.assertEqual(batch['completed_episodes'], 0)
        self.assertFalse(batch['release_tail_complete'])

    def test_asynchronous_termination_is_rejected_before_stepping_done_world(self):
        env = EvaluationWorlds(3, asynchronous=True)
        with self.assertRaisesRegex(ValueError, 'synchronized'):
            evaluate(Agent(), env, 3)
        self.assertEqual(env.step_calls, 3)

    def test_original_diagnostics_do_not_shift_control_frame_phase_boundaries(self):
        result = evaluate(Agent(), ControlFramePhaseWorlds(), episodes=2)
        self.assertEqual(result['preparation_frames'], 40)
        self.assertEqual(result['music_frames'], 315)
        self.assertEqual(result['release_tail_frames'], 10)
        self.assertTrue(result['release_tail_complete'])
        for row in result['episode_statistics']:
            self.assertEqual(row['music_false_negative_count'], 1)
            self.assertEqual(row['music_false_positive_count'], 0)
            self.assertAlmostEqual(row['music_only_f1'], 314/315)
            self.assertEqual(row['release_tail_wrong_key_frames'], 1)
            self.assertEqual(row['release_tail_end_active_keys'], 0)
        legacy = evaluate(Agent(), ControlFramePhaseWorlds(exact=False), episodes=2)
        self.assertEqual(legacy['preparation_frames'], 40)
        self.assertEqual(legacy['music_frames'], 315)
        self.assertEqual(legacy['release_tail_frames'], 10)
        self.assertTrue(legacy['release_tail_complete'])
        self.assertEqual(legacy['episode_statistics'][0]['music_false_negative_count'], 1)
        self.assertEqual(legacy['episode_statistics'][0]['music_false_positive_count'], 0)

    def test_exact_tail_requires_full_elapsed_tail_not_just_a_boundary_sample(self):
        for stop_after, truncated in ((355, False), (364, True)):
            with self.subTest(stop_after=stop_after, truncated=truncated):
                result = evaluate(Agent(), ControlFramePhaseWorlds(
                    stop_after=stop_after, truncated=truncated), episodes=2)
                self.assertFalse(result['release_tail_complete'])
                self.assertIsNone(result['episode_statistics'][0]['release_tail_end_active_keys'])

    def test_renderer_audio_edges_use_exact_time_even_when_phase_has_no_frame(self):
        import importlib.util
        from pathlib import Path
        import numpy as np
        path = Path(__file__).resolve().parents[1] / 'preview/scripts/render_mjlab_checkpoint.py'
        spec = importlib.util.spec_from_file_location('render_phase_test', path)
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        timing = dict(policy_preparation_seconds=.025, music_start_policy_seconds=.025,
                      release_tail_steps=1, music_end_step=2)
        events = [dict(type='note_on', time=time+.5, note=60)
                  for time in (.0225, .025, .0475, .05)]
        notes = np.array([[.025, .05, 39, 1]], dtype=float)
        phases = np.array(['music', 'release_tail'])
        targets = np.zeros((2, 88), dtype=bool)
        metrics = {phase: dict(note_on_events=0, wrong_note_on_events=0) for phase in phases}
        renderer._accumulate_note_on_phase_metrics(
            events, metrics, phases, targets, .025, .5, timing, exact_events=notes)
        self.assertEqual(metrics['preparation']['frames'], 0)
        self.assertIsNone(metrics['preparation']['mean_frame_f1'])
        self.assertEqual(metrics['preparation']['note_on_events'], 1)
        self.assertEqual(metrics['preparation']['wrong_note_on_events'], 1)
        self.assertEqual(metrics['music']['note_on_events'], 2)
        self.assertEqual(metrics['music']['wrong_note_on_events'], 0)
        self.assertEqual(metrics['release_tail']['note_on_events'], 1)
        self.assertEqual(metrics['release_tail']['wrong_note_on_events'], 1)
        legacy_phases = np.array(['preparation', 'music'])
        legacy = {phase: dict(note_on_events=0, wrong_note_on_events=0) for phase in legacy_phases}
        targets[1, 39] = True
        renderer._accumulate_note_on_phase_metrics(
            events, legacy, legacy_phases, targets, .025, .5, timing)
        self.assertEqual(legacy['preparation']['note_on_events'], 2)
        self.assertEqual(legacy['preparation']['wrong_note_on_events'], 2)
        self.assertEqual(legacy['music']['note_on_events'], 2)
        self.assertEqual(legacy['music']['wrong_note_on_events'], 0)

    def test_default_serial_and_explicit_batch_config(self):
        default = effective_config(parser().parse_args([]))['evaluation_execution']
        self.assertEqual(default['num_envs'], 1)
        args = parser().parse_args(['--eval-episodes', '3', '--eval-num-envs', '8'])
        config = effective_config(args)['evaluation_execution']
        self.assertEqual(config['num_envs'], 3)
        self.assertEqual(config['requested_num_envs'], 8)
        for value in ('0', '-1'):
            with self.assertRaisesRegex(ValueError, 'eval_num_envs must be positive'):
                effective_config(parser().parse_args(['--eval-num-envs', value]))


class CouplingDiagnosticTests(unittest.TestCase):
    def info(self):
        return dict(coupling_target_angle_deg=torch.full((2, 8), 60.),
            coupling_target_error_deg=torch.zeros((2, 8)),
            coupling_actual_angle_deg=torch.tensor([[60.]*8, [30.]*8]),
            coupling_actual_error_deg=torch.tensor([[0.]*8, [30.]*8]),
            coupling_infeasible=torch.zeros((2, 8), dtype=torch.bool),
            coupling_follower_at_limit=torch.zeros((2, 8), dtype=torch.bool),
            coupling_active=torch.tensor([[True]+[False]*7, [False]*8]))

    def test_active_angle_metrics_exclude_idle_fingers_and_allow_rest_episodes(self):
        metric = CouplingMetrics('cpu', 2, True)
        metric.sample(self.info())
        rows = [metric.result(1, world) for world in range(2)]
        result = CouplingMetrics.aggregate(rows)
        self.assertTrue(result['finger_coupling_diagnostics_valid'])
        self.assertEqual(result['finger_coupling_actual_error_deg_mean'], 15.)
        self.assertEqual(result['finger_coupling_actual_error_deg_max'], 30.)
        self.assertEqual(result['finger_coupling_active_actual_error_deg_mean'], 0.)
        self.assertEqual(result['finger_coupling_active_actual_error_deg_max'], 0.)
        self.assertEqual(result['finger_coupling_active_samples'], 1.)
        self.assertIsNone(rows[1]['finger_coupling_active_actual_error_deg_mean'])

    def test_missing_or_nonfinite_angle_data_is_unknown(self):
        for missing in (True, False):
            metric = CouplingMetrics('cpu', 2, True)
            info = self.info()
            if missing:
                del info['coupling_active']
            else:
                info['coupling_actual_error_deg'][1, 0] = float('nan')
            metric.sample(info)
            result = CouplingMetrics.aggregate([metric.result(1, world) for world in range(2)])
            self.assertFalse(result['finger_coupling_diagnostics_valid'])
            self.assertIsNone(result['finger_coupling_actual_error_deg_mean'])


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA evaluation metric graphs')
class MetricGraphEvaluationTests(unittest.TestCase):
    class CUDAAgent(Agent):
        device = 'cuda:0'

        def act(self, observation, deterministic=False):
            return super().act(observation, deterministic=deterministic).cuda()

    class CUDAWorlds:
        def __init__(self, **kwargs):
            from copy import copy
            self.base = EvaluationWorlds(3, **kwargs)
            for name in ('preparation_steps', 'release_tail_steps', 'music_end_step', 'num_envs', 'bundle'):
                setattr(self, name, getattr(self.base, name))
            self.logic = copy(self.base.logic)
            self.scene = copy(self.base.scene)
            self.scene.device = 'cuda:0'
            self.scene.data = SimpleNamespace(ctrl=self.base.scene.data.ctrl.cuda(),
                                              qvel=self.base.scene.data.qvel.cuda())

        def _sync(self):
            self.logic.key_counts = self.base.logic.key_counts.cuda()
            self.scene.data.ctrl.copy_(self.base.scene.data.ctrl)
            self.scene.data.qvel.copy_(self.base.scene.data.qvel)

        def reset(self):
            result = self.base.reset()
            self._sync()
            return result.cuda()

        def step(self, action):
            result = self.base.step(action)
            result.info = {k: v.cuda() for k, v in result.info.items()}
            for name in ('observation', 'done', 'terminated', 'truncated'):
                setattr(result, name, getattr(result, name).cuda())
            self._sync()
            return result

    def test_same_all_metrics_and_partial_batch(self):
        for options in ({}, {'invalid_world': 1}, {'truncate': True}):
            with self.subTest(options=options):
                a = evaluate(self.CUDAAgent(), self.CUDAWorlds(**options), 5)
                b = evaluate(self.CUDAAgent(), self.CUDAWorlds(**options), 5, metrics_graph=True)
                self.assertEqual(a, b)

    def test_changing_diagnostic_schema_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic fields changed'):
            evaluate(self.CUDAAgent(), self.CUDAWorlds(missing_frame=3), 3, metrics_graph=True)
