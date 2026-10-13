"""Exact-clock note diagnostics at physical sample times, without any GPU work."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tsensei_v3_2_pianist.rl_torch.note_timing_metrics import NoteTimingMetrics


def history(notes, *, dt=.0025, duration=.3, worlds=1):
    times = (np.arange(round(duration / dt)) + 1) * dt
    active = torch.zeros(worlds, len(times), 88, dtype=torch.bool)
    for start, end, key in notes:
        mask = (times >= start - 1e-10) & (times < end - 1e-10)
        active[:, mask, key] = True
    return active


class NoteTimingTests(unittest.TestCase):
    def test_exact_start_end_and_post_step_samples_have_no_frame_shift(self):
        metrics = NoteTimingMetrics(events=[[.0125, .0225, 2, 1]], physics_dt=.0025, music_start=0)
        metrics.sample(history([(.0125, .0225, 2)]))
        row = metrics.result()
        self.assertEqual(row['exact_original_music_true_positive_count'], 4)
        self.assertEqual(row['exact_original_music_micro_f1'], 1.)
        self.assertEqual(row['note_timing_onset_mean_ms'], 0.)
        self.assertEqual(row['note_timing_offset_mean_ms'], 0.)
        self.assertEqual(row['note_timing_matched_count'], 1)

    def test_nearest_unique_matching_maximizes_valid_repeat_matches(self):
        metrics = NoteTimingMetrics(events=[[0., .04, 2, 1], [.1, .14, 2, 1]],
                                    physics_dt=.01, music_start=0, onset_tolerance=.1)
        metrics.sample(history([(.09, .10, 2), (.19, .20, 2)], dt=.01))
        row = metrics.result()
        self.assertEqual(row['note_timing_matched_count'], 2)
        self.assertEqual(row['note_timing_missed_count'], 0)
        self.assertEqual(row['note_timing_extra_count'], 0)
        self.assertAlmostEqual(row['note_timing_onset_mean_ms'], 90.)
        self.assertEqual(row['note_timing_short_recall'], 1.)

    def test_preparation_tail_excluded_from_music_f1_but_false_notes_retained(self):
        metrics = NoteTimingMetrics(events=[[.1, .2, 2, 1]], physics_dt=.0025, music_start=.1)
        metrics.sample(history([(.02, .04, 6), (.1, .2, 2), (.23, .25, 7)]))
        row = metrics.result()
        self.assertEqual(row['exact_original_music_micro_f1'], 1.)
        self.assertEqual(row['note_timing_extra_count'], 2)
        self.assertIsNone(row['note_timing_short_recall'])  # Exactly 100 ms is not short.

    def test_final_held_note_has_censored_release_not_a_fabricated_note_off(self):
        metrics = NoteTimingMetrics(events=[[.1, .2, 2, 1]], physics_dt=.0025, music_start=.1)
        metrics.sample(history([(.1, .5, 2)]))
        row = metrics.result()
        self.assertEqual(row['note_timing_censored_release_count'], 1)
        self.assertIsNone(row['note_timing_offset_mean_ms'])
        self.assertIsNone(row['note_timing_records'][0]['played_end_s'])

    def test_delays_short_misses_and_wrong_notes_are_reported(self):
        metrics = NoteTimingMetrics(events=[[.05, .1, 2, 1], [.15, .19, 3, 2]],
                                    physics_dt=.0025, music_start=0)
        metrics.sample(history([(.06, .105, 2), (.15, .19, 4)]))
        row = metrics.result()
        self.assertEqual(row['note_timing_missed_count'], 1)
        self.assertEqual(row['note_timing_extra_count'], 1)
        self.assertEqual(row['note_timing_short_recall'], .5)
        self.assertAlmostEqual(row['note_timing_onset_mean_ms'], 10.)
        self.assertAlmostEqual(row['note_timing_offset_mean_ms'], 5.)
        self.assertAlmostEqual(row['note_timing_onset_abs_p95_ms'], 10.)
        self.assertGreater(row['exact_original_music_false_positive_count'], 0)
        self.assertGreater(row['exact_original_music_false_negative_count'], 0)

    def test_samples_cloned_and_multiple_worlds_materialized_once(self):
        metrics = NoteTimingMetrics(events=[[.05, .1, 2, 1]], physics_dt=.0025,
                                    music_start=0, num_envs=2)
        data = history([(.05, .1, 2)], worlds=2)
        data[1] = False
        metrics.sample(data[:, :30])
        metrics.sample(data[:, 30:])
        data.zero_()
        first = metrics.result(0)
        cache = metrics._history_cpu
        self.assertEqual(first['note_timing_matched_count'], 1)
        self.assertEqual(metrics.result(1)['note_timing_missed_count'], 1)
        self.assertIs(metrics._history_cpu, cache)
        self.assertIs(metrics.result(0), first)
        with self.assertRaises(RuntimeError):
            metrics.sample(data)

    def test_aggregate_pools_counts_and_error_samples(self):
        metrics = NoteTimingMetrics(events=[[.05, .1, 2, 1]], physics_dt=.0025,
                                    music_start=0, num_envs=2)
        data = history([(.05, .1, 2)], worlds=2)
        data[1] = False
        metrics.sample(data)
        rows = [metrics.result(i) for i in range(2)]
        combined = NoteTimingMetrics.aggregate(rows)
        self.assertEqual(combined['note_timing_matched_count'], 1)
        self.assertEqual(combined['note_timing_missed_count'], 1)
        self.assertAlmostEqual(combined['exact_original_music_micro_f1'], 2/3)
        self.assertEqual(combined['note_timing_short_recall'], .5)
        self.assertEqual(combined['note_timing_onset_mean_ms'], 0.)

    def test_legacy_coverage_uses_original_times_plus_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'score_coverage.json').write_text(json.dumps({'notes': [dict(
                start_time=.02, end_time=.08, pitch=23, fingering=1,
                target_start_time=0., target_end_time=.1)]}))
            bundle = SimpleNamespace(directory=directory, exact_note_events=None,
                manifest=dict(physics_dt=.01, control_dt=.025,
                              contract=dict(environment=dict(initial_buffer_time=1.))))
            metrics = NoteTimingMetrics(SimpleNamespace(bundle=bundle, num_envs=1))
            metrics.sample(history([(1.03, 1.09, 2)], dt=.01, duration=1.2))
            row = metrics.result()
            self.assertEqual(row['note_timing_source'], 'score_coverage_original_times')
            self.assertAlmostEqual(row['note_timing_onset_mean_ms'], 10.)
            self.assertAlmostEqual(row['note_timing_offset_mean_ms'], 10.)
            self.assertEqual(row['note_timing_records'][0]['target_start_s'], 1.02)

    def test_unavailable_is_none_and_truncated_scores_not_claimed_complete(self):
        missing = NoteTimingMetrics(SimpleNamespace(num_envs=1)).result()
        self.assertFalse(missing['note_timing_available'])
        self.assertIsNone(missing['note_timing_matched_count'])
        metrics = NoteTimingMetrics(events=[[.05, .4, 2, 1]], physics_dt=.0025)
        metrics.sample(history([(.05, .1, 2)]))
        self.assertFalse(metrics.result()['note_timing_available'])
        valid = NoteTimingMetrics(events=[[.05, .1, 2, 1]], physics_dt=.0025)
        valid.sample(None)
        self.assertIsNone(valid.result()['exact_original_music_micro_f1'])
        self.assertFalse(NoteTimingMetrics.aggregate([missing])['note_timing_available'])

    def test_invalid_events_and_history_are_not_silently_accepted(self):
        invalid = NoteTimingMetrics(events=[[.1, .05, 2, 1]], physics_dt=.0025)
        self.assertFalse(invalid.result()['note_timing_available'])
        valid = NoteTimingMetrics(events=[[.05, .1, 2, 1]], physics_dt=.0025)
        with self.assertRaises(ValueError):
            valid.sample(torch.zeros(1, 10, 88))
        with self.assertRaises(ValueError):
            valid.result(1)


if __name__ == '__main__':
    unittest.main()
