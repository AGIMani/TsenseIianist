"""Physical event boundaries, observable subframe times, and unchanged legacy rules."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tsensei_v3_2_pianist.mjlab_env.exact_score import (
    ExactScore, sample_targets, sequence_events, validate_events, physical_coverage)


def timeline(events, **kwargs):
    return ExactScore(np.array(events, dtype=np.float64), np.empty((0, 2)),
        length=kwargs.pop('length', 8), control_dt=.025, physics_dt=.0025,
        lookahead_steps=20, anticipation_horizon=.15, **kwargs)


class ExactClockTest(unittest.TestCase):
    def test_half_open_physics_endpoint_and_control_clocks(self):
        score = timeline([[.0125, .0375, 10, 1]])
        self.assertFalse(score.goal[0, 10])
        self.assertTrue(score.goal[1, 10])
        self.assertTrue(score.end_goal[0, 10])
        self.assertFalse(score.end_goal[1, 10])
        self.assertEqual(score.goal_history[:, :, 10].sum().item(), 10)
        self.assertFalse(score.goal_history[0, 3, 10])  # t=.010
        self.assertTrue(score.goal_history[0, 4, 10])   # t=.0125
        self.assertFalse(score.goal_history[1, 4, 10])  # t=.0375
        torch.testing.assert_close(score.end_goal, score.goal_history[:, -1])

    def test_subframe_note_visible_without_a_control_target(self):
        score = timeline([[.043, .049, 20, 2]])
        self.assertFalse(score.goal[:, 20].any())
        self.assertEqual(score.goal_history[:, :, 20].sum().item(), 2)
        self.assertAlmostEqual(score.observation[1, 20].item(), .018)
        self.assertAlmostEqual(score.observation[1, 88 + 20].item(), .024)
        self.assertTrue(score.future[1, 2, 20])
        self.assertAlmostEqual(score.future_seconds[1, 2].item(), .018)
        self.assertEqual(score.observation[2, 20].item(), -1.)
        self.assertFalse(score.future[2].any())

    def test_sequential_notes_do_not_become_artificial_chord(self):
        score = timeline([[0., .018, 10, 1], [.020, .040, 20, 1]])
        self.assertLessEqual(score.goal_history[:, :, :88].sum(-1).max().item(), 1.)
        # The same control period contains both pitches, but no physical sample does.
        self.assertTrue(score.goal_history[0, :, 10].any())
        self.assertTrue(score.goal_history[0, :, 20].any())
        self.assertFalse(score.goal_history[0, 6, 20].any())  # t=.0175 is still only the first note

    def test_preparation_source_buffer_and_release_tail(self):
        seq = SimpleNamespace(notes=[SimpleNamespace(start_time=.0125, end_time=.05, pitch=60, part=5)],
            control_changes=[SimpleNamespace(time=.01, control_value=127, control_number=64)])
        events, sustain = sequence_events(seq, 1.05, .025, 1.125)
        np.testing.assert_allclose(events[0], [1.0625, 1.10, 39, 5])
        score = ExactScore(events, sustain, length=50, control_dt=.025, physics_dt=.0025,
                           lookahead_steps=20)
        self.assertFalse(score.goal_history[:42, :, :88].any())
        self.assertFalse(score.goal_history[44:, :, :88].any())
        self.assertTrue(score.goal_history[43, :, -1].all())
        self.assertFalse(score.goal_history[45:, :, -1].any())

    def test_multiple_env_indices_and_no_terminal_future_wrap(self):
        score = timeline([[0., .025, 10, 1], [.05, .075, 20, 2]])
        indices = torch.tensor([0, 2, 0, 7])
        self.assertEqual(score.goal_history[indices].shape, (4, 10, 89))
        torch.testing.assert_close(score.goal_history[indices][0], score.goal_history[indices][2])
        self.assertFalse(score.future[7].any())
        self.assertTrue((score.observation[7] == -1).all())

    def test_non_strict_reports_unsampled_gap_without_inserting_release(self):
        events = np.array([[0, .0091, 10, 1], [.010, .02, 10, 1]], dtype=np.float64)
        score = ExactScore(events, np.empty((0, 2)), length=2, control_dt=.02,
                           physics_dt=.002, lookahead_steps=10, score_validation='legacy')
        np.testing.assert_array_equal(score.events, events)
        report = score.physical_coverage
        self.assertFalse(report['valid'])
        self.assertEqual(report['missing_note_samples'], [])
        self.assertEqual(len(report['missing_release_samples']), 1)
        self.assertAlmostEqual(report['missing_release_samples'][0]['gap_seconds'], .0009)
        self.assertTrue(score.goal_history[0, :9, 10].all())  # no invented release
        self.assertFalse(score.goal_history[0, 9, 10])
        with self.assertRaisesRegex(ValueError, 'no release sample'):
            ExactScore(events, np.empty((0, 2)), length=2, control_dt=.02,
                       physics_dt=.002, lookahead_steps=10, score_validation='strict_v1')

    def test_non_strict_reports_missing_note_and_retains_input_checks(self):
        score = timeline([[.0201, .0202, 10, 1]], score_validation='legacy')
        self.assertFalse(score.physical_coverage['valid'])
        self.assertEqual(len(score.physical_coverage['missing_note_samples']), 1)
        self.assertFalse(score.goal_history[:, :, 10].any())
        for mode in ('legacy', 'strict_v1'):
            with self.assertRaisesRegex(ValueError, 'Overlapping real MIDI'):
                timeline([[0, .03, 10, 1], [.02, .04, 20, 1]], score_validation=mode)
            with self.assertRaisesRegex(ValueError, 'finite'):
                timeline([[0, float('nan'), 10, 1]], score_validation=mode)
        with self.assertRaisesRegex(ValueError, 'Unknown score validation mode'):
            timeline([[0, .02, 10, 1]], score_validation='off')

    def test_real_finger_key_conflict_and_unresolvable_events_fail(self):
        for events in ([[0, .03, 10, 1], [.02, .04, 20, 1]],
                       [[0, .03, 10, 1], [.02, .04, 10, 2]]):
            with self.assertRaisesRegex(ValueError, 'Overlapping real MIDI'):
                timeline(events)
        with self.assertRaisesRegex(ValueError, 'no physical target sample'):
            timeline([[.0201, .0202, 10, 1]])
        with self.assertRaisesRegex(ValueError, 'no release sample'):
            timeline([[.01, .0201, 10, 1], [.0202, .03, 10, 1]])
        # Real 5 ms release gap remains valid even inside the same 25 ms frame.
        timeline([[.0025, .01, 10, 1], [.015, .0225, 10, 1]])


if __name__ == '__main__':
    unittest.main()
