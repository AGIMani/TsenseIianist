"""Dependency-free checks for score integrity and explicit release tails."""
from types import SimpleNamespace as NS
import unittest
from tsensei_v3_2_pianist.score_validation import (
    audit_note_coverage, require_note_coverage, ScoreCoverageError, release_tail_steps,
)


def note(start, end, pitch=60, finger=0):
    return NS(start_time=start, end_time=end, pitch=pitch, part=finger, velocity=90)


def target(pitch=60, finger=0):
    return NS(key=pitch - 21, fingering=finger)


class ScoreValidationTests(unittest.TestCase):
    def test_short_repeat_is_rejected_when_second_note_disappears(self):
        seq = NS(notes=[note(0, .1), note(.1, .15)])
        with self.assertRaises(ScoreCoverageError) as caught:
            require_note_coverage(seq, [[target()], [target()], [], []], .05)
        self.assertEqual(caught.exception.report['covered_notes'], 1)
        self.assertEqual(caught.exception.report['errors'][0]['kind'], 'missing_note')

    def test_deleting_release_gap_is_not_an_allowed_fix(self):
        seq = NS(notes=[note(0, .1), note(.1, .2)])
        report = audit_note_coverage(seq, [[target()] for _ in range(4)], .05)
        self.assertEqual(report['covered_notes'], 2)
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['kind'], 'merged_repeat')

    def test_valid_repeat_reports_quantization_without_changing_frames(self):
        seq = NS(notes=[note(0, .1), note(.1, .2)])
        frames = [[target()], [target()], [], [target()], []]
        report = require_note_coverage(seq, frames, .05)
        self.assertEqual(report['covered_notes'], 2)
        self.assertAlmostEqual(report['notes'][1]['onset_error_seconds'], .05)
        self.assertEqual([len(row) for row in frames], [1, 1, 0, 1, 0])

    def test_same_frame_overwrites_do_not_count_as_coverage(self):
        seq = NS(notes=[note(0, .02), note(.025, .045)])
        report = audit_note_coverage(seq, [[target()]], .05)
        self.assertFalse(report['valid'])
        self.assertEqual(report['covered_notes'], 1)

    def test_same_finger_cannot_play_two_keys(self):
        seq = NS(notes=[note(0, .05), note(0, .05, 62)])
        report = audit_note_coverage(seq, [[target(), target(62)]], .05)
        self.assertIn('finger_conflict', [item['kind'] for item in report['errors']])

    def test_distinct_fingers_can_play_chord(self):
        seq = NS(notes=[note(0, .05), note(0, .05, 62, 1)])
        report = require_note_coverage(seq, [[target(), target(62, 1)]], .05)
        self.assertEqual(report['covered_notes'], 2)

    def test_invalid_source_notes_rejected(self):
        for bad in (note(-.1, .1), note(0, 0), note(0, .1, 109), note(0, .1, 60, 10)):
            with self.subTest(note=bad):
                with self.assertRaises(ScoreCoverageError):
                    require_note_coverage(NS(notes=[bad]), [[]], .05)

    def test_tail_must_contain_whole_frames(self):
        self.assertEqual(release_tail_steps(.25, .05), 5)
        self.assertEqual(release_tail_steps(0, .05), 0)
        for value in (-.1, .23, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                release_tail_steps(value, .05)


if __name__ == '__main__':
    unittest.main()
