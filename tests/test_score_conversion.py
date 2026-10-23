"""Actual note-seq conversion regression checks (Composer Python environment)."""
import unittest
from note_seq import music_pb2
from tsensei_v3_2_pianist import music
from tsensei_v3_2_pianist.music.midi_file import NoteTrajectory
from tsensei_v3_2_pianist.score_validation import ScoreCoverageError, require_note_coverage


class ScoreConversionTests(unittest.TestCase):
    def test_three_erased_repeat_reproductions_are_rejected(self):
        for first, second in (((0, .1), (.1, .15)), ((0, .099), (.101, .149)), ((0, .02), (.025, .045))):
            seq = music_pb2.NoteSequence(total_time=second[1])
            for start, end in (first, second):
                seq.notes.add(pitch=60, velocity=90, start_time=start, end_time=end, part=0)
            with self.subTest(intervals=(first, second)):
                with self.assertRaises(ScoreCoverageError):
                    NoteTrajectory.seq_to_trajectory(seq, .05, score_validation='strict_v1')
                # Explicit historical mode remains available; no new silent rewriting.
                NoteTrajectory.seq_to_trajectory(seq, .05, score_validation='legacy')

    def test_current_song_strict_validation_keeps_every_existing_target(self):
        midi = music.load('TwinkleTwinkleRousseau').trim_silence()
        old = NoteTrajectory.from_midi(midi, .05)
        new = NoteTrajectory.from_midi(midi, .05, score_validation='strict_v1')
        self.assertEqual(old.notes, new.notes)
        self.assertEqual(old.sustains, new.sustains)
        report = require_note_coverage(midi.seq, new.notes, .05)
        self.assertEqual(report['source_notes'], 34)
        self.assertEqual(report['covered_notes'], 34)

    def test_shadow_floor_onset_ceil_release_does_not_modify_original_events(self):
        seq = music_pb2.NoteSequence(total_time=.121)
        seq.notes.add(pitch=60, velocity=90, start_time=.073, end_time=.121, part=0)
        original = seq.SerializeToString()
        for dt, expected in ((.05, [1, 2]), (.025, [2, 3, 4])):
            notes, _ = NoteTrajectory.seq_to_trajectory(seq, dt, score_validation='strict_v1')
            self.assertEqual([index for index, row in enumerate(notes) if row], expected)
        self.assertEqual(seq.SerializeToString(), original)

    def test_same_finger_quantized_overlap_preserves_upstream_targets(self):
        seq = music_pb2.NoteSequence(total_time=.14)
        seq.notes.add(pitch=74, velocity=90, start_time=0, end_time=.0716, part=1)
        seq.notes.add(pitch=72, velocity=90, start_time=.0794, end_time=.14, part=1)
        original = seq.SerializeToString()
        frames, sustain = NoteTrajectory.seq_to_trajectory(seq, .05, score_validation='strict_v1')
        upstream, upstream_sustain = NoteTrajectory.seq_to_trajectory(seq, .05, score_validation='legacy')
        self.assertEqual(frames, upstream)
        self.assertEqual(sustain, upstream_sustain)
        self.assertEqual({n.number for n in frames[1]}, {72, 74})
        self.assertEqual(seq.SerializeToString(), original)
        report = require_note_coverage(seq, frames, .05)
        self.assertEqual(len(report['warnings']), 1)
        self.assertEqual(report['covered_notes'], 2)

    def test_valid_long_repeat_retains_explicit_release(self):
        seq = music_pb2.NoteSequence(total_time=.2)
        seq.notes.add(pitch=60, velocity=90, start_time=0, end_time=.1, part=0)
        seq.notes.add(pitch=60, velocity=90, start_time=.1, end_time=.2, part=0)
        notes, _ = NoteTrajectory.seq_to_trajectory(seq, .05, score_validation='strict_v1')
        self.assertEqual([index for index, row in enumerate(notes) if row], [0, 1, 3])


if __name__ == '__main__':
    unittest.main()
