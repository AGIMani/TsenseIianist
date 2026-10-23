"""Offline evidence/score integrity checks; no MuJoCo model construction."""
import unittest
import numpy as np
from note_seq.protobuf import music_pb2
from scripts.revise_prelude_fingering import analyze, validate, alternative


def sequence():
    seq = music_pb2.NoteSequence()
    seq.notes.add(pitch=60,start_time=0.,end_time=.3,velocity=90,part=0,instrument=0)
    seq.notes.add(pitch=64,start_time=.1,end_time=.4,velocity=70,part=2,instrument=0)
    seq.control_changes.add(time=.2,control_number=64,control_value=127)
    return seq


class RevisionTests(unittest.TestCase):
    def test_validation_preserves_all_music_and_detects_conflicts(self):
        old=sequence(); new=music_pb2.NoteSequence.FromString(old.SerializeToString())
        new.notes[0].part=1
        self.assertTrue(validate(old,new)['all_fields_except_part_identical'])
        new.notes[0].part=2
        with self.assertRaises(AssertionError):validate(old,new)
        new.notes[0].part=1;new.notes[0].end_time=.29
        with self.assertRaises(AssertionError):validate(old,new)

    def test_late_unmatched_activation_counts_as_hold_not_total_miss(self):
        seq=music_pb2.NoteSequence()
        seq.notes.add(pitch=60,start_time=0,end_time=1,part=0)
        times=np.arange(1,401)*.005
        actual=np.zeros((400,88),dtype=bool)
        actual[(times>=1.2)&(times<2),39]=True
        ep=dict(note_timing_records=[dict(pitch=60,finger=0,target_start_s=1.,target_end_s=2.,status='missed')],
                note_timing_extra_records=[dict(pitch=60,played_start_s=1.2,played_end_s=2.)],note_timing_played_count=1)
        r=analyze(seq,dict(time=times,actual=actual),[ep],1.)[0]
        self.assertAlmostEqual(r['evaluation_hold_coverage'],.8)
        self.assertEqual(r['evaluation_onset_misses'],1)

    def test_sparse_alternatives_are_not_evidence(self):
        row=dict(pitch=63)
        sparse=[dict(finger_id=1,pitch=63,evidence_coverage=1.)]*2
        self.assertIsNone(alternative(sparse,row,1))


if __name__=='__main__':unittest.main()
