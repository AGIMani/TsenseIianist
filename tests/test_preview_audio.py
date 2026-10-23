"""Audio must follow physical key edges, including wrong notes and pedal release."""
import unittest
import numpy as np
from tsensei_v3_2_pianist.preview_audio import events_from_states


class ActualKeyAudioTest(unittest.TestCase):
    def events(self, states, pedal=None):
        q = np.asarray(states,dtype=float)
        return events_from_states(q,np.tile([0.,1.],(88,1)),
                                  np.zeros(len(q)) if pedal is None else pedal,.0005,1.)

    def test_held_key_is_not_retriggered_and_repress_is(self):
        q = np.zeros((6,88));q[1:3,39]=1;q[4,39]=1
        events = self.events(q)
        self.assertEqual([(e['type'],e['note']) for e in events],
                         [('note_on',60),('note_off',60),('note_on',60),('note_off',60)])
        np.testing.assert_allclose([e['time'] for e in events],[.001,.002,.0025,.003])

    def test_all_actual_notes_are_kept_without_score_filter(self):
        q = np.zeros((3,88));q[0:2,[0,39,87]]=1
        events = self.events(q)
        self.assertEqual([e['note'] for e in events if e['type']=='note_on'],[21,60,108])
        self.assertEqual(sum(e['type']=='note_off' for e in events),3)

    def test_pedal_and_end_release_are_explicit(self):
        q = np.zeros((4,88));q[1:,39]=1
        events = self.events(q,[0,1,1,1])
        self.assertEqual([(e['value'],e.get('end_of_replay',False)) for e in events if e['type']=='sustain'],[(127,False),(0,True)])
        self.assertTrue(next(e for e in events if e['type']=='note_off')['end_of_replay'])

    def test_silence_and_invalid_state(self):
        self.assertEqual(self.events(np.zeros((4,88))),[])
        with self.assertRaises(ValueError):self.events(np.full((2,88),np.nan))


if __name__ == '__main__':
    unittest.main()
