"""Future assignments must respect score time, ownership and episode boundaries."""
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.anticipation import anticipation_fields


class AnticipationTest(unittest.TestCase):
    def test_upcoming_note_target_above_unpressed_key_and_observed_owner(self):
        assignment = torch.zeros(2,10,88)
        assignment[0,5,20] = 1; assignment[1,1,30] = 1
        target = torch.zeros(2,10,88,3)
        target[0,5,20] = torch.tensor([.1,.2,.08])
        target[1,1,30] = torch.tensor([.3,.4,.09])
        tips = torch.full((2,10,3),.03)
        seconds = torch.zeros(2,10); seconds[0,5]=.15;seconds[1,1]=.05
        fields, obs = anticipation_fields(assignment,seconds,target,tips,torch.full((88,),.1))
        torch.testing.assert_close(fields['anticipatory_targets'][0,5],torch.tensor([.1,.2,.105]))
        self.assertEqual(obs.shape,(2,50))
        torch.testing.assert_close(obs.reshape(2,10,5)[0,5],torch.tensor([.07,.17,.075,.15,1.]))
        self.assertTrue((obs.reshape(2,10,5)[0,:5]==0).all())


if __name__=='__main__': unittest.main()
