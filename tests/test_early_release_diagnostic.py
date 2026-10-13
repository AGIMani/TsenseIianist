import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/hpc/early_release_diagnostic.py'
SPEC = importlib.util.spec_from_file_location('early_release_diagnostic', PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class EarlyReleaseAuditTest(unittest.TestCase):
    def test_real_offset_frames_include_straddling_control(self):
        self.assertEqual(AUDIT.probe_frames(5.755208333333, .025), [228, 229, 230])
        self.assertEqual(AUDIT.probe_frames(8.5, .025), [337, 338, 339])

    def test_policy_finger_channels_and_hand(self):
        names = ['R_Index_MCP', 'R_Index_SW', 'R_Wrist_TX',
                 'L_Thumb_Rot', 'L_Thumb_MCP', 'L_Wrist_TX', 'sustain']
        self.assertEqual(AUDIT.finger_channels(names, 1), ([0, 1], [0, 1, 2]))
        self.assertEqual(AUDIT.finger_channels(names, 5), ([3, 4], [3, 4, 5]))

    def test_repairs_only_final_missing_target_samples(self):
        active = np.zeros((10, 88), dtype=bool)
        goal = np.zeros_like(active)
        goal[:8, 40] = True
        active[1:3, 40] = True
        active[4:6, 40] = True
        # Half-open note interval ends at .09, leaving .07 and .08 early.
        result = AUDIT.early_release_mask(active, goal, [[.01, .09, 40, 2]], .01)
        self.assertEqual(result[0]['reference'], 5)
        self.assertEqual(result[0]['samples'].tolist(), [6, 7])
        self.assertEqual(len(result), 1)

    def test_wholly_missing_note_is_not_early_release(self):
        state = np.zeros((10, 88), dtype=bool)
        self.assertEqual(AUDIT.early_release_mask(state, state, [[.01, .09, 40, 2]], .01), [])

    def test_inconsistent_goal_cannot_inflate_release_repair(self):
        state = np.zeros((10, 88), dtype=bool)
        state[0, 40] = True
        with self.assertRaisesRegex(ValueError, 'score boundary'):
            AUDIT.early_release_mask(state, np.zeros_like(state), [[.01, .09, 40, 2]], .01)


if __name__ == '__main__':
    unittest.main()
