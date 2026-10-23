import unittest
import numpy as np
from tsensei_v3_2_pianist.fingering_targets import manifest_config, options_from_manifest, target_position

class FingeringTargetTest(unittest.TestCase):
    def test_all_fingers_distance_and_unchanged_y_z(self):
        for profile in ('tsensei_v1', 'tsensei_collision_v1'):
            for finger in range(10):
                for colour, center, half, distance in [('white', 0., .075, .040), ('black', -.03, .045, .020)]:
                    target = target_position([center, .2, .3], [half, .01, .02], f'piano/{colour}_key_geom_0', thumb=finger in (0,5), task_profile=profile)
                    np.testing.assert_allclose(target, [center+half-distance, .2, .31])
                    self.assertAlmostEqual(center+half-target[0], distance)
    def test_contract_rejects_old_or_missing_geometry(self):
        self.assertIsNone(options_from_manifest(manifest_config()))
        for value in (None, {}, {'schema': 1}, {**manifest_config(), 'schema': 2}):
            with self.assertRaises(ValueError): options_from_manifest(value)
    def test_bad_geometry(self):
        with self.assertRaises(ValueError): target_position([0,0,0], [.001,.01,.01], 'white_key_geom')
        with self.assertRaises(ValueError): target_position([0,0,float('nan')], [.075,.01,.01], 'white_key_geom')

if __name__ == '__main__': unittest.main()
