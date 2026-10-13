import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from tsensei_v3_2_pianist.fingering_targets import (
    manifest_config, options_from_manifest, target_position, thumb_target_options,
    validate_thumb_target,
)


class FingeringTargetTest(unittest.TestCase):
    def test_thumb_distance_is_measured_from_positive_world_x_end(self):
        white = target_position([0, 0, 0], [.075, .01125, .01125],
                                'piano/white_key_geom_0', thumb=True)
        black = target_position([-.03, 0, 0], [.045, .005, .009],
                                'piano/black_key_geom_1', thumb=True)
        np.testing.assert_allclose(white, [.055, 0, .005625], atol=1e-12)
        np.testing.assert_allclose(black, [0, 0, .0045], atol=1e-12)

    def test_non_thumb_retains_legacy_target(self):
        target = target_position([0, 0, 0], [.075, .01125, .01125],
                                 'piano/white_key_geom_0', thumb=False)
        np.testing.assert_allclose(target, [.02625, 0, .005625], atol=1e-12)

    def test_manifest_records_user_selected_geometry(self):
        thumb = manifest_config()['thumb']
        self.assertEqual(thumb['global_finger_indices'], [0, 5])
        self.assertEqual(thumb['reference_edge'], 'positive_world_x_end')
        self.assertEqual(thumb['distance_toward_negative_world_x_m'],
                         {'white': .02, 'black': .015})

    def test_explicit_distances_override_shadow_for_thumbs_only(self):
        options = thumb_target_options(white_distance=.02, black_distance=.015)
        for profile in ('tsensei_collision_v1', 'tsensei_v1'):
            for name, size, distance in (('white', .075, .02), ('black', .045, .015)):
                args = ([.1, .2, .3], [size, .01, .02], f'piano/{name}_key_geom_0')
                with self.subTest(profile=profile, colour=name):
                    thumb = target_position(*args, thumb=True, task_profile=profile,
                                            thumb_target=options)
                    np.testing.assert_allclose(thumb, [.1 + size - distance, .2, .31])
                    finger = target_position(*args, task_profile=profile, thumb_target=options)
                    np.testing.assert_allclose(finger, [.1 + .35 * size, .2, .31])

    def test_partial_override_retains_other_colour_profile_and_uniform_is_explicit(self):
        options = thumb_target_options(white_distance=.025)
        args = ([0, 0, 0], [.045, .01, .02], 'piano/black_key_geom_0')
        self.assertAlmostEqual(target_position(*args, thumb=True, thumb_target=options)[0], .03)
        self.assertAlmostEqual(target_position(*args, thumb=True, task_profile='tsensei_collision_v1',
                                              thumb_target=options)[0], .01575)
        self.assertAlmostEqual(target_position(*args, thumb=True,
                                              thumb_target=thumb_target_options('uniform'))[0], .01575)

    def test_manifest_roundtrip_and_historical_contract(self):
        for profile in ('tsensei_collision_v1', 'tsensei_v1'):
            self.assertIsNone(options_from_manifest(manifest_config(profile), profile))
            for options in (thumb_target_options(), thumb_target_options('uniform'),
                            thumb_target_options(white_distance=.02, black_distance=.015)):
                with self.subTest(profile=profile, options=options):
                    manifest = manifest_config(profile, options)
                    self.assertEqual(manifest['schema'], 2)
                    self.assertEqual(options_from_manifest(manifest, profile), options)
                    manifest['default']['x_offset_half_size'] = .6
                    with self.assertRaisesRegex(ValueError, 'target geometry'):
                        options_from_manifest(manifest, profile)

    def test_invalid_distances_or_ambiguous_mode_fail(self):
        for value in (-.001, float('nan'), float('inf'), True, '0.02', [0.02]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                thumb_target_options(white_distance=value)
        with self.assertRaisesRegex(ValueError, 'Uniform'):
            thumb_target_options('uniform', white_distance=.02)
        with self.assertRaisesRegex(ValueError, 'Invalid thumb'):
            validate_thumb_target({'mode': 'profile'})
        with self.assertRaisesRegex(ValueError, 'key length'):
            target_position([0, 0, 0], [.045, .01, .02], 'piano/black_key_geom_0',
                            thumb=True, thumb_target=thumb_target_options(black_distance=.1))
        with self.assertRaisesRegex(ValueError, 'requires explicit'):
            options_from_manifest({**manifest_config('tsensei_collision_v1'), 'schema': 2}, 'tsensei_collision_v1')

    def test_source_recipe_validation_does_not_require_site_packages(self):
        # run.sh invokes the system Python; Composer and Torch are selected
        # later. Keep option validation usable even when NumPy is unavailable.
        root = Path(__file__).resolve().parents[1]
        code = '''import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe
with patch('tsensei_v3_2_pianist.mjlab_env.source_runtime.source_files', return_value={}), \\
     patch('tsensei_v3_2_pianist.mjlab_env.source_runtime.dependency_versions', return_value={}):
    recipe = make_recipe(task_profile='tsensei_collision_v1', thumb_white_target_distance=.02)
    rebuilt = make_recipe(**recipe['environment'])
    assert rebuilt['environment'] == recipe['environment']
    try:
        make_recipe(thumb_target=recipe['environment']['thumb_target'], thumb_white_target_distance=.03)
    except ValueError as error:
        assert 'cannot also specify' in str(error)
    else:
        raise AssertionError('Canonical and flat options must not silently override one another')
assert recipe['environment']['thumb_target']['white_distance'] == .02
assert 'numpy' not in sys.modules
'''
        subprocess.run([sys.executable, '-S', '-c', code, str(root)], check=True)


if __name__ == '__main__':
    unittest.main()
