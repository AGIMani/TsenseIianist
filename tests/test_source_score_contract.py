"""Composer-built score boundaries remain explicit in a reconstructed task."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from tsensei_v3_2_pianist.mjlab_env import source_runtime as runtime
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle


class SourceScoreContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        common = dict(initial_buffer_time=0, preparation_mode='policy_v1',
                      preparation_time=1, task_profile='tsensei_collision_v1', drive_profile='shadow_v1')
        cls.recipe = runtime.make_recipe(**common, release_tail_time=.25, score_validation='strict_v1')
        cls.scene_path, cls.task_path, _ = runtime.build(cls.recipe)
        cls.scene = SceneBundle(cls.scene_path)
        cls.task = TaskBundle(cls.task_path, cls.scene)
        legacy = runtime.make_recipe(**common)
        legacy['environment'].pop('release_tail_time')
        legacy['environment'].pop('score_validation')
        legacy_scene, legacy_task, _ = runtime.build(legacy)
        cls.legacy = TaskBundle(legacy_task, SceneBundle(legacy_scene))
        original = runtime.make_recipe(**common, release_tail_time=.25,
                                       score_validation='strict_v1', exact_note_timing=True)
        original_scene, original_task, _ = runtime.build(original)
        cls.original = TaskBundle(original_task, SceneBundle(original_scene))
        cls.original_report = json.loads((original_task/'score_coverage.json').read_text())

    def test_original_midi_diagnostics_preserve_shadow_control_grid_and_coverage(self):
        np.testing.assert_array_equal(self.original.goal, self.task.goal)
        np.testing.assert_array_equal(self.original.fingers, self.task.fingers)
        self.assertEqual(len(self.original.exact_note_events), 34)
        grid = self.original_report['control_frame_coverage']
        self.assertEqual(grid['schema'], 'strict_note_coverage_v1')
        self.assertTrue(grid['valid'])
        self.assertEqual(grid['covered_notes'], 34)
        self.assertFalse(self.original.goal[:20].any())
        self.assertTrue(self.original.goal[20, :88].any())
        self.assertFalse(self.original.goal[-5:].any())


    def test_music_unchanged_and_tail_has_five_real_empty_targets(self):
        self.assertEqual(self.task.release_tail_steps, 5)
        self.assertEqual(self.task.music_end_step, 178)
        self.assertEqual(len(self.task.goal), 183)
        np.testing.assert_array_equal(self.task.goal[:-5], self.legacy.goal)
        np.testing.assert_array_equal(self.task.fingers[:-5], self.legacy.fingers)
        self.assertFalse(self.task.goal[-5:].any())
        self.assertFalse(self.task.fingers[-5:].any())

    def test_historical_missing_options_remain_no_tail_and_legacy(self):
        self.assertEqual(self.legacy.release_tail_steps, 0)
        self.assertEqual(self.legacy.music_end_step, len(self.legacy.goal))
        self.assertNotIn('score_coverage.json', self.legacy.manifest['files'])

    def test_coverage_report_is_hashed_with_task_and_has_all_notes(self):
        self.assertIn('score_coverage.json', self.task.manifest['files'])
        report = json.loads((self.task_path/'score_coverage.json').read_text())
        self.assertTrue(report['valid'])
        self.assertEqual(report['source_notes'], 34)
        self.assertEqual(report['covered_notes'], 34)

    def test_corrupt_music_boundary_is_rejected(self):
        original = (self.task_path/'task.json').read_text()
        manifest = json.loads(original)
        manifest['music_end_step'] -= 1
        try:
            (self.task_path/'task.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'music end'):
                TaskBundle(self.task_path, self.scene)
        finally:
            (self.task_path/'task.json').write_text(original)

    def test_recipe_roundtrip_and_invalid_tail(self):
        with patch.object(runtime, 'source_files', return_value={}), patch.object(runtime, 'dependency_versions', return_value={}):
            env = self.recipe['environment']
            self.assertEqual(runtime.make_recipe(**env)['environment'], env)
            for seconds in (.23, -.1, float('nan')):
                with self.assertRaisesRegex(ValueError, 'Release tail'):
                    runtime.make_recipe(release_tail_time=seconds)
            with self.assertRaisesRegex(ValueError, 'validation mode'):
                runtime.make_recipe(score_validation='guess')


if __name__ == '__main__':
    unittest.main()
