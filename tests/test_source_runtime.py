"""Source replay must survive removal of the original compile directory."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from tsensei_v3_2_pianist.mjlab_env import source_runtime as runtime
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
from tsensei_v3_2_pianist.mjlab_env.paths import for_recording


class SourceRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = runtime.make_recipe()
        cls.scene_path, cls.task_path, _ = runtime.build(cls.recipe)
        cls.scene = SceneBundle(cls.scene_path)
        cls.task = TaskBundle(cls.task_path, cls.scene)

    def test_buffer_fingering_and_gravity(self):
        task, scene = self.task, self.scene
        self.assertEqual(np.flatnonzero(task.goal[:, :88].any(axis=1))[0], 20)
        self.assertFalse(task.goal[:20].any())
        self.assertFalse(task.fingers[:20].any())
        self.assertTrue(scene.manifest['gravity_compensation'])
        self.assertEqual(np.count_nonzero(scene.model.body_gravcomp == 1), 44)
        old_scene, old_task, _ = runtime.build(runtime.make_recipe(initial_buffer_time=0))
        baseline = TaskBundle(old_task, SceneBundle(old_scene))
        np.testing.assert_array_equal(task.goal[20:], baseline.goal)
        np.testing.assert_array_equal(task.fingers[20:], baseline.fingers)
        self.assertGreater(len(set(np.nonzero(task.fingers)[1])), 1)
        self.assertEqual(task.manifest['length'], baseline.manifest['length']+20)

    def test_rebuild_after_original_workspace_removed(self):
        meta = dict(source_runtime=self.recipe, no_arms=True,
                    scene_sha256=sha256(self.scene_path/'manifest.json'),
                    task_sha256=sha256(self.task_path/'task.json'))
        # Remove the actual compiler products; only source + JSON metadata remain.
        for workspace in runtime._LIVE_WORKSPACES:
            if Path(workspace.name) == self.scene_path.parent:
                workspace.cleanup()
        self.assertFalse(self.scene_path.exists())
        scene, task = for_recording(json.loads(json.dumps(meta)))
        self.assertEqual(sha256(scene/'manifest.json'), meta['scene_sha256'])
        self.assertEqual(sha256(task/'task.json'), meta['task_sha256'])
        TaskBundle(task, SceneBundle(scene))

    def test_reject_source_and_config_drift(self):
        changed = copy.deepcopy(self.recipe)
        changed['source_sha256'][next(iter(changed['source_sha256']))] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'source differs'):
            runtime.build(changed)
        spec = importlib.util.spec_from_file_location('eval_from_source', runtime.ROOT/'scripts/eval_from_source.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        config = dict(source_runtime=self.recipe, sac={'gamma': .8})
        module.validate_config(config, {**config, 'updates': 3, 'vector_step': 4})
        with self.assertRaisesRegex(ValueError, 'differs'):
            module.validate_config({**config, 'sac': {'gamma': .9}}, config)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            module.validate_config({'source_runtime': self.recipe}, config)

    def test_saved_shadow_profiles_override_ambient_shell_and_rebuild(self):
        recipe = runtime.make_recipe(initial_buffer_time=0, task_profile='tsensei_collision_v1',
                                     collision_profile='shadow_primitive_v1', drive_profile='shadow_v1')
        with patch.dict(os.environ, {'TSENSEI_V32_COLLISION_PROFILE': 'legacy_mesh_v1',
                                     'TSENSEI_V32_DRIVE_PROFILE': 'legacy_v1'}):
            scene_path, task_path, _ = runtime.build(recipe)
        scene = SceneBundle(scene_path, disable_wrist_tz=True, disable_wrist_yaw=True)
        for side in ('right', 'left'):
            joint = scene.model.joint(f'{side}_/forearm_tx').id
            self.assertAlmostEqual(scene.model.dof_armature[scene.model.jnt_dofadr[joint]], .02)
        task = TaskBundle(task_path, SceneBundle(scene_path))
        self.assertEqual(task.manifest['task_profile'], 'tsensei_collision_v1')
        self.assertEqual(list(map(len, task.manifest['forearm_collision_geoms'])), [22, 22])
        expected = (sha256(scene_path/'manifest.json'), sha256(task_path/'task.json'))
        for workspace in runtime._LIVE_WORKSPACES:
            if Path(workspace.name) == scene_path.parent:
                workspace.cleanup()
        rebuilt_scene, rebuilt_task, _ = runtime.build(json.loads(json.dumps(recipe)))
        self.assertEqual(expected, (sha256(rebuilt_scene/'manifest.json'), sha256(rebuilt_task/'task.json')))

    def test_thumb_distances_survive_source_json_and_legacy_missing_field(self):
        from tsensei_v3_2_pianist.fingering_targets import options_from_manifest, target_position
        recipe = runtime.make_recipe(initial_buffer_time=0, task_profile='tsensei_collision_v1',
                                     thumb_white_target_distance=.02, thumb_black_target_distance=.015)
        recipe = json.loads(json.dumps(recipe))
        # Diagnostics rebuild by unpacking a saved environment mapping.
        rebuilt_recipe = runtime.make_recipe(**recipe['environment'])
        self.assertEqual(rebuilt_recipe['environment'], recipe['environment'])
        scene_path, task_path, _ = runtime.build(recipe)
        scene = SceneBundle(scene_path)
        task = TaskBundle(task_path, scene)
        options = options_from_manifest(task.manifest['fingering_target'], 'tsensei_collision_v1')
        self.assertEqual(options, recipe['environment']['thumb_target'])
        self.assertEqual(options, task.manifest['contract']['environment']['thumb_target'])
        for name in task.manifest['key_geoms']:
            geom = scene.model.geom(name)
            distance = .02 if 'white_key_geom' in name else .015
            offset = target_position(np.zeros(3), geom.size, name, thumb=True,
                                     task_profile='tsensei_collision_v1', thumb_target=options)
            self.assertAlmostEqual(offset[0], geom.size[0] - distance)
        # Old source recipes had no options field: Shadow keeps uniform targets,
        # not the new user-specified distances from a different recipe.
        recipe['environment'].pop('thumb_target')
        legacy_scene, legacy_task, _ = runtime.build(recipe)
        legacy = TaskBundle(legacy_task, SceneBundle(legacy_scene))
        self.assertEqual(legacy.manifest['fingering_target']['schema'], 1)
        self.assertIsNone(legacy.manifest['fingering_target']['thumb'])


if __name__ == '__main__':
    unittest.main()
