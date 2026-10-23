"""TsenseI task replacement preserves uniform targets and action contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from tsensei_v3_2_pianist.fingering_targets import manifest_config, target_position
from tsensei_v3_2_pianist.forearm_collision import collision_scope, is_forearm_collision_geom
from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds, policy_bounds
from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
from tsensei_v3_2_pianist.rl_torch.sac import SAC
from tsensei_v3_2_pianist.rl_torch.train import MotionMetrics

ROOT = Path(__file__).resolve().parents[1]


class ShadowTaskProfileChecks(unittest.TestCase):
    def test_all_fingers_share_shadow_target_but_old_thumb_stays_distinct(self):
        args = ([0., 0., 0.], [.075, .01125, .01125], 'piano/white_key_geom_0')
        thumb = target_position(*args, thumb=True, task_profile='tsensei_collision_v1')
        finger = target_position(*args, task_profile='tsensei_collision_v1')
        np.testing.assert_allclose(thumb, [.02625, 0., .005625])
        np.testing.assert_array_equal(thumb, finger)
        self.assertAlmostEqual(target_position(*args, thumb=True)[0], .055)
        self.assertIsNone(manifest_config('tsensei_collision_v1')['thumb'])
        self.assertIsNotNone(manifest_config()['thumb'])
        with self.assertRaises(ValueError):
            manifest_config('unknown')

    def test_tsensei_forearm_scope_includes_all_hand_colliders(self):
        from tsensei_v3_2_pianist.forearm_collision import palm_housing_body_name
        palm = 'right_/' + palm_housing_body_name('right') + '_collision_0'
        finger = 'left_/left_hand_index_DIP_link_collision_0'
        self.assertTrue(is_forearm_collision_geom(palm, 'tsensei_collision_v1'))
        self.assertTrue(is_forearm_collision_geom('right_/Right_hand_palm_link_collision_0', 'tsensei_collision_v1'))
        self.assertTrue(is_forearm_collision_geom(finger, 'tsensei_collision_v1'))
        self.assertTrue(is_forearm_collision_geom(finger))
        self.assertFalse(is_forearm_collision_geom(palm.replace('collision', 'visual'), 'tsensei_collision_v1'))
        self.assertNotEqual(collision_scope(), collision_scope('tsensei_collision_v1'))

    def test_shadow_task_uses_direct_joint_action_contract(self):
        rows = json.loads((ROOT/'configs/actions/direct_joint_v1.json').read_text())['channels']
        names = [r['name'] for r in rows]
        low = [r['physical_min'] for r in rows]
        high = [r['physical_max'] for r in rows]
        capped = (4, 8, 12, 16, 29, 33, 37, 41)
        for i in capped:
            high[i] = 1.57
        physical_low, physical_high = physical_bounds('direct_joint_v1', names, low, high)
        for i in capped:
            self.assertEqual(physical_low[i], 0.)
            self.assertEqual(physical_high[i], 1.2)
        self.assertEqual(policy_bounds('direct_joint_v1', 47)[0][4], 0.)
        self.assertEqual(policy_bounds('direct_joint_v1', 47)[1][4], 2.)
        with self.assertRaisesRegex(ValueError, 'Unknown action profile'):
            policy_bounds('shadow_v1', 47)
        old = SAC(7, 47, device='cpu', seed=7, action_profile='legacy')
        new = SAC(7, 47, device='cpu', seed=7, action_profile='direct_joint_v1')
        with self.assertRaisesRegex(ValueError, 'action profile'):
            new.load_state_dict(old.state_dict())

    def test_task_loader_requires_matching_explicit_shadow_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root/'scene'; scene_dir.mkdir()
            (scene_dir/'manifest.json').write_text('{}')
            task_dir = root/'task'; task_dir.mkdir()
            np.savez(task_dir/'score.npz', goal=np.zeros((1, 89)), fingers=np.zeros((1, 10, 88)))
            scene = SimpleNamespace(directory=scene_dir, neutral_action=np.zeros(51))
            palms = [['right_/right_hand_palm_link_collision_0'], ['left_/left_hand_palm_link_collision_0']]
            manifest = dict(schema_version=1, profile='cpu_compatible_fingered_v1',
                            scene_manifest_sha256=sha256(scene_dir/'manifest.json'),
                            files={'score.npz': sha256(task_dir/'score.npz')}, length=1, control_dt=.05,
                            reset_action=[0.] * 51, palm_geoms=palms,
                            contract={'environment': {}}, fingering_target=manifest_config())
            scene.model = SimpleNamespace(ngeom=2, geom=lambda i: SimpleNamespace(name=palms[i][0]))
            def write():
                (task_dir/'task.json').write_text(json.dumps(manifest))
            write()
            TaskBundle(task_dir, scene)  # Old manifest: no task_profile field.
            manifest.update(task_profile='tsensei_collision_v1', fingering_target=manifest_config('tsensei_collision_v1'),
                            forearm_collision_scope=collision_scope('tsensei_collision_v1'), forearm_collision_geoms=palms)
            manifest['contract']['environment']['task_profile'] = 'tsensei_collision_v1'
            write()
            TaskBundle(task_dir, scene)
            manifest['fingering_target'] = manifest_config()
            write()
            with self.assertRaisesRegex(ValueError, 'target geometry'):
                TaskBundle(task_dir, scene)
            manifest['fingering_target'] = manifest_config('tsensei_collision_v1')
            manifest['forearm_collision_geoms'] = [['finger'], ['finger']]
            write()
            with self.assertRaisesRegex(ValueError, 'all physical hand'):
                TaskBundle(task_dir, scene)

    def test_motion_metrics_capture_first_target_and_episode_maxima(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><joint name="tx" type="slide"/><geom size=".01"/></body>
          <body><joint name="ty" type="slide"/><geom size=".01"/></body>
          <body><joint name="finger"/><geom size=".01"/></body>
          <body><joint name="pitch"/><geom size=".01"/></body>
          </worldbody><actuator><position joint="tx"/><position joint="ty"/>
          <position joint="finger"/><position joint="pitch"/></actuator></mujoco>''')
        scene = SimpleNamespace(model=model, device='cpu',
            bundle=SimpleNamespace(active_action_indices=[0, 1, 2, 3], action_ids=[0, 1, 2, 3],
                manifest={'action_names': ['R_Palm_Base_Tx', 'R_Palm_Base_Ty', 'R_Index_MCP', 'R_Palm_Base_Pitch'],
                          'control_dt': .05}),
            data=SimpleNamespace(ctrl=torch.tensor([[.1, .2, .3, .4]]), qvel=torch.zeros(1, 4)))
        metrics = MotionMetrics(scene)
        metrics.reset()
        scene.data.ctrl += torch.tensor([[.2, -.1, -.25, .05]])
        scene.data.qvel[:] = torch.tensor([[-3., 2., -10., .5]])
        metrics.sample()
        scene.data.ctrl.fill_(100.)  # The next episode reset must not count as a target jump.
        metrics.reset()
        scene.data.qvel.zero_()
        metrics.sample()
        result = metrics.result()
        self.assertAlmostEqual(result['motion_max_wrist_tx_speed_m_s'], 3.)
        self.assertAlmostEqual(result['motion_max_finger_speed_rad_s'], 10.)
        self.assertAlmostEqual(result['motion_max_wrist_tx_target_jump_m'], .2)
        self.assertAlmostEqual(result['motion_max_finger_target_jump_rad'], .25)
        self.assertEqual(result['motion_sample_hz'], 20.)


if __name__ == '__main__':
    unittest.main()
