"""TZ exclusion is a model topology change, not a state correction."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.rl_torch.checkpoint import check_tz_physics

SCENE = Path(__file__).resolve().parents[1] / 'training_mjlab/outputs/scene_twinkle_h12_20hz200hz_20260924'


class StructuralTZChecks(unittest.TestCase):
    @unittest.skipUnless(SCENE.exists(), 'Requires the frozen 12 cm scene')
    def test_joint_absence_and_native_external_force(self):
        enabled = SceneBundle(SCENE)
        locked = SceneBundle(SCENE, disable_wrist_tz=True)
        self.assertEqual((enabled.model.nq, enabled.model.nu), (138, 50))
        m = locked.model
        self.assertEqual((m.nq, m.nv, m.nu), (136, 136, 48))
        for side, prefix in (('right', 'R'), ('left', 'L')):
            self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, side+'_/forearm_tz'), -1)
            self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix+'_Palm_Base_Tz'), -1)
        d = locked.reset_cpu()
        ids = [m.body(locked.manifest['hands'][side]['body']).id for side in ('right', 'left')]
        heights = d.xpos[ids, 2].copy()
        d.xfrc_applied[ids, 2] = -1000
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.xpos[ids, 2], heights, atol=1e-10, rtol=0)
        # No deleted actuator indices can leak into native control arrays.
        self.assertEqual(len(locked.action_ids), 48)
        self.assertEqual(len(set(locked.action_ids)), 48)

    def test_projection_checkpoints_are_not_silently_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, 'projection checkpoint'):
            check_tz_physics({'disable_wrist_tz': True})
        check_tz_physics({'disable_wrist_tz': True, 'wrist_tz_physics': 'joint_removed_v1'})
        check_tz_physics({})


if __name__ == '__main__':
    unittest.main()
