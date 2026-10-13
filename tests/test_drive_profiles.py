"""Compiled Composer contracts for optional reflected platform inertia."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.drive_settings import load


class DriveProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {}
        for name in ('default', 'legacy_v1', 'shadow_v1'):
            with patch.dict(os.environ, {'TSENSEI_V32_COLLISION_PROFILE': 'legacy_mesh_v1'}):
                if name == 'default':
                    os.environ.pop('TSENSEI_V32_DRIVE_PROFILE', None)
                else:
                    os.environ['TSENSEI_V32_DRIVE_PROFILE'] = name
                env = suite.load('TwinkleTwinkleRousseau', task_kwargs={
                    'control_timestep': .05, 'physics_timestep': .005,
                    'initial_buffer_time': 0., 'gravity_compensation': True})
                env.reset()
                cls.envs[name] = env

    @classmethod
    def tearDownClass(cls):
        for env in cls.envs.values():
            env.close()

    def test_default_is_legacy_and_preserves_full_physics_arrays(self):
        default = self.envs['default'].physics.model.ptr
        legacy = self.envs['legacy_v1'].physics.model.ptr
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'dof_armature',
                      'dof_damping', 'dof_frictionloss', 'jnt_range',
                      'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm'):
            np.testing.assert_array_equal(getattr(default, field), getattr(legacy, field))
        self.assertEqual(default.nbody, legacy.nbody)

    def test_shadow_preserves_link_mass_pitch_inertia_ranges_and_pose(self):
        legacy = self.envs['legacy_v1'].physics.model.ptr
        shadow = self.envs['shadow_v1'].physics.model.ptr
        self.assertEqual(legacy.nbody, shadow.nbody)
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'jnt_range', 'actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(legacy, field), getattr(shadow, field))
        np.testing.assert_array_equal(self.envs['legacy_v1'].physics.data.qpos,
                                      self.envs['shadow_v1'].physics.data.qpos)
        np.testing.assert_array_equal(self.envs['legacy_v1'].physics.data.ctrl,
                                      self.envs['shadow_v1'].physics.data.ctrl)
        for side in ('right', 'left'):
            joint = shadow.joint(side+'_/forearm_pitch').id
            dof = shadow.jnt_dofadr[joint]
            self.assertEqual(shadow.dof_armature[dof], legacy.dof_armature[dof])
            self.assertAlmostEqual(shadow.dof_M0[dof], legacy.dof_M0[dof], places=14)

    def test_only_slides_gain_reflected_inertia_and_critical_damping(self):
        legacy = self.envs['legacy_v1'].physics.model.ptr
        shadow = self.envs['shadow_v1'].physics.model.ptr
        changed = []
        for side in ('right', 'left'):
            for axis in ('tx', 'ty'):
                joint = shadow.joint(f'{side}_/forearm_{axis}').id
                dof = int(shadow.jnt_dofadr[joint])
                changed.append(dof)
                self.assertEqual(shadow.dof_armature[dof], .02)
                self.assertAlmostEqual(shadow.dof_M0[dof]-legacy.dof_M0[dof], .02, places=12)
                self.assertAlmostEqual(shadow.dof_damping[dof], 2*np.sqrt(300*shadow.dof_M0[dof]), places=11)
                self.assertAlmostEqual(shadow.dof_M0[dof], .972, places=12)
        difference = shadow.dof_armature-legacy.dof_armature
        np.testing.assert_array_equal(np.flatnonzero(difference), sorted(changed))

    def test_finger_friction_force_caps_and_pitch_reference(self):
        model = self.envs['shadow_v1'].physics.model.ptr
        count = 0
        for a in range(model.nu):
            joint = int(model.actuator_trnid[a, 0])
            name = model.joint(joint).name
            dof = int(model.jnt_dofadr[joint])
            if '_hand_' in name:
                count += 1
                self.assertEqual(model.dof_frictionloss[dof], .01)
                self.assertEqual(model.dof_damping[dof], .05)
                self.assertEqual(model.dof_armature[dof], .0002)
                self.assertEqual(model.actuator_gainprm[a, 0], 1.)
                np.testing.assert_array_equal(model.actuator_forcerange[a], [-.6, .6])
            elif name.endswith('/forearm_pitch'):
                self.assertEqual(model.actuator_gainprm[a, 0], 8.)
                self.assertEqual(model.dof_damping[dof], .5)
                np.testing.assert_array_equal(model.actuator_forcerange[a], [-5., 5.])
            elif name.endswith(('/forearm_tx', '/forearm_ty')):
                np.testing.assert_array_equal(model.actuator_forcerange[a], [-15.32, 15.32])
        self.assertEqual(count, 40)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            load('unknown_profile')


if __name__ == '__main__':
    unittest.main()
