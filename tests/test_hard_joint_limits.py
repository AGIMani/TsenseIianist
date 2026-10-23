"""Bounds, outward stop, normalized policy semantics and legacy opt-out."""
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
import torch
import warp as wp

from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical, rate_overrides
from tsensei_v3_2_pianist.mjlab_env.joint_limits import inward_bounds, bounded_positions, require_bounded, resolve_robot_limits
from tsensei_v3_2_pianist.mjlab_env.joint_limits_warp import HardJointLimits
from tsensei_v3_2_pianist.rl_torch.sac import Actor
from tsensei_v3_2_pianist.mjlab_env.action_mapping import policy_bounds, active_indices


class HardJointLimitsCPU(unittest.TestCase):
    def test_options_preserve_legacy_and_group_overrides(self):
        old = dict(schema=1, mode='target_clip_slew_v1', channels={'L_Palm_Base_Tx': {'max_velocity': 1.}})
        self.assertEqual(canonical(old), old)
        self.assertEqual(rate_overrides(old, SimpleNamespace()), old)
        new = rate_overrides(old, SimpleNamespace(hard_joint_limits=True))
        self.assertEqual(new['joint_state_limits'], 'hard_v1')
        self.assertEqual(rate_overrides(new, SimpleNamespace(hard_joint_limits=False)), old)
        self.assertEqual(rate_overrides(new, SimpleNamespace(wrist_tx_max_velocity=2.))['joint_state_limits'], 'hard_v1')
        self.assertEqual(rate_overrides(None, SimpleNamespace(hard_joint_limits=True))['channels'], {})

    def test_inward_rounding_and_export_fails_on_unbounded_states(self):
        low, high = inward_bounds([0., -.35, .05], [1.57, .35, 1.2])
        self.assertTrue((low.astype(float) >= [0., -.35, .05]).all())
        self.assertTrue((high.astype(float) <= [1.57, .35, 1.2]).all())
        q = np.array([[-.01, -.2, 1.3], [.2, .1, .2]])
        bounded, changed = bounded_positions(q, [0., -.35, .05], [1.57, .35, 1.2])
        self.assertEqual(changed.sum(), 2)
        self.assertEqual(bounded[0, 1], -.2)
        with self.assertRaises(ValueError): require_bounded(q, [0., -.35, .05], [1.57, .35, 1.2])
        require_bounded(bounded, [0., -.35, .05], [1.57, .35, 1.2])

    def test_kernel_limits_and_partial_world_projection(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
          <body><joint name="pip" range="0 1.57"/><geom size=".01"/></body>
          <body><joint name="swing" range="-.35 .35"/><geom size=".01"/></body>
          <body><joint name="key" range="0 .1"/><geom size=".01"/></body>
        </worldbody></mujoco>''')
        contract = resolve_robot_limits(model, key_ids=[2])
        limiter = HardJointLimits(contract, num_envs=2, device='cpu')
        q = np.array([[-.01, .4, -.1], [-.2, -.2, .2]], dtype=np.float32)
        v = np.array([[-1., -1., 2.], [-2., 1., 3.]], dtype=np.float32)
        data = SimpleNamespace(qpos=wp.array(q, dtype=wp.float32, device='cpu'),
            qvel=wp.array(v, dtype=wp.float32, device='cpu'),
            qacc_warmstart=wp.ones((2,3), dtype=wp.float32, device='cpu'))
        limiter.apply(data, wp.array([0], dtype=wp.int32, device='cpu'))
        out, speed = data.qpos.numpy(), data.qvel.numpy()
        self.assertEqual(out[0,0], 0.)
        self.assertLessEqual(float(out[0,1]), .35)
        self.assertEqual(speed[0,0], 0.)
        self.assertEqual(speed[0,1], -1.)  # Preserve inward motion.
        np.testing.assert_array_equal(out[1], q[1])
        np.testing.assert_array_equal(out[:,2], q[:,2])  # No piano modification.
        self.assertEqual(limiter.counts.numpy()[0].sum(), 2)

    def test_direct_policy_never_outputs_negative_flexion(self):
        # Both the full actor and the coupled actor use explicit policy bounds.
        for size in (47, 31):
            actor = Actor(4, size, [8], action_profile='direct_joint_v1')
            with torch.no_grad():
                actor.mean.weight.zero_()
                low, high = [torch.tensor(x) for x in policy_bounds('direct_joint_v1', size)]
                for extreme in (-100., 100.):
                    actor.mean.bias.fill_(extreme)
                    q = actor.deterministic(torch.zeros(2,4))
                    self.assertTrue(((q >= low) & (q <= high)).all())
                q, _ = actor.sample(torch.zeros(100,4))
                self.assertTrue(((q >= low) & (q <= high)).all())
            if size == 31:
                followers = {base+i for base in (0,25) for i in (6,7,10,11,14,15,18,19)}
                self.assertFalse(followers.intersection(active_indices(size)))


if __name__ == '__main__': unittest.main()
