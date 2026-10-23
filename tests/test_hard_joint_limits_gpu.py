"""Scheduled real A3.2 physics: graph/eager/callback, reset and coupled targets."""
import unittest
import numpy as np
import torch
import warp as wp


@unittest.skipUnless(torch.cuda.is_available(), 'Requires Slurm CUDA allocation')
class HardJointLimitsGPU(unittest.TestCase):
    def test_real_coupled_scene_all_substeps_and_graph(self):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1',
            initial_buffer_time=0., preparation_mode='policy_v1', preparation_time=1.)
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent/'hard_limit_hands')
        s = PianoScene(scene, num_envs=2, physics_graph=True, action_profile='direct_joint_v1',
            disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'],
            control_limits=dict(schema=1, mode='target_clip_slew_v1', channels={}, joint_state_limits='hard_v1'),
            finger_coupling=dict(schema=1, mode='distal_60deg_v1', target_angle_deg=60., reduce_observation=False))
        rows = s.joint_limiter.contract['joints']
        ids = torch.tensor([r['qpos_address'] for r in rows], device=s.device)
        dofs = torch.tensor([r['dof_address'] for r in rows], device=s.device)
        low = torch.tensor([r['min'] for r in rows], dtype=torch.float64, device=s.device)
        high = torch.tensor([r['max'] for r in rows], dtype=torch.float64, device=s.device)
        def check(_=None):
            q = s.data.qpos[:,ids].double()
            self.assertTrue(((q >= low) & (q <= high)).all().item())
            ctrl = s.data.ctrl[:,s.action_ids]
            self.assertTrue(((ctrl >= s.hard_target_low[s.physical_ids]) &
                             (ctrl <= s.hard_target_high[s.physical_ids])).all().item())
        s.reset(); check()
        s.configure_reward_sampling(np.zeros((s.model.ngeom,s.model.ngeom), dtype=bool))
        neutral = s.policy_action(s.bundle.neutral_action).expand(2,-1).clone()
        # Regression path used by the failed multi-collector prefill: the
        # outer control graph is disabled, but MJLab still owns its cached
        # step/forward graphs.  Hard limits must refresh geometry through that
        # cached forward graph rather than allocating eager collision buffers.
        s.physics_graph = False
        s.step(neutral, substep_callback=check)
        check()
        s.physics_graph = True
        s.reset(); check()
        s.step(neutral, substep_callback=check)
        q_eager = s.data.qpos[:].clone()
        s.reset(); s.step(neutral); check()
        # Finger coupling intentionally keeps the outer control graph disabled;
        # MJLab's per-substep step/forward graphs remain active underneath.
        self.assertIsNone(s.control_graph)
        torch.testing.assert_close(q_eager, s.data.qpos[:], atol=1e-5, rtol=1e-5)
        # Force genuine violations before integration; the callback must never see them.
        with s.scope():
            s.data.qpos[:,ids] = low.float()-.001
            s.data.qvel[:,dofs] = -1.
        s.step(neutral, substep_callback=check)
        self.assertGreater(wp.to_torch(s.joint_limiter.counts).sum().item(), 0)
        other = s.data.qpos[1].clone()
        counts = wp.to_torch(s.joint_limiter.counts)[1].clone()
        s.reset([0]); check()
        torch.testing.assert_close(s.data.qpos[1], other, atol=0, rtol=0)
        torch.testing.assert_close(wp.to_torch(s.joint_limiter.counts)[1], counts, atol=0, rtol=0)
        for i in range(60):
            action = (s.action_low if i % 2 else s.action_high).expand(2,-1).clone()
            s.step(action, substep_callback=check if i % 3 == 0 else None)
            check()
