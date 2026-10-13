"""Transient events, cycle normalization, CUDA capture and isolated reset."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import warp as wp

from tsensei_v3_2_pianist.mjlab_env.substep_rewards import SubstepRewardCollector


def fixture(device='cpu', decimation=10):
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    tracker = SubstepRewardCollector(
        num_envs=2, decimation=decimation, device=device, key_qadr=[0, 2],
        key_limits=[[0, .1], [0, .2]], action_ids=[0, 2], geom_pair_mask=mask)
    data = SimpleNamespace(
        qpos=wp.zeros((2, 3), dtype=wp.float32, device=device),
        actuator_force=wp.zeros((2, 3), dtype=wp.float32, device=device),
        actuator_velocity=wp.zeros((2, 3), dtype=wp.float32, device=device),
        naconmax=4, nacon=wp.array([0], dtype=wp.int32, device=device),
        contact=SimpleNamespace(
            geom=wp.array([[0, 1], [1, 0], [0, 2], [-1, 0]], dtype=wp.vec2i, device=device),
            dist=wp.array([-.001, -.002, -.1, -.1], dtype=wp.float32, device=device),
            worldid=wp.array([0, 0, 1, 1], dtype=wp.int32, device=device)))
    return tracker, SimpleNamespace(wp_data=data, wp_device=device)


class SubstepRewardTests(unittest.TestCase):
    def test_activation_uses_physical_threshold_without_normalization_roundtrip(self):
        for device in ('cpu', 'cuda:0') if torch.cuda.is_available() else ('cpu',):
            with self.subTest(device=device):
                _, sim = fixture(device)
                upper = np.float32(.0436332)
                threshold = np.float32(.00349)
                boundary = np.float32(upper-threshold)
                tracker = SubstepRewardCollector(
                    num_envs=2, decimation=10, device=device, key_qadr=[0],
                    key_limits=[[0, upper]], action_ids=[0],
                    geom_pair_mask=np.zeros((4, 4)), key_threshold=float(threshold))
                qpos = wp.to_torch(sim.wp_data.qpos)
                qpos[0, 0] = float(boundary)
                qpos[1, 0] = float(np.nextafter(boundary, np.float32(np.inf)))
                tracker.sample(sim, 0)
                wp.synchronize_device(device)
                expected = (qpos[:, 0].clamp(0, float(upper))-float(upper)).abs() <= float(threshold)
                torch.testing.assert_close(tracker.key_activation_history[:, 0, 0], expected, atol=0, rtol=0)
                self.assertEqual(expected.tolist(), [False, True])
                # A normalized round trip misclassifies the lower sample.
                reconstructed = ((1-tracker.key_state_history[:, 0, 0]).abs()*float(upper)
                                 <= float(threshold))
                self.assertEqual(reconstructed.tolist(), [True, True])

    def test_transient_key_is_seen_and_penalty_is_averaged(self):
        tracker, sim = fixture()
        qpos = wp.to_torch(sim.wp_data.qpos)
        for substep in range(10):
            qpos.zero_()
            if substep == 4:
                qpos[0, 0] = .05
                qpos[1, 2] = .25  # Travel saturation, not a larger false-press penalty.
            tracker.sample(sim, substep)
        wp.synchronize_device('cpu')
        self.assertEqual(tracker.key_state_history[:, -1].count_nonzero().item(), 0)
        torch.testing.assert_close(tracker.key_state_history[:, 4], torch.tensor([[.5, 0.], [0., 1.]]))
        # Same old per-key formula, but mean over physical samples. A middle
        # press cannot vanish because the key releases before the final sample.
        travel = tracker.key_state_history
        per_step = torch.where(travel > .001, -.01 - .14 * travel, 0).sum(-1)
        torch.testing.assert_close(per_step.mean(-1), torch.tensor([-.008, -.015]))
        self.assertEqual(per_step[:, -1].tolist(), [0., 0.])

    def test_power_is_mean_of_absolute_power_not_absolute_mean_or_tenfold(self):
        tracker, sim = fixture()
        force = wp.to_torch(sim.wp_data.actuator_force)
        velocity = wp.to_torch(sim.wp_data.actuator_velocity)
        force[:, 0] = 2
        force[:, 1] = 1000  # A non-hand actuator is excluded by action_ids.
        force[:, 2] = 3
        velocity[:, 1] = 1000
        velocity[:, 2] = -2
        for substep in range(10):
            velocity[:, 0] = 1 if substep < 5 else -1
            tracker.sample(sim, substep)
        wp.synchronize_device('cpu')
        torch.testing.assert_close(tracker.power_mean, torch.tensor([8., 8.]))
        torch.testing.assert_close(tracker.signed_power_mean, torch.tensor([[0., -6.], [0., -6.]]), atol=1e-6, rtol=0)

    def test_palm_union_exact_pairs_and_partial_reset(self):
        tracker, sim = fixture()
        wp.to_torch(sim.wp_data.nacon)[0] = 4
        tracker.sample(sim, 3)
        wp.to_torch(sim.wp_data.nacon)[0] = 0
        tracker.sample(sim, 9)
        wp.synchronize_device('cpu')
        self.assertEqual(tracker.palm_collision_any.tolist(), [True, False])
        tracker.key_state_history[1] = .5
        tracker.power_mean[1] = 4
        tracker.signed_power_mean[1] = -2
        tracker.clear(torch.tensor([0]))
        self.assertFalse(tracker.palm_collision_any.any())
        self.assertEqual(tracker.key_state_history[0].count_nonzero().item(), 0)
        self.assertTrue((tracker.key_state_history[1] == .5).all())
        self.assertEqual(tracker.power_mean.tolist(), [0., 4.])
        self.assertEqual(tracker.signed_power_mean[1].tolist(), [-2., -2.])
        tracker.clear()
        self.assertEqual(tracker.key_state_history.count_nonzero().item(), 0)
        self.assertEqual(tracker.power_mean.count_nonzero().item(), 0)

    def test_contact_margin_and_separate_query_source(self):
        tracker, sim = fixture()
        _, query = fixture()
        wp.to_torch(query.wp_data.nacon)[0] = 1
        wp.to_torch(query.wp_data.contact.dist)[0] = .001
        tracker.sample(sim, 0, contact_data=query.wp_data)
        self.assertFalse(tracker.palm_collision_any.any())
        tracker.collision_margin = .002
        tracker.sample(sim, 1, contact_data=query.wp_data)
        self.assertEqual(tracker.palm_collision_any.tolist(), [True, False])

    def test_final_forward_contact_union_does_not_resample_power_or_keys(self):
        tracker, sim = fixture()
        wp.to_torch(sim.wp_data.actuator_force)[:, 0] = 2
        wp.to_torch(sim.wp_data.actuator_velocity)[:, 0] = 3
        tracker.sample(sim, 9)
        keys, power = tracker.key_state_history.clone(), tracker.power_mean.clone()
        wp.to_torch(sim.wp_data.nacon)[0] = 1
        wp.to_torch(sim.wp_data.qpos)[:, 0] = .1
        tracker.sample_contacts(sim)
        self.assertEqual(tracker.palm_collision_any.tolist(), [True, False])
        torch.testing.assert_close(tracker.key_state_history, keys, atol=0, rtol=0)
        torch.testing.assert_close(tracker.power_mean, power, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph requires a GPU')
    def test_cuda_capture_matches_eager_and_uses_live_input(self):
        device = 'cuda:0'
        wp.init()
        stream = wp.get_stream(device)
        torch_stream = wp.stream_to_torch(stream)
        caller = torch.cuda.current_stream()
        torch_stream.wait_stream(caller)
        with torch.cuda.stream(torch_stream), wp.ScopedStream(stream):
            tracker, sim = fixture(device)
            wp.to_torch(sim.wp_data.qpos)[:, 0] = .05
            wp.to_torch(sim.wp_data.actuator_force)[:, 0] = 2
            wp.to_torch(sim.wp_data.actuator_velocity)[:, 0] = 3
            wp.to_torch(sim.wp_data.nacon)[0] = 2
            tracker.sample(sim, 0)  # Compile outside capture.
            tracker.clear()
            for substep in range(10):
                tracker.sample(sim, substep)
            expected = tuple(x.clone() for x in (tracker.key_state_history, tracker.power_mean,
                                                 tracker.signed_power_mean, tracker.palm_collision_any))
            tracker.clear()
            with wp.ScopedCapture() as capture:
                for substep in range(10):
                    tracker.sample(sim, substep)
            wp.capture_launch(capture.graph)
        caller.wait_stream(torch_stream)
        for actual, reference in zip((tracker.key_state_history, tracker.power_mean,
                                      tracker.signed_power_mean, tracker.palm_collision_any), expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        torch_stream.wait_stream(caller)
        with torch.cuda.stream(torch_stream), wp.ScopedStream(stream):
            tracker.clear()
            wp.to_torch(sim.wp_data.qpos)[:, 0] = .025
            wp.to_torch(sim.wp_data.nacon)[0] = 0
            wp.capture_launch(capture.graph)
        caller.wait_stream(torch_stream)
        torch.testing.assert_close(tracker.key_state_history[:, :, 0], torch.full((2, 10), .25, device=device))
        self.assertFalse(tracker.palm_collision_any.any())


if __name__ == '__main__':
    unittest.main()
