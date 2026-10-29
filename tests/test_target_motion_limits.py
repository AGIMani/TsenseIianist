"""CPU contracts and actual issued-target differences, including narrow stops."""
import argparse
from types import SimpleNamespace
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.action_mapping import _direct_contract
from tsensei_v3_2_pianist.mjlab_env.control_limits import (
    canonical, resolve, rate_overrides, add_rate_arguments, TargetLimiter)
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter


NAMES = [row['name'] for row in _direct_contract()['channels']]
PITCH = NAMES.index('R_Palm_Base_Pitch')
YAW = NAMES.index('L_Palm_Base_Yaw')


def config(**channels):
    return dict(schema=1, mode='target_clip_slew_v1', channels=channels)


def contract(options, dt=.05, reset=None, active=range(52)):
    low, high = [-3.] * 52 + [0.], [3.] * 52 + [1.]
    for name in ('R_Palm_Base_Pitch', 'L_Palm_Base_Yaw'):
        i = NAMES.index(name)
        low[i], high[i] = -.16, .16
    return resolve(options, action_names=NAMES, active_indices=active,
                   physical_low=low, physical_high=high,
                   reset_action=[0.] * 53 if reset is None else reset, control_dt=dt)


def make(dt=.05, worlds=8, reset=None, **kwargs):
    options = config(**{name: dict(max_velocity=3.1416, max_acceleration=15.708)
                        for name in ('R_Palm_Base_Pitch', 'L_Palm_Base_Yaw')})
    reset = torch.zeros(53) if reset is None else reset
    resolved = contract(options, dt, reset.tolist())
    return TargetMotionLimiter(resolved, num_envs=worlds, device='cpu', reset_action=reset, **kwargs)


class TargetMotionLimitsCPU(unittest.TestCase):
    def test_wrist_rate_limiter_keeps_hard_limits_without_goal_braking(self):
        # The final limiter follows the requested target subject only to the
        # current-step rate and endpoint feasibility bounds.
        options = config(R_Palm_Base_Tx=dict(max_velocity=.99, max_acceleration=1.99))
        resolved = contract(options)
        limiter = TargetMotionLimiter(resolved, num_envs=2, device='cpu',
            reset_action=torch.zeros(53))
        first = TargetLimiter(resolved, device='cpu')
        previous = torch.zeros(2, 53)
        history=[]; velocities=[]
        for step in range(140):
            request = torch.zeros_like(previous)
            request[:, 20] = torch.tensor([.48, -.48]) if step < 60 else 0.
            clipped = first.apply(request, previous)[0]
            issued, _ = limiter.apply(clipped)
            history.append(issued[:, 20].double().clone())
            velocities.append(limiter.velocity.clone())
            previous = issued
        q=torch.stack(history); v=torch.diff(torch.cat([torch.zeros(1, 2),q]),dim=0)/.05
        a=torch.diff(torch.cat([torch.zeros(1, 2),v]),dim=0)/.05
        self.assertLessEqual(float(v.abs().max()), .99+1e-12)
        self.assertLessEqual(float(a.abs().max()), 1.99+1e-12)
        # The removed goal-braking branch no longer hides the unslewed request
        # behind a one-step target; endpoint/rate bounds still hold.
        self.assertGreater(float(q.abs().max()), .48)
        self.assertEqual(limiter.contract['controller'], 'wrist_target_rate_limits')

    def test_legacy_resolved_contract_and_target_math_are_unchanged(self):
        old = config(R_Palm_Base_Pitch=dict(max_velocity=4.))
        self.assertEqual(canonical(old), old)
        self.assertEqual(rate_overrides(old, SimpleNamespace()), old)
        resolved = contract(old)
        self.assertEqual(resolved['channels']['R_Palm_Base_Pitch'], dict(index=PITCH,
            min=-.16, max=.16, max_velocity=4., position_unit='rad', velocity_unit='rad/s',
            original_min=-.16, original_max=.16, reset_target=0.))
        self.assertEqual(resolved['application'],
            'original policy mapping, static clip, previous applied target slew, zero-order hold')
        previous = torch.zeros(2, 53)
        requested = torch.ones_like(previous)
        actual = TargetLimiter(resolved, device='cpu').apply(requested, previous)[0]
        self.assertTrue(torch.equal(actual[:, PITCH], torch.full((2,), .16)))

    def test_cli_overrides_persist_acceleration_without_activating_removed_yaw(self):
        parser = argparse.ArgumentParser()
        add_rate_arguments(parser)
        args = parser.parse_args(['--wrist-pitch-max-velocity', '3.1416',
                                  '--wrist-pitch-max-acceleration', '15.708'])
        args.disable_wrist_tz = args.disable_wrist_yaw = True
        options = rate_overrides(None, args)
        self.assertEqual(set(options['channels']), {'R_Palm_Base_Pitch', 'L_Palm_Base_Pitch'})
        for row in options['channels'].values():
            self.assertEqual(row, dict(max_velocity=3.1416, max_acceleration=15.708))
        disabled = rate_overrides(options, SimpleNamespace(wrist_pitch_max_acceleration=0.))
        self.assertTrue(all('max_acceleration' not in row for row in disabled['channels'].values()))
        with self.assertRaises(ValueError):
            rate_overrides(None, SimpleNamespace(wrist_yaw_max_velocity=3.1416,
                wrist_yaw_max_acceleration=15.708, disable_wrist_yaw=True))

    def test_invalid_acceleration_contracts_fail(self):
        invalid = [config(R_Palm_Base_Pitch=dict(max_acceleration=15.708)),
                   config(R_Palm_Base_Pitch=dict(max_velocity=3., max_acceleration=0.)),
                   config(R_Palm_Base_Pitch=dict(max_velocity=3., max_acceleration=float('inf'))),
                   config(R_Palm_Base_Tz=dict(max_velocity=3., max_acceleration=15.)),
                   config(R_Thumb_MCP=dict(max_velocity=3., max_acceleration=15.))]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical(value)
        with self.assertRaises(ValueError):
            make(dt=1e-6)

    def assert_sequence(self, limiter, requests):
        previous = limiter.previous.clone()
        velocity = limiter.velocity.clone()
        peak_v = peak_a = 0.
        for requested in requests:
            applied, changed = limiter.apply(requested)
            positions = applied[:, limiter.indices].double()
            actual_velocity = (positions - previous) / limiter.dt
            actual_acceleration = (actual_velocity - velocity) / limiter.dt
            peak_v = max(peak_v, actual_velocity.abs().max().item())
            peak_a = max(peak_a, actual_acceleration.abs().max().item())
            # Double-precision arithmetic comparison tolerance only. Position
            # commands themselves are float32 and limits are exact user values.
            self.assertTrue((actual_velocity.abs() <= 3.1416 + 1e-12).all(), peak_v)
            self.assertTrue((actual_acceleration.abs() <= 15.708 + 1e-12).all(), peak_a)
            self.assertTrue((positions >= limiter.low).all())
            self.assertTrue((positions <= limiter.high).all())
            self.assertTrue(torch.equal(changed[:, limiter.indices],
                                       applied[:, limiter.indices] != requested[:, limiter.indices]))
            self.assertTrue(torch.equal(applied[:, 0], requested[:, 0]))
            self.assertTrue(torch.equal(applied[:, 20:23], requested[:, 20:23]))
            self.assertTrue(torch.equal(applied[:, -1], requested[:, -1]))
            previous, velocity = positions.clone(), actual_velocity.clone()
        return peak_v, peak_a

    def test_random_reversals_keep_the_same_hard_limits(self):
        generator = torch.Generator().manual_seed(273)
        for dt in (.01, .02, .05, .1):
            with self.subTest(dt=dt):
                limiter = make(dt=dt)
                requests = [torch.rand((8, 53), generator=generator)*4-2 for _ in range(1000)]
                self.assert_sequence(limiter, requests)

    def test_dense_alternation_random_requests_and_endpoint_holds_at_multiple_dt(self):
        generator = torch.Generator().manual_seed(72719)
        for dt in (.01, .025, .05, .1):
            with self.subTest(dt=dt):
                limiter = make(dt=dt)
                requests = []
                for step in range(1600):
                    request = torch.rand((8, 53), generator=generator) * 2 - 1
                    if step < 150:
                        request[:, limiter.indices] = .16
                    elif step < 300:
                        request[:, limiter.indices] = -.16
                    elif step < 500:
                        request[:, limiter.indices] = 100. * (-1 if step % 2 else 1)
                    requests.append(request)
                self.assert_sequence(limiter, requests)

    def test_endpoint_has_zero_final_velocity_without_a_position_clip_jump(self):
        limiter = make(worlds=1)
        request = torch.zeros(1, 53)
        request[:, limiter.indices] = 10.
        self.assert_sequence(limiter, [request] * 30)
        torch.testing.assert_close(limiter.previous[0], limiter.high, rtol=0, atol=0)
        torch.testing.assert_close(limiter.velocity, torch.zeros_like(limiter.velocity), rtol=0, atol=0)
        request[:, limiter.indices] = -10.
        self.assert_sequence(limiter, [request] * 30)
        torch.testing.assert_close(limiter.previous[0], limiter.low, rtol=0, atol=0)
        torch.testing.assert_close(limiter.velocity, torch.zeros_like(limiter.velocity), rtol=0, atol=0)

    def test_hard_position_intersection_and_partial_reset(self):
        low, high = torch.full((53,), -3.), torch.full((53,), 3.)
        low[[PITCH, YAW]], high[[PITCH, YAW]] = -.12, .11
        reset = torch.zeros(53)
        reset[PITCH] = .03
        limiter = make(worlds=2, reset=reset, physical_low=low, physical_high=high)
        request = torch.ones(2, 53)
        self.assert_sequence(limiter, [request] * 2)
        keep = limiter.previous[1].clone(), limiter.velocity[1].clone()
        limiter.reset(torch.tensor([0]))
        torch.testing.assert_close(limiter.previous[0], reset[limiter.indices].double(), rtol=0, atol=0)
        torch.testing.assert_close(limiter.velocity[0], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertTrue(torch.equal(limiter.previous[1], keep[0]))
        self.assertTrue(torch.equal(limiter.velocity[1], keep[1]))
        self.assert_sequence(limiter, [request] * 52)
        self.assertTrue((limiter.previous <= .11).all())
        self.assertTrue((limiter.previous >= -.12).all())


if __name__ == '__main__':
    unittest.main()
