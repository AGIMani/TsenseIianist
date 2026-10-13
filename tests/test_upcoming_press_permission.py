"""The imminent-press experiment changes requested risk, never actual geometry."""
from copy import deepcopy
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.finger_clearance import (
    FingerClearanceGuard, canonical,
)


class UpcomingRequestRulesTests(unittest.TestCase):
    def setUp(self):
        # Deliberately simple CPU geometry fixture. It is not a training pose,
        # a hand mapping, or a claim about calibrated physical dimensions.
        self.guard = g = FingerClearanceGuard.__new__(FingerClearanceGuard)
        g.options = canonical(dict(schema=1, mode='full_finger_clearance_v1',
            active_command_mode='active_contact_v1', idle_finger_protection=False,
            active_press_permission=False))
        g.radii = torch.full((10, 4), .001)
        centers = torch.arange(88) * .03
        g.key_min_xy = torch.stack((centers - .01, torch.full((88,), -.1)), -1)
        g.key_max_xy = torch.stack((centers + .01, torch.full((88,), .1)), -1)
        g.key_top = torch.zeros(88)
        g.key_half_z = torch.full((88,), .01)
        g.key_bottom = torch.full((88,), -.02)
        g.lowest_bottom = -.02
        g.white_top = g.highest_top = 0.
        g.black_ids = torch.empty(0, dtype=torch.int64)
        g.pressed_center = torch.stack((centers, torch.zeros(88), torch.full((88,), -.02)), -1)
        g.pressed_normal = torch.tensor([0., 0., 1.]).expand(88, -1)
        self.ends = torch.zeros(1, 10, 4, 2, 3)
        self.ends[..., 2] = .02
        self.ends[:, :, -1, :, 2] = -.006
        # A depth of -7 mm is above the assigned key's fully pressed surface
        # (-10 mm) but below its unpressed surface (0 mm).
        g.fk = lambda physical, lift: (
            self.ends + lift[:, 0, None, None, None, None] * torch.tensor([0., 0., 1.]),
            torch.zeros(physical.shape[0], 10, 4, 3),
            torch.zeros(physical.shape[0], 10, 4, 3))
        self.physical = torch.zeros(1, 51)
        self.lift = torch.zeros(1, 2)
        self.assigned = torch.zeros(1, 10, 88, dtype=torch.bool)
        self.future = torch.zeros_like(self.assigned)
        self.future[:, 2, 0] = True
        self.seconds = torch.full((1, 10), .025)
        self.requested = g._evaluate(self.physical, self.lift, self.assigned)['finger_clearance_deficit_m']

    def call(self, *, window=.05):
        return self.guard.upcoming_request_metrics(self.physical, self.lift, self.assigned,
            self.future, self.seconds, window, self.requested)

    def test_only_imminent_assigned_finger_receives_lower_requested_plane(self):
        out = self.call()
        self.assertTrue(out['finger_upcoming_requested_permission_mask'][0, 2])
        self.assertEqual(out['finger_upcoming_requested_permission_mask'].sum().item(), 1)
        self.assertEqual(out['finger_reward_requested_deficit_m'][0, 2].item(), 0.)
        other = torch.arange(10) != 2
        torch.testing.assert_close(out['finger_reward_requested_deficit_m'][:, other], self.requested[:, other])
        self.assertGreater(self.requested[0, 2].item(), .008)

    def test_actual_idle_contact_and_under_key_deficits_are_never_exempted(self):
        out = self.call()
        actual = self.guard.evaluate(self.physical, self.lift, self.assigned)
        result = self.guard.reward_deficit(actual, out['finger_reward_requested_deficit_m'], self.assigned)
        torch.testing.assert_close(result, self.requested)
        # Retain a measured key-bottom breach even if requested geometry is clear.
        actual['finger_clearance_deficit_m'][0, 2] = .04
        result = self.guard.reward_deficit(actual, out['finger_reward_requested_deficit_m'], self.assigned)
        self.assertAlmostEqual(result[0, 2].item(), .04)

    def test_time_window_is_bounded_and_excludes_the_current_onset(self):
        for seconds, permitted in ((-.01, False), (0., False), (.025, True),
                                   (.05, True), (.0501, False), (.1, False),
                                   (float('nan'), False)):
            with self.subTest(seconds=seconds):
                self.seconds[:, 2] = seconds
                self.assertEqual(self.call()['finger_upcoming_requested_permission_mask'][0, 2].item(), permitted)

    def test_current_finger_and_wrong_future_key_do_not_receive_permission(self):
        self.assigned[:, 2, 3] = True
        torch.testing.assert_close(self.call()['finger_reward_requested_deficit_m'], self.requested)
        self.assigned.zero_()
        self.future.zero_(); self.future[:, 2, 1] = True
        out = self.call()
        self.assertFalse(out['finger_upcoming_requested_permission_mask'].any())
        torch.testing.assert_close(out['finger_reward_requested_deficit_m'], self.requested)

    def test_ambiguous_multiple_future_keys_are_not_exempted(self):
        self.future[:, 2, 1] = True
        self.assertFalse(self.call()['finger_upcoming_requested_permission_mask'].any())

    def test_other_key_overlap_retains_requested_surface_penalty(self):
        self.guard.key_min_xy[1] = self.guard.key_min_xy[0]
        self.guard.key_max_xy[1] = self.guard.key_max_xy[0]
        out = self.call()
        self.assertTrue(out['finger_upcoming_requested_permission_mask'][0, 2])
        torch.testing.assert_close(out['finger_reward_requested_deficit_m'], self.requested)

    def test_proximal_clearance_and_deep_terminal_request_are_retained(self):
        self.ends[:, 2, 0, :, 2] = 0.
        out = self.call()
        self.assertAlmostEqual(out['finger_reward_requested_deficit_m'][0, 2].item(), .003)
        self.ends[:, 2, 0, :, 2] = .02
        self.ends[:, 2, -1, :, 2] = -.025
        self.requested = self.guard._evaluate(self.physical, self.lift, self.assigned)['finger_clearance_deficit_m']
        out = self.call()
        self.assertTrue(out['finger_upcoming_requested_permission_mask'][0, 2])
        # Active-contact pad slack is 5 mm below the -10 mm full-travel plane.
        self.assertAlmostEqual(out['finger_reward_requested_deficit_m'][0, 2].item(), .011, places=6)

    def test_disabled_feature_is_exact_noop_and_inputs_are_not_mutated(self):
        before = [x.clone() for x in (self.physical, self.lift, self.assigned,
                                      self.future, self.seconds, self.requested)]
        old_options = deepcopy(self.guard.options)
        disabled = self.call(window=0)
        torch.testing.assert_close(disabled['finger_reward_requested_deficit_m'], self.requested, atol=0, rtol=0)
        self.assertFalse(disabled['finger_upcoming_requested_permission_mask'].any())
        self.call()
        for actual, expected in zip((self.physical, self.lift, self.assigned,
                                     self.future, self.seconds, self.requested), before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(self.guard.options, old_options)

    def test_projection_and_unsupported_mode_are_rejected(self):
        for field, value in (('idle_finger_protection', True), ('active_press_permission', True),
                             ('active_command_mode', 'strict_v1')):
            original = self.guard.options[field]
            self.guard.options[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'both physical controls disabled'):
                self.call()
            self.guard.options[field] = original
        for value in (-.1, True, float('nan'), float('inf')):
            with self.subTest(window=value), self.assertRaises(ValueError):
                self.call(window=value)


if __name__ == '__main__':
    unittest.main()
