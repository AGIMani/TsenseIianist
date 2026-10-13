"""CPU contracts for the bounded physical actuator-target motion penalty."""
from copy import deepcopy
import unittest

import torch

from tests.test_paper_rewards import scene_fixture
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager


TERM = 'joint_motion_limit_penalty'


class JointMotionLimitRewardTest(unittest.TestCase):
    def config(self, **params):
        cfg = template()
        for name, term in cfg['terms'].items():
            term['enabled'] = name == TERM
        cfg['terms'][TERM]['params'].update(params)
        return cfg

    def fixture(self, count=1, dt=.05, cfg=None, neutral=0.):
        cfg = self.config() if cfg is None else cfg
        scene = scene_fixture()
        scene['control_dt'] = dt
        scene['neutral_action'][:50] = [neutral] * 50
        manager = RewardManager(cfg, scene, {}, num_envs=count, device='cpu')
        action = torch.tensor(scene['neutral_action']).expand(count, -1).clone()
        return manager, action

    def evaluate(self, manager, action, snapshot=None):
        count = len(action)
        return manager({}, action, {} if snapshot is None else snapshot,
                       torch.zeros(count, 89), torch.zeros(count, 10, 88))[TERM]

    def assert_values(self, actual, expected):
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_velocity_boundaries_and_both_signs(self):
        # Large acceleration limit isolates velocity. At +10% the score is
        # half the budget; +20% saturates, and signs must not change the score.
        cfg = self.config(finger_acceleration_limit=1e6)
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                manager, action = self.fixture(6, cfg=cfg)
                action[:, 0] = sign * torch.tensor([0., 5., 10., 11., 12., 20.]) * .05
                self.assert_values(self.evaluate(manager, action), [0., 0., 0., -2.5, -5., -5.])

    def test_acceleration_boundaries_and_both_signs(self):
        cfg = self.config(finger_velocity_limit=100.)
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                manager, action = self.fixture(6, cfg=cfg)
                action[:, 0] = sign * torch.tensor([0., 100., 200., 220., 240., 400.]) * .05**2
                self.assert_values(self.evaluate(manager, action), [0., 0., 0., -2.5, -5., -5.])

    def test_reversing_below_speed_limit_still_penalizes_acceleration(self):
        manager, action = self.fixture()
        action[:, 0] = .3  # +6 rad/s; +120 rad/s^2 from rest.
        self.assert_values(self.evaluate(manager, action), [0.])
        action[:, 0] = 0.  # -6 rad/s; -240 rad/s^2 reverses motion.
        self.assert_values(self.evaluate(manager, action), [-5.])

    def test_shared_maximum_does_not_sum_joints_groups_or_derivatives(self):
        cfg = self.config(wrist_velocity_limit=10., wrist_acceleration_limit=200.)
        manager, action = self.fixture(3, cfg=cfg)
        ids = torch.cat((manager.groups['finger'], manager.groups['wrist']))
        action[0, 0] = .55  # Velocity and acceleration both exceed by 10%.
        action[1, ids] = .55  # The same violation on all 44 joints.
        action[2, ids] = .525
        action[2, 0] = .55  # Different violations still take only the largest.
        self.assert_values(self.evaluate(manager, action), [-2.5, -2.5, -2.5])

    def test_every_finger_including_coupled_pip_dip_followers_is_scored(self):
        names = scene_fixture()['action_names']
        finger_names = [name for name in names if name != 'sustain' and '_Wrist_' not in name]
        self.assertEqual(len(finger_names), 40)
        self.assertIn('R_Index_PIP', finger_names)
        self.assertIn('L_Pinky_DIP', finger_names)
        for name in finger_names:
            with self.subTest(joint=name):
                manager, action = self.fixture()
                action[:, names.index(name)] = .55
                self.assert_values(self.evaluate(manager, action), [-2.5])

    def test_wrist_thresholds_are_separate_and_cover_both_rotation_axes(self):
        cfg = self.config(wrist_velocity_limit=3., wrist_acceleration_limit=15.)
        names = scene_fixture()['action_names']
        for side in ('R', 'L'):
            for axis in ('Pitch', 'Yaw'):
                with self.subTest(side=side, axis=axis):
                    manager, action = self.fixture(2, cfg=cfg)
                    wrist = names.index(f'{side}_Wrist_{axis}')
                    # Identical physical speeds: wrist is 10% over its own
                    # limit while a finger remains below the finger limit.
                    delta = 3.3 * .05
                    action[0, wrist] = delta
                    action[1, 0] = delta
                    self.evaluate(manager, action)
                    action[0, wrist] = 2 * delta
                    action[1, 0] = 2 * delta
                    self.assert_values(self.evaluate(manager, action), [-2.5, 0.])
                    manager, action = self.fixture(2, cfg=cfg)
                    # 16.5 rad/s^2: 10% above wrist threshold, below finger.
                    action[0, wrist] = 16.5 * .05**2
                    action[1, 0] = 16.5 * .05**2
                    self.assert_values(self.evaluate(manager, action), [-2.5, 0.])

    def test_null_wrist_limits_exclude_wrists(self):
        manager, action = self.fixture()
        action[:, manager.groups['wrist']] = 2.
        self.assert_values(self.evaluate(manager, action), [0.])

    def test_translations_and_pedal_do_not_consume_joint_budget(self):
        cfg = self.config(wrist_velocity_limit=3., wrist_acceleration_limit=15.)
        manager, action = self.fixture(cfg=cfg)
        self.assertEqual(len(manager.groups['forearm']), 6)
        action[:, manager.groups['forearm']] = 2.
        action[:, -1] = 1.
        self.assert_values(self.evaluate(manager, action), [0.])

    def test_first_frame_and_partial_reset_use_resting_neutral_history(self):
        manager, action = self.fixture(2, neutral=.2)
        self.assert_values(self.evaluate(manager, action), [0., 0.])
        action[:, 0] += .55
        self.assert_values(self.evaluate(manager, action), [-2.5, -2.5])
        self.assert_values(self.evaluate(manager, action), [-2.5, -2.5])  # Abrupt stop.
        self.assert_values(self.evaluate(manager, action), [0., 0.])
        manager.reset(torch.tensor([0]))
        self.assert_values(self.evaluate(manager, action), [-2.5, 0.])
        torch.testing.assert_close(manager.previous2[0], manager.neutral)
        torch.testing.assert_close(manager.previous2[1], action[1, :50])

    def test_action_units_do_not_change_physical_target_penalty(self):
        for units in (None, 'physical', 'normalized'):
            with self.subTest(units=units):
                cfg = self.config()
                cfg['action_units'] = units
                manager, action = self.fixture(cfg=cfg)
                action[:, 0] = .55
                self.assert_values(self.evaluate(manager, action), [-2.5])

    def test_uses_supplied_final_command_not_measured_state_or_raw_policy(self):
        manager, action = self.fixture()
        snapshot = dict(qvel=torch.full((1, 50), 1e6), qacc=torch.full((1, 50), -1e6),
                        raw_policy_action=torch.ones(1, 31))
        self.assert_values(self.evaluate(manager, action, snapshot), [0.])
        action[:, 0] = .55
        self.assert_values(self.evaluate(manager, action, snapshot), [-2.5])

    def test_control_dt_changes_derivatives(self):
        for dt, expected in ((.1, 0.), (.05, -2.5), (.025, -5.)):
            with self.subTest(dt=dt):
                manager, action = self.fixture(dt=dt)
                action[:, 0] = .55
                self.assert_values(self.evaluate(manager, action), [expected])

    def test_final_cap_is_not_scaled_by_dt_or_reference_dt(self):
        for dt in (.025, .05, .1):
            for reference in (None, .05):
                with self.subTest(dt=dt, reference_dt=reference):
                    cfg = self.config()
                    cfg['scale_by_dt'] = True
                    if reference is not None:
                        cfg['reference_dt'] = reference
                    manager, action = self.fixture(dt=dt, cfg=cfg)
                    action[:, 0] = 2.
                    self.assert_values(self.evaluate(manager, action), [-5.])
                    self.assert_values(manager.raw_terms[TERM], [-5.])

    def test_weight_maximum_and_ramp_are_explicit(self):
        cfg = self.config(full_penalty_excess=.4, maximum_penalty_per_step=4.)
        cfg['terms'][TERM]['weight'] = .5
        manager, action = self.fixture(2, cfg=cfg)
        action[:, 0] = torch.tensor([.55, 2.])
        self.assert_values(self.evaluate(manager, action), [-.5, -2.])
        self.assert_values(manager.raw_terms[TERM], [-1., -4.])
        self.assert_values(manager.weighted_terms[TERM], [-.5, -2.])
        for field in ('weight', 'maximum_penalty_per_step'):
            cfg = self.config()
            if field == 'weight':
                cfg['terms'][TERM][field] = 0.
            else:
                cfg['terms'][TERM]['params'][field] = 0.
            manager, action = self.fixture(cfg=cfg)
            action[:, 0] = 2.
            self.assert_values(self.evaluate(manager, action), [0.])

    def test_reordering_terms_does_not_change_fingering_assignments(self):
        cfg = self.config()
        cfg['terms']['fingering_reward']['enabled'] = True
        cfg['terms']['fingering_reward']['params']['mode'] = 'paper'
        cfg['terms'] = {TERM: cfg['terms'][TERM], **cfg['terms']}
        manager, action = self.fixture(cfg=cfg)
        fingers = torch.zeros(1, 10, 88)
        fingers[:, 0, 0] = 1.
        values = manager({}, action, dict(fingertips=torch.zeros(1, 10, 3),
                         key_targets=torch.zeros(1, 88, 3)), torch.zeros(1, 89), fingers)
        self.assert_values(values['fingering_reward'], [1.])
        self.assert_values(values[TERM], [0.])

    def test_invalid_limits_weights_and_contracts_are_rejected(self):
        positive_fields = ('finger_velocity_limit', 'finger_acceleration_limit',
                           'wrist_velocity_limit', 'wrist_acceleration_limit', 'full_penalty_excess')
        for field in positive_fields:
            for value in (0., -1., True, '10', float('nan'), float('inf')):
                cfg = self.config(wrist_velocity_limit=3., wrist_acceleration_limit=15.)
                cfg['terms'][TERM]['params'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate(cfg)
        for field, values in (('weight', (-1., 1.01, True, float('nan'))),
                              ('maximum_penalty_per_step', (-1., 5.01, True, float('nan')))):
            for value in values:
                cfg = self.config()
                target = cfg['terms'][TERM] if field == 'weight' else cfg['terms'][TERM]['params']
                target[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate(cfg)
        for field in ('wrist_velocity_limit', 'wrist_acceleration_limit'):
            cfg = self.config()
            cfg['terms'][TERM]['params'][field] = 1.
            with self.subTest(single_wrist_limit=field), self.assertRaises(ValueError):
                validate(cfg)
        for change in ('missing', 'unknown'):
            cfg = self.config()
            if change == 'missing':
                del cfg['terms'][TERM]['params']['finger_velocity_limit']
            else:
                cfg['terms'][TERM]['params']['sum_joints'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(cfg)

    def test_missing_motion_term_is_rejected_without_migration(self):
        cfg = template()
        del cfg['terms'][TERM]
        before = deepcopy(cfg)
        with self.assertRaisesRegex(ValueError, 'Reward terms must match'):
            validate(cfg)
        self.assertEqual(cfg, before)



if __name__ == '__main__':
    unittest.main()
