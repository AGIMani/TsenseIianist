"""CPU contract and exact MuJoCo FK tests for optional follower servos."""
from copy import deepcopy
import json
import math
import unittest

import mujoco
import numpy as np
import torch

from tsensei_v3_2_pianist.mjlab_env.action_mapping import active_indices, policy_bounds, physical_bounds
from tsensei_v3_2_pianist.mjlab_env.finger_coupling import (
    canonical, FingerCoupler, observation_contract, FOLLOWER_INDICES, FINGER_NAMES, RANGE_MODE)

OPTIONS = dict(schema=1, mode='distal_60deg_v1')
RANGE_OPTIONS = dict(schema=1, mode=RANGE_MODE)


class CouplingOptionsTests(unittest.TestCase):
    def test_disabled_and_validation(self):
        self.assertIsNone(canonical(None))
        self.assertEqual(canonical(OPTIONS)['target_angle_deg'], 60.)
        for values in ({}, dict(OPTIONS, unknown=True), dict(OPTIONS, target_angle_deg=90),
                       dict(OPTIONS, target_angle_deg=float('nan')), dict(OPTIONS, reduce_observation=1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                canonical(values)

    def test_independent_ranges_validation_and_roundtrip(self):
        options = canonical(RANGE_OPTIONS)
        self.assertEqual(options['angle_ranges_deg'], {name: [50., 80.] for name in FINGER_NAMES})
        custom = deepcopy(options)
        custom['angle_ranges_deg']['right_index'] = [55, 70]
        custom['angle_ranges_deg']['left_pinky'] = [65, 65]
        self.assertEqual(canonical(json.loads(json.dumps(custom))), custom)
        self.assertEqual(options['angle_ranges_deg']['right_index'], [50, 80])
        for pair in ([80, 50], [0, 80], [50, 90], [True, 80], [50, float('nan')], [60], '50,80'):
            broken = deepcopy(options)
            broken['angle_ranges_deg']['left_mid'] = pair
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                canonical(broken)
        for ranges in ({'right_index': [50, 80]}, dict(options['angle_ranges_deg'], right_thumb=[50, 80])):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                canonical(dict(RANGE_OPTIONS, angle_ranges_deg=ranges))
        with self.assertRaises(ValueError):
            canonical(dict(options, target_angle_deg=60))
        with self.assertRaises(ValueError):
            canonical(dict(OPTIONS, angle_ranges_deg=options['angle_ranges_deg']))

    def test_action_layouts_keep_all_thumb_mcp_swing_and_pedal_channels(self):
        for size in (31, 33, 35):
            ids = active_indices(size)
            self.assertEqual(len(ids), size)
            self.assertEqual(ids, [i for i in active_indices(size+16) if i not in FOLLOWER_INDICES])
            for i in (0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 25, 26, 27, 28, 50):
                self.assertIn(i, ids)
            for profile in ('legacy', 'mixed_v1', 'direct_joint_v1'):
                low, high = policy_bounds(profile, size)
                full_low, full_high = policy_bounds(profile, 51)
                self.assertEqual(low, [full_low[i] for i in ids])
                self.assertEqual(high, [full_high[i] for i in ids])


class CouplingGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
        recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1', initial_buffer_time=0.,
                             preparation_time=1., preparation_mode='policy_v1')
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent/'coupling_hand_only')
        cls.bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'])
        low, high = physical_bounds('direct_joint_v1', cls.bundle.manifest['action_names'], cls.bundle.low, cls.bundle.high)
        cls.coupler = FingerCoupler(cls.bundle, device='cpu', options=OPTIONS, physical_low=low, physical_high=high)
        cls.manifest = json.loads((task/'task.json').read_text())
        cls.low = np.asarray(low); cls.high = np.asarray(high)

    def states(self, count=12):
        b = self.bundle; m = b.model
        rng = np.random.default_rng(31)
        physical = self.low+(self.high-self.low)*rng.random((count, 51))
        qpos = []; matrices = []
        for world in range(count):
            data = b.reset_cpu(); data.qpos[:] = b.origin_qpos
            # Exercise actual follower lag, all wrist axes and four-finger swings.
            actual = self.low+(self.high-self.low)*rng.random(51)
            for actuator, full in zip(b.action_ids, b.active_action_indices):
                joint = int(m.actuator_trnid[actuator, 0])
                data.qpos[m.jnt_qposadr[joint]] = actual[full]
            mujoco.mj_forward(m, data)
            qpos.append(data.qpos.copy()); matrices.append(data.xmat.copy())
        return tuple(torch.tensor(np.asarray(x), dtype=torch.float32) for x in (physical, qpos, matrices))

    def test_predicted_angle_matches_native_mirrored_nonparallel_axes(self):
        self.assert_native_fk(self.coupler)

    def test_range_angles_match_native_fk_and_distance_outside_each_interval(self):
        options = dict(RANGE_OPTIONS, angle_ranges_deg={name: [45+i, 65+i] for i, name in enumerate(FINGER_NAMES)})
        g = self.range_coupler(options)
        self.assert_native_fk(g)
        _, _, matrices = self.states()
        info = g.actual_diagnostics(matrices)
        angle = info['coupling_actual_angle_deg'].numpy()
        bounds = np.array(list(options['angle_ranges_deg'].values()))
        expected = np.maximum(bounds[:, 0]-angle, 0)+np.maximum(angle-bounds[:, 1], 0)
        np.testing.assert_allclose(info['coupling_actual_error_deg'].numpy(), expected, atol=1e-5)
        samples = torch.tensor([[50]*8, [65]*8, [80]*8, [45]*8, [85]*8], dtype=torch.float32)
        default = self.range_coupler()
        torch.testing.assert_close(default._angle_error(samples),
            torch.tensor([[0]*8, [0]*8, [0]*8, [5]*8, [5]*8], dtype=torch.float32))

    def range_coupler(self, options=RANGE_OPTIONS):
        return FingerCoupler(self.bundle, device='cpu', options=options,
                             physical_low=self.low, physical_high=self.high)

    def assert_native_fk(self, g):
        b = self.bundle; m = b.model
        physical, qpos, matrices = self.states()
        originals = [x.clone() for x in (physical, qpos, matrices)]
        output, info = g.apply(physical, actual_qpos=qpos, body_xmat=matrices)
        expected = []
        for world in range(len(qpos)):
            data = b.reset_cpu(); data.qpos[:] = qpos[world].numpy()
            # Preserve actual wrists/swing, only commanded MCP and followers change.
            for group in g.groups:
                for address, full in zip(group['qpos_addresses'], group['full_indices']):
                    data.qpos[address] = output[world, full]
            mujoco.mj_forward(m, data)
            row = []
            for group in g.groups:
                direction = data.xmat[group['body_ids'][2]].reshape(3, 3)@np.asarray(group['distal_direction_local'])
                row.append(math.degrees(math.asin(np.clip(-direction@g.normal.numpy(), -1, 1))))
            expected.append(row)
        np.testing.assert_allclose(info['coupling_target_angle_deg'].numpy(), expected, atol=6e-4, rtol=0)
        for before, after in zip(originals, (physical, qpos, matrices)):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        unchanged = [i for i in range(51) if i not in FOLLOWER_INDICES]
        torch.testing.assert_close(output[:, unchanged], physical[:, unchanged], atol=0, rtol=0)
        followers = output[:, g.indices[:, 1:]]
        # Coupled followers must consume the same narrowed command bounds as
        # independent policy channels, without narrowing the physical joints.
        torch.testing.assert_close(g.low, torch.zeros((8, 2)))
        torch.testing.assert_close(g.high, torch.tensor([[.9, .8]]).expand(8, -1))
        for group in g.groups:
            for name in group['joint_names'][1:]:
                np.testing.assert_allclose(m.joint(name).range, [0., 1.57])
        self.assertTrue(torch.all(followers >= g.low))
        self.assertTrue(torch.all(followers <= g.high))
        torch.testing.assert_close(info['coupling_infeasible'], info['coupling_target_error_deg'] > g.TOLERANCE_DEG)
        if g.range_mode:
            expected = np.asarray(expected)
            bounds = g.angle_bounds.numpy()
            error = np.maximum(bounds[:, 0]-expected, 0)+np.maximum(expected-bounds[:, 1], 0)
            np.testing.assert_allclose(info['coupling_target_error_deg'].numpy(), error, atol=6e-4)

    def test_changing_one_finger_range_leaves_seven_other_solutions_unchanged(self):
        g = self.range_coupler()
        physical, qpos, matrices = self.states()
        baseline, _ = g.apply(physical, actual_qpos=qpos, body_xmat=matrices)
        for i, name in enumerate(FINGER_NAMES):
            options = canonical(RANGE_OPTIONS)
            options['angle_ranges_deg'][name] = [70., 75.]
            changed, _ = self.range_coupler(options).apply(physical, actual_qpos=qpos, body_xmat=matrices)
            own = g.groups[i]['full_indices'][1:]
            other = [k for k in range(51) if k not in own]
            with self.subTest(finger=name):
                torch.testing.assert_close(changed[:, other], baseline[:, other], atol=0, rtol=0)
                self.assertGreater((changed[:, own]-baseline[:, own]).abs().max(), 1e-4)

    def test_feasible_interior_origin_is_preserved_without_pulling_to_60(self):
        b = self.bundle; g = self.range_coupler()
        physical = torch.tensor(b.neutral_action, dtype=torch.float32)[None].expand(33, -1).clone()
        physical[:, g.indices[:, 0]] = torch.linspace(0, 1.2, 33)[:, None]
        data = b.reset_cpu(); data.qpos[:] = b.origin_qpos
        mujoco.mj_forward(b.model, data)
        qpos = torch.tensor(data.qpos, dtype=torch.float32)[None].expand(33, -1)
        matrices = torch.tensor(data.xmat, dtype=torch.float32)[None].expand(33, -1, -1)
        angles = []
        for row in physical:
            for group in g.groups:
                data.qpos[group['qpos_addresses'][0]] = row[group['full_indices'][0]]
            mujoco.mj_forward(b.model, data)
            actual = g.actual_diagnostics(torch.tensor(data.xmat, dtype=torch.float32)[None])
            angles.append(actual['coupling_actual_angle_deg'][0])
        angles = torch.stack(angles)
        interior = (angles > 52) & (angles < 78) & ((angles-60).abs() > 2)
        self.assertTrue(interior.any())
        output, info = g.apply(physical, actual_qpos=qpos, body_xmat=matrices)
        expected = g.origin[None].expand(33, -1, -1)
        torch.testing.assert_close(output[:, g.indices[:, 1:]][interior], expected[interior], atol=1e-6, rtol=0)
        torch.testing.assert_close(info['coupling_target_angle_deg'][interior], angles[interior], atol=6e-4, rtol=0)

    def test_actual_angle_uses_physics_not_requested_targets(self):
        g = self.coupler
        physical, qpos, matrices = self.states(2)
        diagnostics = g.actual_diagnostics(matrices)
        _, target = g.apply(physical, actual_qpos=qpos, body_xmat=matrices)
        self.assertGreater((diagnostics['coupling_actual_angle_deg']-target['coupling_target_angle_deg']).abs().max(), 1.)
        torch.testing.assert_close(diagnostics['coupling_actual_error_deg'],
                                   (diagnostics['coupling_actual_angle_deg']-60.).abs())

    def test_observation_drop_matches_names_preserves_score_and_stacking(self):
        b = self.bundle
        reduced = observation_contract(self.manifest, b.manifest['action_names'], OPTIONS)
        self.assertEqual(reduced['full_base_size']-reduced['reduced_base_size'], 32)
        self.assertEqual(reduced['reduced_base_size'], 1148)
        self.assertEqual(reduced['removed_base_indices'][:16], list(FOLLOWER_INDICES))
        self.assertIsNone(observation_contract(self.manifest, b.manifest['action_names'], None))
        full = observation_contract(self.manifest, b.manifest['action_names'], dict(OPTIONS, reduce_observation=False))
        self.assertEqual(full['retained_base_indices'], list(range(1180)))
        stacked = deepcopy(self.manifest)
        stacked['contract']['environment']['frame_stack'] = 3
        stacked['contract']['observation_size'] *= 3
        contract = observation_contract(stacked, b.manifest['action_names'], OPTIONS)
        self.assertEqual(contract['full_base_size']-contract['reduced_base_size'], 96)
        self.assertEqual(contract['removed_base_indices'][:3], [18, 19, 20])

    def test_origin_targets_feasible_and_model_unchanged(self):
        b = self.bundle; g = self.coupler
        data = b.reset_cpu(); data.qpos[:] = b.origin_qpos
        mujoco.mj_forward(b.model, data)
        physical = torch.tensor(b.neutral_action, dtype=torch.float32)[None]
        output, info = g.apply(physical, actual_qpos=torch.tensor(data.qpos, dtype=torch.float32)[None],
                               body_xmat=torch.tensor(data.xmat, dtype=torch.float32)[None])
        self.assertTrue(torch.all(info['coupling_target_error_deg'] < .05))
        self.assertEqual((b.model.nq, b.model.nu), (134, 46))
        self.assertEqual(len(b.active_action_indices), 46)
        self.assertEqual(len(set(FOLLOWER_INDICES) & set(active_indices(31))), 0)

    def test_unreachable_angle_clamps_followers_and_reports_error(self):
        low = self.low.copy(); high = self.high.copy()
        high[list(FOLLOWER_INDICES)] = 1e-4
        physical, qpos, matrices = self.states(2)
        for options in (OPTIONS, RANGE_OPTIONS):
            with self.subTest(mode=options['mode']):
                g = FingerCoupler(self.bundle, device='cpu', options=options, physical_low=low, physical_high=high)
                output, info = g.apply(physical, actual_qpos=qpos, body_xmat=matrices)
                self.assertTrue(info['coupling_infeasible'].any())
                self.assertTrue(torch.all(output[:, g.indices[:, 1:]] <= 1e-4))
                self.assertTrue(torch.all(output[:, g.indices[:, 1:]] >= 0))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
