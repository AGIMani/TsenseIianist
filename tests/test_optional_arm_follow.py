"""CPU kinematics/contracts, plus opt-in Slurm GPU integration for arm followers."""
import argparse
import copy
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault('MUJOCO_GL', 'disable')
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tsensei_v3_2_pianist.mjlab_env.arm_options import add_arguments, from_args, canonical
from tsensei_v3_2_pianist.mjlab_env.arm_follow import ArmFollower
from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.source_runtime import build, make_recipe
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.rl_torch.checkpoint import arm_contract


# Test fixtures only, never runtime defaults or user-approved initial poses.
ANGLES = [40., -30., 20., -50., 40., 10., -20.]
FLAGS = ['--arms', '--arm-initial-right-deg', *map(str, ANGLES),
         '--arm-initial-left-deg', *map(str, ANGLES)]
PREP = dict(duration=1., lift=.025, trajectory='quintic_v1', mode='policy_v1')


def options():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return from_args(parser.parse_args(FLAGS))


def core(worlds=3, dt=.005):
    poses = {s: (np.array([.2, -.1 if s == 'left' else .1, .125]),
                  Rotation.from_euler('xyz', [.2, -.4, .8]).as_matrix()) for s in ('right', 'left')}
    return ArmFollower(options(), poses, num_envs=worlds, dt=dt), poses


class ArmCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_uses_current_step_limits_without_goal_braking(self):
        f, poses = core(worlds=2)
        targets = {}
        for side, a in f.sides.items():
            goal = a['q0'][None].repeat(2, 1)
            goal[0, :4] += .15
            goal[1, :4] -= .15
            p, r, *_ = f.fk(a, goal)
            targets[side] = (p, r)
        feasible = {}
        for side, a in f.sides.items():
            feasible[side] = (a['q'].clone(), *f.bounds(a))
        f.step(targets)
        for side, a in f.sides.items():
            raw = f.last[side]['requested_q']
            q0, low, high = feasible[side]
            # With no remaining-distance braking, the issued q follows the
            # current-step feasible box directly.
            expected = f._coupled_segment(q0, raw.clamp(low, high))
            torch.testing.assert_close(a['q'], expected, atol=1e-12, rtol=0)
            self.assertGreater(float(a['v'].abs().max()), 0.)

    def test_explicit_angles_and_units_and_source_validation(self):
        p = argparse.ArgumentParser(); add_arguments(p)
        self.assertIsNone(from_args(p.parse_args([])))
        defaults = from_args(p.parse_args(['--arms']))
        np.testing.assert_allclose(defaults['initial_q_rad']['right'], np.deg2rad([-84.27055, -68.29417, 61.39675, -89.05803, -64.834334, 27.92905, -7.30834]))
        self.assertEqual(defaults['controller'], 'wuji_cartesian_v1')
        o = options()
        np.testing.assert_allclose(o['initial_q_rad']['left'], np.deg2rad(ANGLES))
        self.assertAlmostEqual(o['joint_limits']['left'][0]['velocity'], np.pi)
        self.assertAlmostEqual(o['joint_limits']['right'][0]['acceleration'], 2.5*np.pi)
        self.assertEqual(o['coupling']['constraint'], 'ccs696_j6_j7_absolute')
        self.assertAlmostEqual(o['coupling']['coefficient'], 1.025)
        self.assertAlmostEqual(o['coupling']['sum_limit_rad'], np.deg2rad(110.5))
        bad_angles = list(map(str, ANGLES)); bad_angles[5:7] = ['60', '90']
        with self.assertRaisesRegex(ValueError, 'J6/J7'):
            from_args(p.parse_args(['--arms', '--arm-initial-right-deg', *bad_angles,
                                    '--arm-initial-left-deg', *map(str, ANGLES)]))
        bad = copy.deepcopy(o); bad['source_sha256']['configs/preview/m6_sdk_joint_limits.json'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source differs'):
            canonical(bad)
        bad = copy.deepcopy(o); bad['initial_q_rad']['right'][0] = float('nan')
        with self.assertRaises(ValueError): canonical(bad)
        old = copy.deepcopy(o); old['controller'] = 'legacy_wrist_controller'
        with self.assertRaises(ValueError): canonical(old)
        broken = copy.deepcopy(o)
        broken['source_sha256']['configs/preview/hand_connector_20260930.json'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source differs'): canonical(broken)

    def test_base_aligns_full_pose_and_q0_remains_explicit(self):
        f, poses = core()
        for side, a in f.sides.items():
            p, r, *_ = f.fk(a, a['q'])
            np.testing.assert_allclose(p.numpy(), np.tile(poses[side][0], (3, 1)), atol=1e-12)
            np.testing.assert_allclose(r.numpy(), np.tile(poses[side][1], (3, 1, 1)), atol=1e-12)
            np.testing.assert_allclose(a['q'][0].numpy(), np.deg2rad(ANGLES), atol=0)

    def test_unreachable_requests_rate_bounds_reset_and_fixed_base(self):
        f, poses = core(worlds=2)
        bases = copy.deepcopy(f.contract['base'])
        for step in range(20):
            f.begin_control()
            targets = {s: (torch.tensor(poses[s][0]).repeat(2, 1) + torch.tensor(
                [[10. if step < 10 else -10., 2., -3.], [.01, .02, 0.]]),
                torch.tensor(poses[s][1]).repeat(2, 1, 1)) for s in f.sides}
            previous = {s: (a['q'].clone(), a['v'].clone()) for s, a in f.sides.items()}
            accepted = f.step(targets)
            for side, a in f.sides.items():
                q0, v0 = previous[side]
                v = (a['q'] - q0) / f.dt
                acc = (v - v0) / f.dt
                self.assertTrue(torch.all(v.abs() <= a['vmax']+1e-8))
                self.assertTrue(torch.all(acc.abs() <= a['amax']+1e-7))
                self.assertTrue(torch.all((a['q'] >= a['low']) & (a['q'] <= a['high'])))
                p, r, *_ = f.fk(a, a['q'])
                torch.testing.assert_close(p, accepted[side][0]); torch.testing.assert_close(r, accepted[side][1])
            self.assertTrue(torch.all(f.penalty() >= -5))
        self.assertEqual(f.contract['base'], bases)
        before = {s: a['q'][1].clone() for s, a in f.sides.items()}
        f.reset(torch.tensor([0]))
        for side, a in f.sides.items():
            torch.testing.assert_close(a['q'][0], a['q0']); torch.testing.assert_close(a['q'][1], before[side])
            self.assertTrue(torch.equal(a['v'][0], torch.zeros(7, dtype=torch.float64)))
        self.assertEqual(f.observation().shape, (2, 54))
        self.assertEqual(f.contract['base'], bases)

    def test_solver_and_trajectory_reject_ccs696_j6_j7_infeasible_points(self):
        f, poses = core(worlds=2)
        for side, a in f.sides.items():
            bad = a['q0'][None].repeat(2, 1)
            bad[:, 5] = np.deg2rad(60.)
            bad[:, 6] = np.deg2rad(90.)
            target_p, target_r, *_ = f.fk(a, bad)
            solved = f.solve(a, a['q'], target_p, target_r,
                              a['low'].expand_as(a['q']), a['high'].expand_as(a['q']))
            self.assertTrue(torch.all(f._coupling_valid(solved)))
            self.assertTrue(torch.any((solved - bad).abs() > 1e-4))
        targets = {}
        for side, a in f.sides.items():
            bad = a['q0'][None].repeat(2, 1)
            bad[:, 5] = np.deg2rad(60.)
            bad[:, 6] = np.deg2rad(90.)
            targets[side] = f.fk(a, bad)[:2]
        with self.assertRaisesRegex(RuntimeError, 'hard-limit verification failed'):
            for _ in range(120):
                f.step(targets)

    def test_penalty_is_one_max_budget_and_failed_step_is_atomic(self):
        f, poses = core(worlds=3)
        f.excess.copy_(torch.tensor([0, .05, .2]))
        torch.testing.assert_close(f.penalty(), torch.tensor([0., -2.5, -5.]))
        before = {s: a['q'].clone() for s, a in f.sides.items()}
        targets = {s: (torch.tensor(poses[s][0]).repeat(3, 1), torch.tensor(poses[s][1]).repeat(3, 1, 1)) for s in f.sides}
        targets['left'][0][0, 0] = float('nan')
        with self.assertRaises(ValueError): f.step(targets)
        for s, a in f.sides.items(): torch.testing.assert_close(a['q'], before[s])

    def test_position_boundary_retains_braking_feasibility(self):
        f, _ = core(worlds=2)
        for k in range(600):
            for a in f.sides.values():
                low, high = f.bounds(a)
                self.assertTrue(torch.all(low <= high))
                q = high if k < 300 else low
                v = (q - a['q']) / f.dt
                acc = (v - a['v']) / f.dt
                self.assertTrue(torch.all(q >= a['low']-1e-12) and torch.all(q <= a['high']+1e-12))
                self.assertTrue(torch.all(v.abs() <= a['vmax']+1e-8))
                self.assertTrue(torch.all(acc.abs() <= a['amax']+1e-7))
                a['q'] = q; a['v'] = v

    def test_checkpoint_requires_resolved_arm_contract(self):
        f, _ = core()
        self.assertEqual(arm_contract({}), (None, None))
        with self.assertRaisesRegex(ValueError, 'resolved'):
            arm_contract(dict(arm_follow=options()))
        self.assertEqual(arm_contract(dict(arm_follow=options(), arm_follow_contract=f.contract))[0], options())
        reward = template(); reward['terms']['arm_motion_limit_penalty']['enabled'] = True
        validate(reward)
        reward['terms']['arm_motion_limit_penalty']['params']['maximum_penalty_per_step'] = 10
        with self.assertRaises(ValueError): validate(reward)

    def test_arm_penalty_has_independent_budget_and_no_dt_scaling(self):
        from tests.test_paper_rewards import scene_fixture
        from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
        reward = template()
        for name, term in reward['terms'].items():
            term['enabled'] = name in ('arm_motion_limit_penalty', 'joint_motion_limit_penalty')
        reward['scale_by_dt'] = True
        manager = RewardManager(reward, scene_fixture(), {}, num_envs=3, device='cpu')
        values = manager({}, torch.ones(3, 51), {'arm_motion_excess': torch.tensor([0., .05, .2])},
                         torch.zeros(3, 89), torch.zeros(3, 10, 88))
        torch.testing.assert_close(values['arm_motion_limit_penalty'], torch.tensor([0., -2.5, -5.]))
        torch.testing.assert_close(values['joint_motion_limit_penalty'], torch.full((3,), -5.))

    def test_training_config_preserves_action_options_and_enables_only_arm_terms(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
        common = ['--no-arms', '--disable-wrist-tz', '--disable-wrist-yaw', '--preparation-time', '1',
                  '--preparation-mode', 'policy_v1', '--finger-coupling', '--no-coupling-reduce-observation']
        baseline = effective_config(parser().parse_args(common))
        arm_flags = ['--arms', '--disable-wrist-yaw', '--action-profile', 'wuji_cartesian_v1'] + FLAGS[1:]
        arm_common = ['--disable-wrist-yaw', '--preparation-time', '1',
                      '--preparation-mode', 'policy_v1', '--finger-coupling',
                      '--no-coupling-reduce-observation']
        enabled = effective_config(parser().parse_args(arm_common + arm_flags))
        self.assertEqual(enabled['finger_coupling'], baseline['finger_coupling'])
        self.assertIsNone(baseline['arm_follow'])
        self.assertIsNotNone(enabled['arm_follow'])
        for name in baseline['reward_config']['terms']:
            if name not in ('arm_motion_limit_penalty', 'arm_tracking_error_penalty'):
                self.assertEqual(enabled['reward_config']['terms'][name], baseline['reward_config']['terms'][name])
        self.assertTrue(enabled['reward_config']['terms']['arm_motion_limit_penalty']['enabled'])
        self.assertTrue(enabled['reward_config']['terms']['arm_tracking_error_penalty']['enabled'])

    def test_arms_and_no_arms_are_explicitly_mutually_exclusive(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
        with self.assertRaisesRegex(ValueError, '--arms and --no-arms'):
            effective_config(parser().parse_args(['--no-arms', *FLAGS]))


class ArmNativeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.task, _ = build(make_recipe(initial_buffer_time=0, preparation_time=1.,
            preparation_lift=.025, preparation_mode='policy_v1', task_profile='tsensei_collision_v1',
            drive_profile='shadow_v1', exact_note_timing=True))

    def bundle(self, arms=True):
        return SceneBundle(self.scene, disable_wrist_tz=False, disable_wrist_yaw=True,
                           preparation=PREP, arm_follow=options() if arms else None)

    def test_native_assembly_is_rigid_and_finger_key_inputs_preserved(self):
        original = self.bundle(False); b = self.bundle()
        self.assertEqual((original.model.nq, original.model.nu), (136, 48))
        self.assertEqual((b.model.nq, b.model.nu), (128, 40))
        self.assertEqual(len(b.policy_action_indices), 46)
        self.assertEqual(len(b.active_action_indices), 40)
        self.assertEqual(b.preparation['method'], 'policy_wrist_request_preparation_v1')
        self.assertIsNone(b.preparation['terminal_support_offset_world'])
        self.assertIn('actual_rigid_initial', b.preparation)
        old, new = original.model, b.model
        reference, d = original.reset_cpu(), b.reset_cpu()
        # Cartesian zero is the installed reference after removing the
        # hand-scene slide offsets; compare the native hand geometry at that
        # pose with the rigid-arm conversion.
        for side in ('right', 'left'):
            for axis in ('tx', 'ty', 'tz'):
                joint = reference.model.joint(f'{side}_/forearm_{axis}')
                reference.qpos[reference.model.jnt_qposadr[joint.id]] = 0.
        mujoco.mj_forward(reference.model, reference)
        for side in ('right', 'left'):
            for axis in ('tx', 'ty', 'tz', 'pitch'):
                self.assertEqual(mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT,
                    f'{side}_/forearm_{axis}'), -1)
            hand = new.body(f'{side}_/base_link')
            connector = new.body(f'arm_connector_{side}')
            flange = new.body(f'arm_flange_{side}')
            link = new.body(f'arm_follow_{side}_Link7_{side[0].upper()}')
            self.assertEqual(int(hand.parentid[0]), connector.id)
            self.assertEqual(int(connector.parentid[0]), flange.id)
            self.assertEqual(int(flange.parentid[0]), link.id)
            self.assertEqual(sum(int(x.jntnum[0]) for x in (hand, connector, flange, link)), 0)
            self.assertGreaterEqual(int(link.mocapid[0]), 0)
        for kind, count, fields in (
            ('body', old.nbody, ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat')),
            ('geom', old.ngeom, ('geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp')),
            ('actuator', old.nu, ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'))):
            for i in range(count):
                name = getattr(old, kind)(i).name
                if not name or (kind == 'actuator' and '_Palm_Base_' in name):
                    continue
                try: ni = getattr(new, kind)(name).id
                except KeyError: continue  # obsolete empty preparation wrappers
                for field in fields:
                    np.testing.assert_array_equal(getattr(new, field)[ni], getattr(old, field)[i])
        # The approved connector changes the world placement of the hand root;
        # verify the hand geometry in its own root frame instead of comparing
        # the old virtual-wrist world translation.
        for side in ('right', 'left'):
            old_root = reference.body(f'{side}_/base_link')
            new_root = d.body(f'{side}_/base_link')
            for i in range(old.ngeom):
                name = old.geom(i).name or ''
                if not name.startswith(side + '/'): continue
                old_local = old_root.xmat.reshape(3, 3).T @ (reference.geom(name).xpos - old_root.xpos)
                new_local = new_root.xmat.reshape(3, 3).T @ (d.geom(name).xpos - new_root.xpos)
                np.testing.assert_allclose(new_local, old_local, atol=2e-12, rtol=0)
                old_local_r = old_root.xmat.reshape(3, 3).T @ reference.geom(name).xmat.reshape(3, 3)
                new_local_r = new_root.xmat.reshape(3, 3).T @ d.geom(name).xmat.reshape(3, 3)
                np.testing.assert_allclose(new_local_r, old_local_r, atol=2e-12, rtol=0)

    def test_unachievable_command_has_request_error_but_zero_attachment_error(self):
        b = self.bundle(); d = b.reset_cpu()
        f = SceneArmFollower(b, num_envs=1, device='cpu')
        demand = torch.tensor(b.neutral_action)[None]
        start = f.requests(demand, torch.zeros(1))
        for side, a in f.sides.items():
            p, r, *_ = f.fk(a, a['q'])
            expected_p = p.clone(); expected_p[:, 2] += PREP['lift']
            torch.testing.assert_close(start[side][0], expected_p, atol=1e-12, rtol=0)
            torch.testing.assert_close(start[side][1], r, atol=1e-12, rtol=0)
        demand[:, [20, 45]] += .2
        for k in range(20):
            f.step(f.requests(demand, torch.tensor([(k+1)*f.dt])))
            before = {name: getattr(d, name).copy() for name in ('qpos', 'qvel', 'ctrl')}
            f.publish_numpy(d.mocap_pos, d.mocap_quat)
            for name, value in before.items(): np.testing.assert_array_equal(getattr(d, name), value)
            mujoco.mj_step(b.model, d); mujoco.mj_forward(b.model, d)
            for side, a in f.sides.items():
                p, r, *_ = f.fk(a, a['q'])
                actual = d.body(f.wrists[side]['body_name'])
                np.testing.assert_allclose(actual.xpos, p[0], atol=2e-12, rtol=0)
                np.testing.assert_allclose(actual.xmat.reshape(3, 3), r[0], atol=2e-12, rtol=0)
                flange = d.body(f'arm_flange_{side}')
                relative = flange.xmat.reshape(3, 3).T @ (actual.xpos-flange.xpos)
                expected = a['tool_pos'].numpy()-np.array([0, -.095, 0])
                np.testing.assert_allclose(relative, expected, atol=2e-12, rtol=0)
                self.assertTrue(torch.all(a['v'].abs() <= a['vmax']+1e-8))
                self.assertTrue(torch.all(a['acceleration'].abs() <= a['amax']+1e-7))
        self.assertGreater(float(f.last['left']['position_error'][0]), .02)
        self.assertGreater(float(f.excess[0]), 0.)

    def test_clearance_guard_rejects_convex_arm_geometry(self):
        from tsensei_v3_2_pianist.mjlab_env.finger_clearance import FingerClearanceGuard
        from tsensei_v3_2_pianist.mjlab_env.arm_options import ROOT
        b = self.bundle()
        options = json.loads((ROOT/'configs/finger_clearance/control_disabled_20260927.json').read_text())
        with self.assertRaisesRegex(ValueError, 'capsule collision geometry'):
            FingerClearanceGuard(b, device='cpu', options=options)

    def test_rigid_root_contacts_are_not_filtered_as_world_weld(self):
        from tsensei_v3_2_pianist.mjlab_env.self_collision import collision_weld_ids
        b = self.bundle(); d = b.reset_cpu(); m = b.model
        f = SceneArmFollower(b, num_envs=1, device='cpu')
        right = d.body('right_/base_link')
        a = f.sides['left']
        r = right.xmat.reshape(3, 3) @ a['tool_rot'].numpy().T
        p = right.xpos.copy() + np.array([.004, 0, 0]) - r @ a['tool_pos'].numpy()
        mid = f.wrists['left']['visual_ids'][-1]
        d.mocap_pos[mid] = p
        d.mocap_quat[mid] = np.roll(Rotation.from_matrix(r).as_quat(), 1)
        mujoco.mj_forward(m, d)
        names = {'right_/base_link_collision_0', 'left_/base_link_collision_0'}
        contacts = [c for c in d.contact if {m.geom(int(i)).name for i in c.geom} == names]
        self.assertTrue(contacts)
        self.assertTrue(any(c.dist < 0 for c in contacts))
        groups = collision_weld_ids(m)
        self.assertNotEqual(groups[m.body('right_/base_link').id], groups[m.body('left_/base_link').id])
        self.assertEqual(groups[m.body('right_/base_link').id], groups[m.body('right_/Right_hand_palm_link').id])
        mujoco.mj_step(m, d)
        self.assertTrue(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())

    def test_arm_collision_query_is_separate_and_covers_conflict_scopes(self):
        b = self.bundle(); query = b.arm_collision_query_model
        self.assertIsNotNone(query)
        self.assertEqual((query.nq, query.ngeom), (b.model.nq, b.model.ngeom))
        self.assertTrue(b.arm_collision_query_pairs)
        categories = {row[2] for row in b.arm_collision_query_pairs}
        self.assertEqual(categories, {'arm_arm', 'arm_hand', 'arm_piano'})
        # The active force model remains reaction-free for kinematic arm meshes.
        for gid in range(b.model.ngeom):
            name = b.model.geom(gid).name or ''
            if name.startswith(('arm_follow_', 'arm_connector_')):
                self.assertEqual((int(b.model.geom_contype[gid]), int(b.model.geom_conaffinity[gid])), (0, 0))
                self.assertEqual((int(query.geom_contype[gid]), int(query.geom_conaffinity[gid])), (1, 1))
        # Fixed Link7/flange/connector-to-own-hand mount contacts are excluded.
        for first, second, category in b.arm_collision_query_pairs:
            if category != 'arm_hand':
                continue
            body_a = query.body(int(query.geom_bodyid[query.geom(first).id])).name or ''
            body_b = query.body(int(query.geom_bodyid[query.geom(second).id])).name or ''
            if body_a.startswith('arm_follow_right_') or body_a.startswith('arm_connector_right_'):
                self.assertFalse(body_b.startswith('right_/') and any(
                    token in body_a for token in ('Link7_', 'arm_flange_', 'arm_connector_')))

    def test_finger_coupling_accepts_rigid_arm_bundle(self):
        from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
        from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
        b = self.bundle()
        low, high = physical_bounds('wuji_cartesian_v1', b.manifest['action_names'], b.low, b.high,
                                    range_scales=b.manifest.get('action_range_scales'))
        g = FingerCoupler(b, device='cpu', options=dict(schema=1, mode='distal_60deg_v1'),
                          physical_low=low, physical_high=high)
        self.assertFalse(any(g.reference_diagnostics['coupling_infeasible']))

    def gpu_env(self, arm_options=None):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.reward_config import canonical as reward_canonical
        from tsensei_v3_2_pianist.mjlab_env.arm_options import ROOT
        reward = reward_canonical(json.loads((ROOT/'configs/rewards/default_20261006.json').read_text()))
        reward['terms']['arm_motion_limit_penalty']['enabled'] = arm_options is not False
        controls = dict(schema=1, mode='target_clip_slew_v1', joint_state_limits='hard_v1', channels={})
        for side in ('R', 'L'):
            for axis in ('Tx', 'Ty'):
                controls['channels'][f'{side}_Palm_Base_{axis}'] = dict(max_velocity=.99, max_acceleration=1.99)
            if arm_options is False:
                controls['channels'][f'{side}_Palm_Base_Pitch'] = dict(max_velocity=3.1416, max_acceleration=15.708)
        return PianoTaskEnv(self.scene, self.task, num_envs=2, preparation=PREP,
            disable_wrist_tz=False, disable_wrist_yaw=True, action_profile='wuji_cartesian_v1',
            finger_coupling=dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False),
            arm_follow=(options() if arm_options is None else arm_options or None), reward_config=reward, physics_graph=True, collision_diagnostics=True,
            # Convex-hull links do not support the retired capsule envelope;
            # keep both arm and no-arm paths explicitly disabled.
            finger_clearance=None,
            control_limits=controls)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'GPU integration only inside explicitly selected Slurm validation')
    def test_gpu_training_step_and_partial_reset(self):
        env = self.gpu_env()
        obs = env.reset()
        self.assertEqual(env.action_size, 31)
        self.assertEqual((env.scene.model.nq, env.scene.model.nu), (128, 40))
        self.assertIsNone(env.scene.finger_clearance_guard)
        self.assertIsNone(env.scene.key_contact_diagnostics)
        action = env.scene.policy_action(env.scene.last_applied_physical_action)
        result = env.step(action)
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertEqual(obs.shape[-1], env.observation_size)
        self.assertIn('arm_motion_limit_penalty', result.info['reward_weighted_terms'])
        self.assertIn('coupling_substep_actual_error_deg_max', result.info)
        self.assertTrue(torch.isfinite(result.info['coupling_substep_actual_error_deg_max']).all())
        self.assertIn('arm_collision_substeps', result.info)
        from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
        from tsensei_v3_2_pianist.rl_torch.replay import Transition
        learner = SAC(env.observation_size, env.action_size,
            SACConfig(actor_hidden=(32, 32), critic_hidden=(32, 32)), device='cuda:0', action_profile='direct_joint_v1')
        update = learner.update(Transition(obs, action, result.reward, result.discount,
                                           result.observation, result.terminated, result.truncated))
        self.assertTrue(all(torch.isfinite(value).all() for value in update.values() if isinstance(value, torch.Tensor)))
        from tsensei_v3_2_pianist.mjlab_env.arm_follow import rotation_error
        checks = []
        def substep(scene):
            for side, a in scene.arm_follower.sides.items():
                p, r, *_ = scene.arm_follower.fk(a, a['q'])
                bid = scene.model.body(scene.arm_follower.wrists[side]['body_name']).id
                actual_p = scene.data.xpos[:, bid].double()
                actual_r = scene.data.xmat[:, bid].reshape(2, 3, 3).double()
                diagnostic = scene.arm_follower.last[side]
                torch.testing.assert_close(actual_p, p, atol=2e-6, rtol=0)
                torch.testing.assert_close(actual_r, r, atol=2e-6, rtol=0)
                torch.testing.assert_close((diagnostic['target_position']-actual_p).norm(dim=-1),
                    diagnostic['position_error'], atol=2e-6, rtol=0)
                flange = scene.model.body(f'arm_flange_{side}').id
                fp = scene.data.xpos[:, flange].double()
                fr = scene.data.xmat[:, flange].reshape(2, 3, 3).double()
                relative = (fr.transpose(-1, -2) @ (actual_p-fp)[..., None]).squeeze(-1)
                expected = (a['tool_pos']-a['q'].new_tensor([0, -.095, 0])).expand(2, 3)
                torch.testing.assert_close(relative, expected, atol=2e-6, rtol=0)
                self.assertTrue(torch.all(a['v'].abs() <= a['vmax'] + 1e-8))
                self.assertTrue(torch.all(a['acceleration'].abs() <= a['amax'] + 1e-7))
            checks.append(True)
        demand = env.scene.last_applied_physical_action.clone()
        demand[:, [20, 45]] += .1
        demand = demand.clamp(env.scene.physical_low, env.scene.physical_high)
        env.scene.step(env.scene.policy_action(demand), substep_callback=substep,
                       assigned_keys=env.logic.fingers[env.logic.index.clamp_max(env.logic.length-1)])
        self.assertEqual(len(checks), env.scene.decimation)
        scene = env.scene
        scene.step(scene.policy_action(demand), substep_callback=substep,
                   assigned_keys=env.logic.fingers[env.logic.index.clamp_max(env.logic.length-1)])
        self.assertEqual(len(checks), 2*scene.decimation)
        q = env.scene.arm_follower.sides['right']['q'][1].clone()
        physical = {name: getattr(env.scene.data, name)[1].clone()
                    for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat')}
        env.reset([0])
        torch.testing.assert_close(env.scene.arm_follower.sides['right']['q'][1], q)
        for name, before in physical.items():
            torch.testing.assert_close(getattr(env.scene.data, name)[1], before, atol=0, rtol=0)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'No-arm regression only inside explicit Slurm validation')
    def test_gpu_no_arm_default_preserves_geometry_graph_and_step(self):
        env = self.gpu_env(False); scene = env.scene
        self.assertIsNone(scene.arm_follower)
        self.assertEqual((scene.model.nq, scene.model.nu, env.action_size), (134, 46, 31))
        for k in range(2):
            action = scene.policy_action(scene.last_applied_physical_action)
            result = env.step(action)
            self.assertTrue(torch.isfinite(result.observation).all())
            self.assertNotIn('arm_motion_limit_penalty', result.info['reward_weighted_terms'])
        self.assertIsNone(scene.finger_clearance_guard)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'Rigid contact regression requires explicit Slurm validation')
    def test_gpu_rigid_root_contacts_reach_reward_and_diagnostics(self):
        from tsensei_v3_2_pianist.mjlab_env.arm_ik import quaternion
        env = self.gpu_env(); s = env.scene; f = s.arm_follower
        with s.scope():
            right = f.wrists['right']['body_id']; a = f.sides['left']
            r = s.data.xmat[:, right].reshape(2, 3, 3).double() @ a['tool_rot'].T
            p = s.data.xpos[:, right].double() + a['q'].new_tensor([.004, 0, 0]) - (r @ a['tool_pos'][:, None]).squeeze(-1)
            mid = f.wrists['left']['visual_ids'][-1]
            s.data.mocap_pos[:, mid] = p.float()
            s.data.mocap_quat[:, mid] = quaternion(r).float()
            s.sim.forward()
            s.self_collision.clear(); s._sample_self_collision(count_substep=True)
            ids = {s.model.geom(name).id for name in ('right_/base_link_collision_0', 'left_/base_link_collision_0')}
            count = int(s.data.nacon[0]); geoms = s.data.contact.geom[:count].cpu().numpy()
            dist = s.data.contact.dist[:count].cpu().numpy()
            self.assertTrue(any(set(map(int,pair)) == ids and gap < 0 for pair,gap in zip(geoms,dist)))
            pair = tuple(sorted(('right_/base_link', 'left_/base_link')))
            tracker_id = next(i+1 for i,names in enumerate(s.self_collision.pair_names) if tuple(sorted(names)) == pair)
            self.assertTrue(torch.all(s.self_collision.seen[:, tracker_id] == 1))
            self.assertTrue(torch.all(env.snapshot()['palm_collision']))
            s.sim.step(); s.sim.forward()
            self.assertTrue(torch.isfinite(s.data.qpos[:]).all())

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'Arm query regression requires explicit Slurm validation')
    def test_gpu_arm_joint_limit_cached_forward_matches_eager(self):
        import warp as wp
        env = self.gpu_env(); s = env.scene
        self.assertIsNotNone(s.joint_limiter)
        row = s.joint_limiter.contract['joints'][0]
        with s.scope():
            # Restore the same solver inputs before each branch so warm starts
            # cannot make a second forward appear equivalent by coincidence.
            fields = ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'mocap_pos', 'mocap_quat')
            initial = {name: getattr(s.data, name)[:].clone() for name in fields}
            for bound, speed in ((row['float32_min']-.01, -1.),
                                 (row['float32_max']+.01, 1.)):
                snapshots = []
                for cached in (False, True):
                    for name, value in initial.items():
                        getattr(s.data, name)[:] = value
                    s.data.qpos[:, row['qpos_address']] = bound
                    s.data.qvel[:, row['dof_address']] = speed
                    s.joint_limiter.counts.zero_()
                    s._enforce_joint_limits(use_cached_forward=cached)
                    d = s.sim.wp_data
                    state = {name: wp.to_torch(getattr(d, name)).clone() for name in
                             ('qpos', 'qvel', 'xpos', 'geom_xpos', 'qacc',
                              'qfrc_constraint', 'nefc', 'overflow')}
                    state['limit_counts'] = wp.to_torch(s.joint_limiter.counts).clone()
                    n = int(wp.to_torch(d.nacon)[0])
                    # Contact/constraint insertion order may vary across GPU
                    # launches; compare the same physical rows after sorting.
                    contacts = np.column_stack((d.contact.worldid.numpy()[:n],
                        d.contact.geom.numpy()[:n], d.contact.dist.numpy()[:n],
                        d.contact.pos.numpy()[:n]))
                    contacts = np.asarray(sorted(map(tuple, contacts))).reshape(n, 7)
                    constraints = []
                    for world, count in enumerate(state['nefc'].tolist()):
                        values = np.column_stack([wp.to_torch(getattr(d.efc, name))[
                            world, :count].cpu().numpy() for name in
                            ('type', 'pos', 'margin', 'vel', 'aref', 'force')])
                        constraints.append(np.asarray(sorted(map(tuple, values))).reshape(count, 6))
                    snapshots.append((state, contacts, constraints))
                    self.assertTrue(torch.all(state['limit_counts'].sum(-1) > 0))
                    self.assertTrue(torch.all(state['qvel'][:, row['dof_address']] == 0))
                eager, cached = snapshots
                for name in eager[0]:
                    exact = name in ('qpos', 'qvel', 'xpos', 'geom_xpos', 'nefc', 'overflow', 'limit_counts')
                    torch.testing.assert_close(cached[0][name], eager[0][name],
                                               atol=0 if exact else 1e-5,
                                               rtol=0 if exact else 1e-5)
                np.testing.assert_allclose(cached[1], eager[1], atol=1e-6, rtol=1e-6)
                for actual, reference in zip(cached[2], eager[2]):
                    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'Arm query regression requires explicit Slurm validation')
    def test_gpu_arm_query_detects_arm_piano_conflict_without_force_contacts(self):
        env = self.gpu_env(); s = env.scene
        query = s.bundle.arm_collision_query_model
        first, second, category = next(row for row in s.bundle.arm_collision_query_pairs
                                       if row[2] == 'arm_piano')
        self.assertEqual(category, 'arm_piano')
        aid = s.model.geom(first).id; pid = s.model.geom(second).id
        body = int(s.model.geom_bodyid[aid]); mocap = int(s.model.body_mocapid[body])
        self.assertGreaterEqual(mocap, 0)
        with s.scope():
            # Put the arm geom center exactly on the piano geom center in the
            # main model; only the query model should report the overlap.
            target = s.data.geom_xpos[:, pid].clone()
            local = s.tensor(s.model.geom_pos[aid])
            s.data.mocap_pos[:, mocap] = target - local
            s.data.mocap_quat[:, mocap] = s.tensor([1., 0., 0., 0.])
            s.sim.forward()
            tracker = s.arm_collision_tracker
            import warp as wp
            original_graph = s.physics_graph
            physical = tuple(value.clone() for value in
                             (s.data.qpos[:], s.data.qvel[:], s.data.ctrl[:]))
            expected = []
            # The first capture must not add a second sample. Subsequent
            # replays must see live geometry and preserve all pair counters.
            for use_graph in (False, True):
                s.physics_graph = use_graph
                tracker.clear()
                for sample in range(3):
                    offset = s.tensor([0., 0., .4 if sample == 1 else 0.])
                    s.data.mocap_pos[:, mocap] = target - local + offset
                    s.sim.forward()
                    s._query_and_sample_arm_collision()
                    snapshot = tuple(value.clone() for value in (
                        tracker.seen, tracker.pair_substep_seen,
                        tracker.pair_substep_totals,
                        wp.to_torch(s.arm_collision_data.nacon),
                        wp.to_torch(s.arm_collision_data.overflow)))
                    if not use_graph:
                        expected.append(snapshot)
                    else:
                        for actual, reference in zip(snapshot, expected[sample]):
                            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                    self.assertTrue(torch.all(tracker.count_substeps('arm_piano') > 0))
                if use_graph and s.sim.use_cuda_graph:
                    self.assertIsNotNone(s._arm_collision_graph)
            s.physics_graph = original_graph
            if s.sim.use_cuda_graph:
                old_graph = s._arm_collision_graph
                s.sim.create_graph()
                tracker.clear()
                s._query_and_sample_arm_collision()
                self.assertIsNot(s._arm_collision_graph, old_graph)
                for field in ('geom_xpos', 'geom_xmat', 'xpos', 'xquat', 'xmat'):
                    self.assertIs(getattr(s.arm_collision_data, field), getattr(s.sim.wp_data, field))
                for actual, reference in zip((tracker.seen, tracker.pair_substep_seen,
                                              tracker.pair_substep_totals,
                                              wp.to_torch(s.arm_collision_data.nacon),
                                              wp.to_torch(s.arm_collision_data.overflow)), expected[0]):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            for actual, reference in zip((s.data.qpos[:], s.data.qvel[:], s.data.ctrl[:]), physical):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            # Main force contacts still contain no arm geometry because the
            # active model keeps arm meshes collision-disabled.
            n = int(s.data.nacon[0])
            geom = s.data.contact.geom[:n].cpu().numpy()
            self.assertFalse(any(int(a) == aid or int(b) == aid for a, b in geom))

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_LONG_TEST') == '1',
                         'Long physical trajectory only inside explicit Slurm validation')
    def test_gpu_left_ramp_stop_reverse_and_unreachable_recovery(self):
        # Explicit historical user posture as a regression fixture, not defaults.
        p = argparse.ArgumentParser(); add_arguments(p)
        left = [84.27055, -68.29417, -61.39675, -89.05803, 158.27133, 6.45346, -28.12498]
        right = [-84.27055, -68.29417, 61.39675, -89.05803, -64.8343338991625, 27.929049544537655, -7.308340216508003]
        arm_options = from_args(p.parse_args(['--arms', '--arm-initial-right-deg', *map(str, right),
                                             '--arm-initial-left-deg', *map(str, left)]))
        env = self.gpu_env(arm_options); scene = env.scene
        initial = scene.control_reset_action.clone()
        # World 0: the same accelerating 42 cm left request and abrupt hold.
        # World 1: deliberately farther request, then reverse to the initial XY.
        # Preparation is an IK request; the base remains fixed throughout.
        rows = {k: [] for k in ('time', 'q', 'v', 'a', 'request_error', 'attachment_error', 'actual_wrist', 'reference', 'wrist_target')}
        bases = copy.deepcopy(scene.arm_follower.contract['base'])
        def sample(s):
            with s.scope():
                row = {k: [] for k in ('q', 'v', 'a', 'request_error', 'attachment_error', 'actual_wrist', 'reference')}
                for side, a in s.arm_follower.sides.items():
                    p, r, *_ = s.arm_follower.fk(a, a['q'])
                    w = s.arm_follower.wrists[side]
                    actual_p = s.data.xpos[:, w['body_id']].double()
                    actual_r = s.data.xmat[:, w['body_id']].reshape(2, 3, 3).double()
                    torch.testing.assert_close(actual_p, p, atol=2e-6, rtol=0)
                    torch.testing.assert_close(actual_r, r, atol=2e-6, rtol=0)
                    fp = s.data.xpos[:, w['flange_id']].double()
                    fr = s.data.xmat[:, w['flange_id']].reshape(2, 3, 3).double()
                    expected = a['tool_pos']-a['q'].new_tensor([0, -.095, 0])
                    gap = (actual_p - fp - (fr @ expected[:, None]).squeeze(-1)).norm(dim=-1)
                    self.assertTrue(torch.all(gap < 2e-6))
                    self.assertTrue(torch.all(a['v'].abs() <= a['vmax']+1e-8))
                    self.assertTrue(torch.all(a['acceleration'].abs() <= a['amax']+1e-7))
                    self.assertTrue(torch.all((a['q'] >= a['low']) & (a['q'] <= a['high'])))
                    for key, value in [('q', a['q']), ('v', a['v']), ('a', a['acceleration']),
                        ('request_error', s.arm_follower.last[side]['position_error']),
                        ('attachment_error', gap), ('actual_wrist', actual_p), ('reference', a['reference'])]: row[key].append(value)
                for key, values in row.items(): rows[key].append(torch.stack(values, 1).cpu().numpy())
                rows['wrist_target'].append(s.last_applied_physical_action[:, [20,21,23,45,46,48]].cpu().numpy().copy())
        assignments = torch.zeros((2, 10, 88), device=scene.device)
        for step in range(260):
            t = step*.05
            d = .42 * min(max((t-1.)/3., 0.), 1.)**3
            demand = initial[None].repeat(2, 1)
            demand[0, 45] -= d
            demand[1, 45] -= d if t < 4 else (.48 if t < 6 else 0.)
            scene.step(scene.policy_action(demand), substep_callback=sample, assigned_keys=assignments)
            if (step+1) % 40 == 0: print('rigid_validation_physical_seconds', (step+1)*.05, flush=True)
        rows = {k: np.asarray(v) for k, v in rows.items() if k != 'time'}
        times = (np.arange(len(rows['q']))+1)*.005
        self.assertEqual(scene.arm_follower.contract['base'], bases)
        # Stop/return must settle, not merely obey rate bounds while oscillating.
        late = times >= 12.
        from tsensei_v3_2_pianist.mjlab_env.arm_options import ROOT
        out = ROOT.parent/'rigid_arm_validation'; out.mkdir(exist_ok=True)
        np.savez_compressed(out/'physical_200hz.npz', time=times, **rows)
        report = dict(physics_hz=200, duration_s=13, worlds=2, policy_loaded=False,
            description='world0 accelerating left request then stop; world1 farther request then return',
            max_attachment_error_m=float(rows['attachment_error'].max()),
            max_request_error_m=float(rows['request_error'].max()),
            last_second_max_speed_rad_s=float(np.abs(rows['v'][late]).max()),
            last_second_max_request_error_m=float(rows['request_error'][late].max()),
            arm_contract=scene.arm_follower.contract)
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        self.assertLess(float(np.abs(rows['v'][late]).max()), .001)
        self.assertLess(float(rows['request_error'][late].max()), .0001)
        # After a stationary far target reverses, its limited wrist goal must
        # approach the new goal monotonically instead of manufacturing swings.
        command = rows['wrist_target'][:, 1, 3]
        self.assertTrue(np.all(np.diff(command[times>=6.005]) >= -1e-7))


if __name__ == '__main__':
    unittest.main()
