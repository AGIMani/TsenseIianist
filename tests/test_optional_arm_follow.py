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

    def test_stationary_reference_brakes_without_repeated_overshoot(self):
        f, poses = core(worlds=2)
        targets = {}
        for side, a in f.sides.items():
            goal = a['q0'][None].repeat(2, 1)
            goal[0, :4] += .15
            goal[1, :4] -= .15
            p, r, *_ = f.fk(a, goal)
            targets[side] = (p, r)
        history = []
        for k in range(350):
            f.step(targets)
            history.append(f.sides['left']['q'].clone())
        for side, a in f.sides.items():
            self.assertLess(float(a['v'].abs().max()), 1e-6)
            self.assertLess(float(f.last[side]['position_error'].max()), 1e-6)
            self.assertLess(float(f.last[side]['orientation_error'].max()), 1e-5)
        self.assertLess(float((torch.stack(history[-100:])-history[-1]).abs().max()), 1e-6)

    def test_explicit_angles_and_units_and_source_validation(self):
        p = argparse.ArgumentParser(); add_arguments(p)
        self.assertIsNone(from_args(p.parse_args([])))
        with self.assertRaisesRegex(ValueError, '无法开启机械臂训练'):
            from_args(p.parse_args(['--arms']))
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
        for _ in range(120):
            f.step(targets)
            for a in f.sides.values():
                self.assertTrue(torch.all(f._coupling_valid(a['q'])))
                self.assertTrue(torch.all(a['v'].abs() <= a['vmax'] + 1e-8))
                self.assertTrue(torch.all(a['acceleration'].abs() <= a['amax'] + 1e-7))

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

    def test_training_config_preserves_action_options_and_enables_only_arm_term(self):
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
        common = ['--no-arms', '--disable-wrist-tz', '--disable-wrist-yaw', '--preparation-time', '1',
                  '--preparation-mode', 'policy_v1', '--finger-coupling', '--no-coupling-reduce-observation']
        baseline = effective_config(parser().parse_args(common))
        enabled = effective_config(parser().parse_args(common + FLAGS))
        self.assertEqual(enabled['finger_coupling'], baseline['finger_coupling'])
        self.assertIsNone(baseline['arm_follow'])
        for name in baseline['reward_config']['terms']:
            if name != 'arm_motion_limit_penalty':
                self.assertEqual(enabled['reward_config']['terms'][name], baseline['reward_config']['terms'][name])
        self.assertTrue(enabled['reward_config']['terms']['arm_motion_limit_penalty']['enabled'])


class ArmNativeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.task, _ = build(make_recipe(initial_buffer_time=0, preparation_time=1.,
            preparation_lift=.025, preparation_mode='policy_v1', task_profile='tsensei_collision_v1',
            drive_profile='shadow_v1', exact_note_timing=True))

    def bundle(self, arms=True):
        return SceneBundle(self.scene, disable_wrist_tz=True, disable_wrist_yaw=True,
                           preparation=PREP, arm_follow=options() if arms else None)

    def test_native_assembly_is_rigid_and_finger_key_inputs_preserved(self):
        original = self.bundle(False); b = self.bundle()
        self.assertEqual((original.model.nq, original.model.nu), (134, 46))
        self.assertEqual((b.model.nq, b.model.nu), (128, 40))
        self.assertEqual(len(b.policy_action_indices), 46)
        self.assertEqual(len(b.active_action_indices), 40)
        self.assertEqual(b.preparation['method'], 'policy_wrist_request_preparation_v1')
        self.assertIsNone(b.preparation['terminal_support_offset_world'])
        self.assertIn('actual_rigid_initial', b.preparation)
        old, new = original.model, b.model
        reference, d = original.reset_cpu(), b.reset_cpu()
        for side in ('right', 'left'):
            for axis in ('tx', 'ty', 'pitch'):
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
                if not name or (kind == 'actuator' and '_Wrist_' in name):
                    continue
                try: ni = getattr(new, kind)(name).id
                except KeyError: continue  # obsolete empty preparation wrappers
                for field in fields:
                    np.testing.assert_array_equal(getattr(new, field)[ni], getattr(old, field)[i])
        for i in range(old.ngeom):
            name = old.geom(i).name
            if name:
                np.testing.assert_allclose(d.geom(name).xpos, reference.geom(name).xpos, atol=2e-12, rtol=0)
                np.testing.assert_allclose(d.geom(name).xmat, reference.geom(name).xmat, atol=2e-12, rtol=0)

    def test_unachievable_command_has_request_error_but_zero_attachment_error(self):
        b = self.bundle(); d = b.reset_cpu()
        f = SceneArmFollower(b, num_envs=1, device='cpu')
        demand = torch.tensor(b.neutral_action)[None]
        start = f.requests(demand, torch.zeros(1))
        for side, a in f.sides.items():
            p, r, *_ = f.fk(a, a['q'])
            torch.testing.assert_close(start[side][0], p, atol=1e-12, rtol=0)
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

    def test_clearance_fk_uses_full_actual_rigid_hand_frame(self):
        from tsensei_v3_2_pianist.mjlab_env.finger_clearance import FingerClearanceGuard
        from tsensei_v3_2_pianist.mjlab_env.arm_options import ROOT
        b = self.bundle(); d = b.reset_cpu()
        guard = FingerClearanceGuard(b, device='cpu', options=json.loads(
            (ROOT/'configs/finger_clearance/control_disabled_20260927.json').read_text()))
        f = SceneArmFollower(b, num_envs=1, device='cpu')
        for a in f.sides.values(): a['q'][:, :3] += torch.tensor([.08, -.03, .05])
        f.publish_numpy(d.mocap_pos, d.mocap_quat); mujoco.mj_forward(b.model, d)
        support = guard.support(torch.tensor(d.mocap_pos)[None].float(),
            torch.tensor(d.xpos)[None].float(), torch.tensor(d.xmat)[None].float())
        physical = guard.physical_qpos(torch.tensor(d.qpos)[None].float())
        endpoints, *_ = guard.fk(physical, support)
        for index, group in enumerate(guard.groups):
            for j, geom in enumerate(group['geoms']):
                p = d.geom_xpos[geom]; axis = d.geom_xmat[geom].reshape(3, 3)[:, 2]
                h = b.model.geom_size[geom, 1]
                np.testing.assert_allclose(endpoints[0, index, j].numpy(),
                    np.stack([p-axis*h, p+axis*h]), atol=2e-7, rtol=0)
        projected, metrics = guard.filter(physical, actual_qpos=torch.tensor(d.qpos)[None].float(),
            mocap_pos=torch.tensor(d.mocap_pos)[None].float(), support=support,
            assigned_keys=torch.zeros(1, 10, 88))
        torch.testing.assert_close(projected, physical, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(metrics['finger_actual_deficit_m']).all())

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

    def gpu_env(self, arm_options=None):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.reward_config import canonical as reward_canonical
        from tsensei_v3_2_pianist.mjlab_env.arm_options import ROOT
        reward = reward_canonical(json.loads((ROOT/'configs/rewards/default_20261006.json').read_text()))
        reward['terms']['arm_motion_limit_penalty']['enabled'] = arm_options is not False
        controls = dict(schema=1, mode='target_clip_slew_v1', joint_state_limits='hard_v1', channels={})
        for side in ('R', 'L'):
            for axis in ('TX', 'TY'):
                controls['channels'][f'{side}_Wrist_{axis}'] = dict(max_velocity=.99, max_acceleration=1.99)
            controls['channels'][f'{side}_Wrist_Pitch'] = dict(max_velocity=3.1416, max_acceleration=15.708)
        return PianoTaskEnv(self.scene, self.task, num_envs=2, preparation=PREP,
            disable_wrist_tz=True, disable_wrist_yaw=True, action_profile='direct_joint_v1',
            finger_coupling=dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False),
            arm_follow=(options() if arm_options is None else arm_options or None), reward_config=reward, physics_graph=True, collision_diagnostics=True,
            finger_clearance=json.loads((ROOT/'configs/finger_clearance/control_disabled_20260927.json').read_text()),
            control_limits=controls)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'GPU integration only inside explicitly selected Slurm validation')
    def test_gpu_training_step_and_partial_reset(self):
        env = self.gpu_env()
        obs = env.reset()
        self.assertEqual(env.action_size, 31)
        self.assertEqual((env.scene.model.nq, env.scene.model.nu), (128, 40))
        self.assertIsNotNone(env.scene.finger_clearance_guard)
        self.assertIsNotNone(env.scene.key_contact_diagnostics)
        action = env.scene.policy_action(env.scene.last_applied_physical_action)
        result = env.step(action)
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertEqual(obs.shape[-1], env.observation_size)
        self.assertIn('arm_motion_limit_penalty', result.info['reward_weighted_terms'])
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
        keys = env.logic.fingers[env.logic.index]
        with scene.scope():
            request = scene.last_applied_physical_action.clone()
            expected, expected_metrics = scene.finger_clearance_guard.filter(request,
                actual_qpos=scene.data.qpos[:], mocap_pos=scene.data.mocap_pos[:], assigned_keys=keys)
            actual, metrics = scene._filter_finger_targets(request, keys)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
            for key, values in expected_metrics.items():
                torch.testing.assert_close(metrics[key], values, atol=1e-6, rtol=0)

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
