"""Whole collision links, source-model FK parity, projection and role gating."""
from copy import deepcopy
import unittest

import mujoco
import numpy as np
import torch

from tsensei_v3_2_pianist.mjlab_env.finger_clearance import canonical, FingerClearanceGuard


OPTIONS = dict(schema=1, mode='full_finger_clearance_v1')


class OptionsTests(unittest.TestCase):
    def test_explicit_and_validated(self):
        self.assertIsNone(canonical(None))
        self.assertEqual(canonical(OPTIONS)['clearance_m'], .002)
        for value in ({}, dict(OPTIONS, extra=2), dict(OPTIONS, clearance_m=float('nan')),
                      dict(OPTIONS, projection_iterations=True), dict(OPTIONS, contact_slack_m=-1),
                      dict(OPTIONS, idle_finger_protection=1), dict(OPTIONS, active_press_permission='false')):
            with self.subTest(value=value), self.assertRaises(ValueError): canonical(value)
        self.assertNotIn('idle_finger_protection', canonical(OPTIONS))
        self.assertFalse(canonical(dict(OPTIONS, idle_finger_protection=False))['idle_finger_protection'])


class RealGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
        recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1', initial_buffer_time=0.,
                             preparation_time=1., preparation_mode='policy_v1')
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent/'hand_only')
        cls.bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'])
        cls.guard = FingerClearanceGuard(cls.bundle, device='cpu', options=OPTIONS)
        cls.score = np.load(task/'score.npz')['fingers']

    def inputs(self, n=1):
        b = self.bundle
        qpos = torch.tensor(b.origin_qpos, dtype=torch.float32).expand(n, -1).clone()
        mocap = torch.tensor(b.origin_mocap_pos, dtype=torch.float32).expand(n, -1, -1).clone()
        physical = self.guard.physical_qpos(qpos)
        assigned = torch.zeros((n, 10, 88), dtype=torch.bool)
        return physical, qpos, mocap, assigned

    def test_full_capsule_fk_matches_native_and_translated_mounts(self):
        g = self.guard;b = self.bundle
        physical, qpos, mocap, assigned = self.inputs(3)
        generator = torch.Generator().manual_seed(491)
        physical[1:, g.finger_indices] = g.low+(g.high-g.low)*torch.rand((2, 10, 4), generator=generator)
        physical[1, 20] = .2;physical[2, 45] = -.15
        physical[1, 23] = .15;physical[2, 48] = -.19
        mocap[1, :, 2] += .017
        ends, _, _ = g.fk(physical, g.lift(mocap))
        for world in range(3):
            data = b.reset_cpu();data.qpos[:] = b.origin_qpos
            data.mocap_pos[:] = mocap[world].numpy();data.mocap_quat[:] = b.origin_mocap_quat
            for act, full in zip(b.action_ids, b.active_action_indices):
                joint = int(b.model.actuator_trnid[act, 0]);data.qpos[b.model.jnt_qposadr[joint]] = physical[world, full]
            mujoco.mj_forward(b.model, data)
            expected = []
            for group in g.groups:
                row = []
                for geom in group['geoms']:
                    span = data.geom_xmat[geom].reshape(3, 3)[:, 2]*b.model.geom_size[geom, 1]
                    row.append([data.geom_xpos[geom]-span, data.geom_xpos[geom]+span])
                expected.append(row)
            np.testing.assert_allclose(ends[world].numpy(), expected, atol=8e-8, rtol=0)

    def test_origin_is_locally_clear_and_padding_rejects_false_exemption(self):
        physical, _, mocap, assigned = self.inputs()
        g = self.guard
        result = g.evaluate(physical, g.lift(mocap), assigned)
        self.assertEqual(result['finger_clearance_deficit_m'].max().item(), 0.)
        self.assertGreater(result['finger_min_clearance_m'][0, 0].item(), .009)
        # Marking a finger active is insufficient: its terminal capsule must
        # actually fit over the assigned key before any lower plane is allowed.
        assigned[0, 5, 0] = True
        result = g.evaluate(physical, g.lift(mocap), assigned)
        self.assertTrue(result['finger_active'][0, 5])
        self.assertFalse(result['finger_terminal_allowed'][0, 5])

    def test_projection_leaves_wrists_and_inputs_and_is_stateless(self):
        g = self.guard
        physical, qpos, mocap, assigned = self.inputs(8)
        generator = torch.Generator().manual_seed(22)
        physical[:, g.finger_indices] = g.low+(g.high-g.low)*torch.rand((8, 10, 4), generator=generator)
        before = physical.clone();before_q = qpos.clone();before_m = mocap.clone()
        applied, info = g.filter(physical, actual_qpos=qpos, mocap_pos=mocap, assigned_keys=assigned)
        self.assertGreater(info['finger_requested_deficit_m'].max().item(), .01)
        self.assertLessEqual(info['finger_applied_deficit_m'].max().item(), 1.1e-6)
        unmodified = torch.ones(51, dtype=torch.bool);unmodified[g.finger_indices.flatten()] = False
        torch.testing.assert_close(applied[:, unmodified], physical[:, unmodified], atol=0, rtol=0)
        torch.testing.assert_close(physical, before, atol=0, rtol=0)
        torch.testing.assert_close(qpos, before_q, atol=0, rtol=0)
        torch.testing.assert_close(mocap, before_m, atol=0, rtol=0)
        again, _ = g.filter(physical, actual_qpos=qpos, mocap_pos=mocap, assigned_keys=assigned)
        torch.testing.assert_close(again, applied, atol=0, rtol=0)

    def test_capsule_jacobian_matches_finite_difference(self):
        physical, _, mocap, assigned = self.inputs()
        g = self.guard;lift = g.lift(mocap)
        result = g._evaluate(physical, lift, assigned, jacobian=True)
        epsilon = 1e-3
        for finger in (0, 2, 5, 9):
            for joint in range(4):
                changed = physical.clone();changed[:, g.finger_indices[finger, joint]] += epsilon
                plus = g.evaluate(changed, lift, assigned)['finger_min_clearance_m'][0, finger]
                changed[:, g.finger_indices[finger, joint]] -= 2*epsilon
                minus = g.evaluate(changed, lift, assigned)['finger_min_clearance_m'][0, finger]
                self.assertAlmostEqual(float((plus-minus)/(2*epsilon)),
                                       float(result['gradient'][0, finger, joint]), delta=2e-5)

    def test_active_pad_preserves_white_and_black_full_travel_endpoints(self):
        # CPU IK fixtures, not prescribed policy postures or natural-pose maps.
        # Same exact endpoint must be rejected as an idle finger or a different
        # key assignment; only its correctly assigned terminal pad is exempt.
        fixtures = [
            (5, 39, [-.0132817422, .1102701363, .0125622524,
                     .5710508567, 1.1698944773, .4581148256, .5953316132]),
            (5, 40, [-.0052808764, .1437679927, .1242447358,
                     .3849702098, 1.2575165872, .2975314135, .8479951000]),
            (6, 40, [.0641514892, .0236265381, -.0178640849,
                     .4442309614, .0370124923, .6540901639, .2693992817]),
        ]
        g = self.guard
        for finger, key, angles in fixtures:
            with self.subTest(finger=finger, key=key):
                physical, _, mocap, assigned = self.inputs()
                physical[:, g.chain_indices[finger]] = torch.tensor(angles)
                assigned[0, finger, key] = True
                active = g.evaluate(physical, g.lift(mocap), assigned)
                self.assertTrue(active['finger_terminal_allowed'][0, finger])
                self.assertLessEqual(active['finger_clearance_deficit_m'][0, finger], 1e-6)
                idle = g.evaluate(physical, g.lift(mocap), torch.zeros_like(assigned))
                self.assertGreater(idle['finger_clearance_deficit_m'][0, finger], .005)
                wrong = torch.zeros_like(assigned);wrong[0, finger, key+2] = True
                self.assertFalse(g.evaluate(physical, g.lift(mocap), wrong)['finger_terminal_allowed'][0, finger])

    def test_idle_only_ablation_passes_active_force_commands_and_keeps_diagnostics(self):
        options = dict(OPTIONS, active_command_mode='idle_only_probe')
        guard = FingerClearanceGuard(self.bundle, device='cpu', options=options)
        physical, qpos, mocap, assigned = self.inputs(2)
        physical[:, guard.finger_indices] = guard.high
        assigned[:, 5, 39] = True
        applied, metrics = guard.filter(physical, actual_qpos=qpos, mocap_pos=mocap, assigned_keys=assigned)
        torch.testing.assert_close(applied[:,guard.finger_indices[5]],physical[:,guard.finger_indices[5]],atol=0,rtol=0)
        self.assertFalse(metrics['finger_projection_mask'][:,5].any())
        self.assertFalse(metrics['finger_fallback_mask'][:,5].any())
        self.assertGreater(metrics['finger_requested_deficit_m'][:,5].max().item(), .01)
        self.assertGreater(metrics['finger_applied_deficit_m'][:,5].max().item(), .01)
        self.assertTrue(metrics['finger_projection_mask'][:,:5].any())

    def test_contact_mode_uses_actual_support_for_force_and_reward(self):
        guard=FingerClearanceGuard(self.bundle,device='cpu',options=dict(OPTIONS,active_command_mode='active_contact_v1'))
        physical,qpos,mocap,assigned=self.inputs()
        # Verified white-key endpoint; actual pad can press while the servo's
        # desired pose extends through the surface to supply contact force.
        pose=[-.0132817422,.1102701363,.0125622524,.5710508567,1.1698944773,.4581148256,.5953316132]
        physical[:,guard.chain_indices[5]]=torch.tensor(pose)
        for act,full in zip(self.bundle.action_ids,self.bundle.active_action_indices):
            joint=int(self.bundle.model.actuator_trnid[act,0]);qpos[:,self.bundle.model.jnt_qposadr[joint]]=physical[:,full]
        assigned[0,5,39]=True
        requested=physical.clone();requested[:,guard.finger_indices[5]]=guard.high[5]
        applied,metrics=guard.filter(requested,actual_qpos=qpos,mocap_pos=mocap,assigned_keys=assigned)
        self.assertTrue(metrics['finger_active_force_permitted'][0,5])
        torch.testing.assert_close(applied[:,guard.finger_indices[5]],requested[:,guard.finger_indices[5]],atol=0,rtol=0)
        self.assertGreater(metrics['finger_requested_deficit_m'][0,5].item(),.01)
        self.assertLessEqual(metrics['finger_clearance_deficit_m'][0,5].item(),1e-6)
        actual=guard.evaluate(physical,guard.lift(mocap),assigned)
        refreshed=guard.reward_deficit(actual,metrics['finger_requested_deficit_m'],assigned)
        self.assertLessEqual(refreshed[0,5].item(),1e-6)

    def test_controls_can_be_disabled_without_disabling_geometry_feedback(self):
        guard=FingerClearanceGuard(self.bundle,device='cpu',options=dict(OPTIONS,
            active_command_mode='active_contact_v1', idle_finger_protection=False,
            active_press_permission=False, wrist_clearance_interlock=True))
        physical,qpos,mocap,assigned=self.inputs(2)
        physical[:,guard.finger_indices]=guard.high
        assigned[0,5,39]=True
        applied,metrics=guard.filter(physical,actual_qpos=qpos,mocap_pos=mocap,assigned_keys=assigned)
        torch.testing.assert_close(applied,physical,atol=0,rtol=0)
        self.assertGreater(metrics['finger_requested_deficit_m'].max().item(),.01)
        self.assertGreater(metrics['finger_clearance_deficit_m'][1].max().item(),.01)
        for name in ('finger_projection_mask','finger_fallback_mask','finger_recovery_mask',
                     'wrist_clearance_interlock_mask'):
            self.assertFalse(metrics[name].any(),name)

    def test_controls_apply_only_to_their_current_finger_role(self):
        physical,qpos,mocap,assigned=self.inputs(2)
        physical[:,self.guard.finger_indices]=self.guard.high
        assigned[:,5,39]=True
        for idle,active in ((True,False),(False,True)):
            with self.subTest(idle=idle,active=active):
                guard=FingerClearanceGuard(self.bundle,device='cpu',options=dict(OPTIONS,
                    active_command_mode='active_contact_v1', idle_finger_protection=idle,
                    active_press_permission=active))
                applied,metrics=guard.filter(physical,actual_qpos=qpos,mocap_pos=mocap,assigned_keys=assigned)
                untouched=[5] if idle else [0,1,2,3,4,6,7,8,9]
                ids=guard.finger_indices[untouched]
                torch.testing.assert_close(applied[:,ids],physical[:,ids],atol=0,rtol=0)
                self.assertFalse(metrics['finger_projection_mask'][:,untouched].any())
                self.assertFalse(metrics['finger_fallback_mask'][:,untouched].any())
                modified=[0,1,2,3,4,6,7,8,9] if idle else [5]
                self.assertTrue(metrics['finger_projection_mask'][:,modified].any())

    def test_contact_recovery_is_local_and_idle_interlock_has_no_workspace_crop(self):
        guard=FingerClearanceGuard(self.bundle,device='cpu',options=dict(OPTIONS,
            active_command_mode='active_contact_v1',wrist_clearance_interlock=True))
        physical,qpos,mocap,assigned=self.inputs(2)
        physical[:,guard.finger_indices[5]]=guard.high[5]
        for act,full in zip(self.bundle.action_ids,self.bundle.active_action_indices):
            joint=int(self.bundle.model.actuator_trnid[act,0]);qpos[:,self.bundle.model.jnt_qposadr[joint]]=physical[:,full]
        requested=physical.clone();requested[:,45]=.3
        assigned[0,5,39]=True  # First world is active danger; second is idle.
        before_q=qpos.clone()
        applied,metrics=guard.filter(requested,actual_qpos=qpos,mocap_pos=mocap,assigned_keys=assigned)
        self.assertTrue(metrics['finger_recovery_mask'][0,5])
        self.assertFalse(metrics['finger_active_force_permitted'][0,5])
        self.assertLessEqual((applied[0,guard.finger_indices[5]]-physical[0,guard.finger_indices[5]]).abs().max().item(),.200001)
        self.assertTrue(metrics['wrist_clearance_interlock_mask'][1,1])
        torch.testing.assert_close(applied[1,guard.wrist_indices[1]],physical[1,guard.wrist_indices[1]],atol=0,rtol=0)
        torch.testing.assert_close(qpos,before_q,atol=0,rtol=0)
        # Clear idle hand is free to request the same far-away TX value.
        clear,clear_q,clear_m,role=self.inputs()
        clear[:,45]=.3
        unblocked,info=guard.filter(clear,actual_qpos=clear_q,mocap_pos=clear_m,assigned_keys=role)
        self.assertFalse(info['wrist_clearance_interlock_mask'].any())
        self.assertEqual(unblocked[0,45].item(),clear[0,45].item())


if __name__ == '__main__': unittest.main()
