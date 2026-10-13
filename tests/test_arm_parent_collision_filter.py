"""Rigid hand conversion preserves native contacts, not just reward filtering."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from tests import test_optional_arm_follow as arm_tests
from tsensei_v3_2_pianist.mjlab_env.arm_scene import (
    SceneArmFollower, _restore_parent_collision_filter, augment_bundle,
)
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent


def contact_pairs(model, data, *, same_hand=False):
    pairs = set()
    for contact in data.contact:
        if contact.dist > 0:
            continue
        geoms = [int(i) for i in contact.geom]
        names = [model.body(int(model.geom_bodyid[i])).name for i in geoms]
        if same_hand and not any(all(n.startswith(side + '_/') for n in names)
                                 for side in ('right', 'left')):
            continue
        pairs.add(tuple(sorted(model.geom(i).name for i in geoms)))
    return pairs


class ArmParentFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scene, cls.task, _ = arm_tests.build(arm_tests.make_recipe(
            initial_buffer_time=0, preparation_time=1., preparation_lift=.025,
            preparation_mode='policy_v1', task_profile='tsensei_collision_v1',
            drive_profile='shadow_v1', exact_note_timing=True))

    def bundle(self, arms=True):
        return SceneBundle(self.scene, disable_wrist_tz=True, disable_wrist_yaw=True,
                           preparation=arm_tests.PREP, arm_follow=arm_tests.options() if arms else None)

    def test_reset_and_varied_finger_poses_preserve_physical_contact_pairs(self):
        baseline, rigid = self.bundle(False), self.bundle()
        models = baseline.model, rigid.model
        states = baseline.reset_cpu(), rigid.reset_cpu()
        restored = rigid.arm_follow_spec['parent_collision_filter']['restored_body_pairs']
        self.assertEqual(len(restored), 20)  # Full adjacent groups, not just the 12 reset overlaps.
        random = np.random.default_rng(103)
        detected_nonadjacent = set()
        detected_keys = set()
        for pose in range(12):
            if pose:
                for j in range(rigid.model.njnt):
                    name = rigid.model.joint(j).name
                    if name.startswith(('right_/', 'left_/')):
                        value = random.uniform(*rigid.model.jnt_range[j])
                        states[1].qpos[rigid.model.jnt_qposadr[j]] = value
                        old = baseline.model.joint(name).id
                        states[0].qpos[baseline.model.jnt_qposadr[old]] = value
            # Translate both complete hands identically to exercise key contacts
            # as well as nonadjacent finger contacts; no policy/IK is involved.
            for lowering in (0., .06):
                for bundle, data in zip((baseline, rigid), states):
                    data.mocap_pos[:] = bundle.reset_mocap_pos
                    data.mocap_pos[:, 2] -= lowering
                    mujoco.mj_forward(bundle.model, data)
                before, after = [contact_pairs(m, d) for m, d in zip(models, states)]
                with self.subTest(pose=pose, lowering=lowering):
                    self.assertEqual(before, after)
                own = contact_pairs(rigid.model, states[1], same_hand=True)
                if pose == 0 and lowering == 0:
                    self.assertEqual(own, set())
                detected_nonadjacent.update(own)
                detected_keys.update(pair for pair in after if any(n.startswith('piano/') for n in pair)
                                     and any(n.startswith(('left_/', 'right_/')) for n in pair))
        self.assertTrue(detected_nonadjacent, 'Nonadjacent hand collision must remain active')
        self.assertTrue(detected_keys, 'Hand/key collision must remain active')

    def test_disabled_parent_filter_remains_disabled(self):
        baseline = self.bundle(False)
        baseline.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        before = contact_pairs(baseline.model, baseline.reset_cpu(), same_hand=True)
        augment_bundle(baseline, arm_tests.options())
        contract = baseline.arm_follow_spec['parent_collision_filter']
        self.assertFalse(contract['source_enabled'])
        self.assertEqual(contract['restored_body_pairs'], [])
        self.assertTrue(before)
        self.assertEqual(contact_pairs(baseline.model, baseline.reset_cpu(), same_hand=True), before)

    def test_explicit_geom_pair_keeps_precedence_over_restored_body_exclusion(self):
        # The native parent filter does not apply to explicit geom pairs. A
        # preserved body exclusion must not accidentally remove that override.
        body = '''<body name="root" {root}><geom name="root_g" size=".1"/>
          <body name="child"><joint type="slide"/><geom name="child_g" size=".1"/></body>
        </body>'''
        pair = '<contact><pair geom1="root_g" geom2="child_g"/></contact>'
        old = mujoco.MjModel.from_xml_string('<mujoco><worldbody>' +
            body.format(root='').replace('<geom name="root_g"', '<joint type="slide"/><geom name="root_g"') +
            '</worldbody>' + pair + '</mujoco>')
        xml = ET.fromstring('<mujoco><worldbody>' + body.format(root='mocap="true"') +
                            '</worldbody>' + pair + '</mujoco>')
        new = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        self.assertEqual(_restore_parent_collision_filter(old, new, xml.find('contact')), [['child', 'root']])
        fixed = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        for model in (old, fixed):
            data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            self.assertEqual(contact_pairs(model, data), {('child_g', 'root_g')})

    def test_checkpoint_without_resolved_filter_contract_is_rejected(self):
        bundle = self.bundle()
        current = SceneArmFollower(bundle, num_envs=1, device='cpu').contract
        historical = copy.deepcopy(current)
        historical.pop('parent_collision_filter')
        payload = dict(schema=1, kind='learner_warm_restart', metadata=dict(
            arm_follow=arm_tests.options(), arm_follow_contract=historical))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old_filter.pt'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'arm follower/observation contract mismatch'):
                load_agent(path, scene_sha256='scene', task_sha256='task', obs_size=1461,
                    action_size=31, device='cpu', arm_follow=arm_tests.options(), arm_follow_contract=current)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'GPU contact regression only inside explicitly selected Slurm validation')
    def test_gpu_physics_contacts_and_substep_reward_use_restored_filter(self):
        # The real task uses the same constructor as the existing GPU control,
        # cross-hand contact and local-reset regressions.
        env = arm_tests.ArmNativeSceneTests.gpu_env(self)
        scene = env.scene
        excluded = {tuple(pair) for pair in scene.arm_follower.contract['parent_collision_filter']['restored_body_pairs']}
        samples = []

        def sample(s):
            count = int(s.data.nacon[0])
            geom = s.data.contact.geom[:count].cpu().numpy()
            dist = s.data.contact.dist[:count].cpu().numpy()
            worlds = s.data.contact.worldid[:count].cpu().numpy()
            actual = [set() for _ in range(env.num_envs)]
            for ids, gap, world in zip(geom, dist, worlds):
                if gap > 0:
                    continue
                names = tuple(sorted(s.model.body(int(s.model.geom_bodyid[int(i)])).name for i in ids))
                self.assertNotIn(names, excluded, 'Excluded parent contact reached the physical solver')
                if any(all(n.startswith(side + '_/') for n in names) for side in ('right', 'left')):
                    actual[int(world)].add(names)
            samples.append([len(pairs) for pairs in actual])

        # Validate a real task step's reward first, then count the main contact
        # buffer at each 200Hz substep of a complete 20Hz control cycle.
        initial = scene.policy_action(scene.last_applied_physical_action)
        result = env.step(initial)
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertTrue(torch.all(result.info['reward_weighted_terms']['self_collision_reward'] == 0))
        scene.step(initial, substep_callback=sample,
                   assigned_keys=env.logic.fingers[env.logic.index.clamp_max(env.logic.length-1)])
        self.assertEqual(len(samples), scene.decimation)
        torch.testing.assert_close(scene.self_collision.count_pair_substeps(('self',)),
                                   torch.tensor(np.sum(samples, axis=0), device=scene.device))
        self.assertEqual(np.sum(samples), 0)
        print('rigid_neutral_parent_filter', dict(restored_pairs=len(excluded),
              self_pair_substeps=np.sum(samples, axis=0).tolist(),
              weighted_reward=result.info['reward_weighted_terms']['self_collision_reward'].cpu().tolist()))

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('TSENSEI_ARM_GPU_TEST') == '1',
                         'GPU self-collision regression only inside explicitly selected Slurm validation')
    def test_gpu_noarms_and_arms_keep_nonadjacent_contacts_and_penalty(self):
        from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
        for arms in (False, True):
            with self.subTest(arms=arms):
                env = arm_tests.ArmNativeSceneTests.gpu_env(self, None if arms else False)
                s = env.scene
                term = env.reward_config['terms']['self_collision_reward']
                self.assertEqual(term, dict(enabled=True, weight=.2, params=dict(aggregation='substep_sum', maximum_penalty_per_step=5.)))
                random = np.random.default_rng(103)
                with s.scope():
                    for j in range(s.model.njnt):
                        if s.model.joint(j).name.startswith(('right_/', 'left_/')) and '_/forearm_' not in s.model.joint(j).name:
                            s.data.qpos[:, int(s.model.jnt_qposadr[j])] = float(random.uniform(*s.model.jnt_range[j]))
                    s.sim.forward()
                    s.self_collision.clear()
                    for _ in range(s.decimation):
                        s._sample_self_collision(count_substep=True)
                    once = s.self_collision.count(('self',))
                    total = s.self_collision.count_finger_pair_substeps()
                    self.assertTrue(torch.all(once > 0), 'Physical nonadjacent self-collision was suppressed')
                    torch.testing.assert_close(s.self_collision.count_pair_substeps(), once * s.decimation)
                    self.assertTrue(torch.all(total <= once * s.decimation))
                    config = copy.deepcopy(env.reward_config)
                    for name, row in config['terms'].items():
                        row['enabled'] = name == 'self_collision_reward'
                    manager = RewardManager(config, s.bundle.manifest, env.reward_manager.task_manifest,
                                            num_envs=env.num_envs, device=s.device)
                    result = manager({}, s.last_applied_physical_action,
                        dict(self_collision_pairs=once, self_collision_pair_substeps=total),
                        torch.zeros((env.num_envs, 89), device=s.device),
                        torch.zeros((env.num_envs, 10, 88), device=s.device))
                    torch.testing.assert_close(result['self_collision_reward'], (-.2 * total.float()).clamp_min(-5.))
                    self.assertTrue(torch.all(result['self_collision_reward'] < 0))
                    print('nonadjacent_collision_preserved', dict(arms=arms, body_pairs=once.cpu().tolist(),
                          pair_substeps=total.cpu().tolist(), weighted_reward=result['self_collision_reward'].cpu().tolist()))


if __name__ == '__main__':
    unittest.main()
