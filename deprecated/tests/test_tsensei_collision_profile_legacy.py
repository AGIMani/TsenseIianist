"""Full hand coverage, transient contact scoring and legacy compatibility."""
import json
from pathlib import Path
import unittest

import torch

from tsensei_v3_2_pianist.forearm_collision import (
    collision_scope, forearm_contact_reward, is_forearm_collision_geom, task_collision_groups)
from tsensei_v3_2_pianist.fingering_targets import manifest_config, target_position
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
from tests.test_mjlab_task import fixture, snapshot
from tests.test_paper_rewards import only, scene_fixture
from tests.test_hpc_40hz_entry import arguments
from tests.test_twinkle_20hz_experiments import resolved_source_fields

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'tsensei_collision_v1'


class TsenseiCollisionProfileTest(unittest.TestCase):
    def test_all_hand_parts_and_uniform_targets(self):
        for side in ('right', 'left'):
            for part in ('base_link', f'{side.capitalize()}_hand_palm_link',
                         f'{side}_hand_thumb_rotaback_link2', f'{side}_hand_index_swing_link',
                         f'{side}_hand_pinky_DIP_link'):
                self.assertTrue(is_forearm_collision_geom(f'{side}_/{part}_collision_0', PROFILE))
                self.assertFalse(is_forearm_collision_geom(f'{side}_/{part}_visual_0', PROFILE))
        with self.assertRaises(ValueError):
            collision_scope('shadow_v1')
        with self.assertRaisesRegex(ValueError, 'archived source'):
            task_collision_groups({'palm_geoms': [['palm'], ['palm']]})
        self.assertIsNone(manifest_config(PROFILE)['thumb'])
        for thumb in (True, False):
            args = ([0., 0., 0.], [.075, .01125, .01125], 'piano/white_key_geom_0')
            self.assertEqual(target_position(*args, thumb=thumb, task_profile=PROFILE).tolist(),
                             target_position(*args, thumb=False, task_profile=PROFILE).tolist())

    def test_endpoint_reward_is_independent_of_diagnostic_contact_union(self):
        for profile, expected in ((PROFILE, [0., -.5]), ('tsensei_v1', [.5, 0.])):
            bundle = fixture(); bundle.manifest['task_profile'] = profile
            logic = MusicTaskLogic(bundle, num_envs=2, device='cpu')
            sample = snapshot(2)
            sample['palm_collision'] = torch.tensor([False, True])
            base, _ = logic.reward_terms(sample, logic.index)
            torch.testing.assert_close(base['forearm_reward'], torch.tensor(expected))
            self.assertEqual([forearm_contact_reward(x, profile) for x in (False, True)], expected)
            # Earlier transient contact remains diagnostic, not a public reward sample.
            sample['palm_collision_any'] = torch.tensor([True, False])
            cfg = only('forearm_reward'); cfg['terms']['forearm_reward']['weight'] = 1.
            manager = RewardManager(cfg, scene_fixture(), bundle.manifest, num_envs=2, device='cpu')
            result = manager(base, torch.zeros(2, 51), sample, logic.goal[:2], logic.fingers[:2])
            torch.testing.assert_close(result['forearm_reward'], torch.tensor(expected))

    def test_current_entry_and_alias_preserve_cli_overrides(self):
        fields = resolved_source_fields(arguments())
        self.assertEqual(fields['task_profile'], PROFILE)
        self.assertEqual(fields['drive_profile'], 'shadow_v1')
        self.assertEqual(fields['collision_profile'], 'shadow_primitive_v1')
        self.assertEqual(arguments(), arguments(entry=str(ROOT/'run_tsensei_a32.sh')))
        self.assertEqual(resolved_source_fields(arguments('--task-profile', 'tsensei_v1'))['task_profile'],
                         'tsensei_v1')
        script = (ROOT/'scripts/hpc/train_coupled_20hz.sh').read_text()
        self.assertIn('exec python3 scripts/train_from_source.py', script)
        self.assertNotIn('exec bash \"$ROOT/run.sh\"', script)
        self.assertNotIn('exec bash \"$ROOT/run_tsensei_a32.sh\"', script)
        self.assertIn('ARCHIVED', (ROOT/'run.sh').read_text())


@unittest.skipUnless(torch.cuda.is_available(), 'Slurm CUDA allocation required')
class TsenseiCollisionGpuTest(unittest.TestCase):
    def test_compiled_coverage_and_transient_contact_union(self):
        import warp as wp
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

        argv = arguments()
        recipe = make_recipe(**resolved_source_fields(argv))
        scene, task, _ = build(recipe)
        args, _ = parser().parse_known_args(argv)
        config = effective_config(args)
        # Exercise actual current model/coupling/reward configuration in a
        # small batch; no SAC update or long training is needed for this test.
        env = PianoTaskEnv(scene, task, num_envs=8, physics_graph=False,
            action_profile=args.action_profile, disable_wrist_tz=True, disable_wrist_yaw=True,
            preparation=recipe['preparation'], reward_config=config['reward_config'],
            control_limits=config['control_limits'], finger_coupling=config['finger_coupling'],
            finger_clearance=config['finger_clearance'], collision_diagnostics=True)
        model = env.scene.model
        groups = env.bundle.manifest['forearm_collision_geoms']
        self.assertEqual(list(map(len, groups)), [22, 22])
        physical = [{model.geom(i).name for i in range(model.ngeom)
                     if (model.geom(i).name or '').startswith(side+'_/')
                     and (model.geom_contype[i] or model.geom_conaffinity[i])}
                    for side in ('right', 'left')]
        self.assertEqual([set(g) for g in groups], physical)
        for group in groups:
            for name in group:
                if '_DIP_link_' in name or 'thumb_PIP_link_' in name:
                    radius = model.geom_size[model.geom(name).id, 0]
                    self.assertGreater(radius, .0058)
                    self.assertLess(radius, .0065)
        # Full-scope manifests must not accept stale palm-only lists.
        path = task/'task.json'; original = path.read_text()
        damaged = json.loads(original)
        damaged['forearm_collision_geoms'] = damaged['palm_geoms']
        try:
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, 'all physical hand'):
                TaskBundle(task, env.scene.bundle)
        finally:
            path.write_text(original)
        rp, lp = [model.geom(group[0]).id for group in groups]
        rf, lf = [model.geom(group[-1]).id for group in groups]
        key = model.geom(env.bundle.manifest['key_geoms'][0]).id
        # palm/palm, finger/finger, finger/palm, same hand, hand/key,
        # separated pair, reversed palm/finger, multiple contacts in world 0.
        pairs = [(rp, lp), (rf, lf), (rf, lp), (rf, rp), (rf, key), (rf, lf), (lf, rp), (rf, lf)]
        raw = env.scene.sim.wp_data
        wp.to_torch(raw.contact.geom)[:8] = torch.tensor(pairs, device=env.device, dtype=torch.int32)
        wp.to_torch(raw.contact.worldid)[:8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0], device=env.device, dtype=torch.int32)
        wp.to_torch(raw.contact.dist)[:8] = -.001
        wp.to_torch(raw.contact.dist)[5] = .01
        wp.to_torch(raw.nacon)[0] = 8
        collector = env.scene.reward_sampling
        collector.clear()
        for _ in range(10):
            collector.sample_contacts(env.scene.sim)
        wp.to_torch(raw.nacon)[0] = 0
        collector.sample_contacts(env.scene.sim)
        sample = env.snapshot(endpoint=True)
        expected = [True, True, True, False, False, False, True, False]
        self.assertEqual(sample['palm_collision'].tolist(), [False]*8)
        self.assertEqual(sample['palm_collision_any'].tolist(), expected)
        base, _ = env.logic.reward_terms(sample, env.logic.index)
        result = env.reward_manager(base, torch.zeros(8, 51, device=env.device), sample,
            env.logic.goal[env.logic.index], env.logic.fingers[env.logic.index])
        torch.testing.assert_close(result['forearm_reward'],
            torch.tensor([-.5 if c else 0. for c in expected], device=env.device))
        # Unified single-GPU S4 entry keeps params={} and relies on the new
        # task profile's base/substep rule, not the older HPC penalty override.
        s4 = json.loads((ROOT/'configs/experiments/twinkle_20hz_20261004/S4_false01.json').read_text())
        s4_manager = RewardManager(s4, env.scene.bundle.manifest, env.bundle.manifest,
                                   num_envs=8, device=env.device)
        s4_result = s4_manager(base, torch.zeros(8, 51, device=env.device), sample,
            env.logic.goal[env.logic.index], env.logic.fingers[env.logic.index])
        torch.testing.assert_close(s4_result['forearm_reward'], result['forearm_reward'])
        collector.clear([1])
        self.assertFalse(collector.palm_collision_any[1])
        self.assertTrue(collector.palm_collision_any[0])
        # Restore genuine contacts; run real physical control cycles.
        env.reset()
        for _ in range(3):
            out = env.step(env.scene.action_shift.expand(8, -1).clone())
            value = out.info['reward_terms']['forearm_reward']
            self.assertTrue(torch.all((value == 0) | (value == -.5)))
            self.assertTrue(torch.isfinite(out.reward).all())
        # Six seconds including policy preparation: hold the actual origin
        # controls, then small bounded target excursions. This checks solver
        # execution/finite state after widening all ten terminal colliders;
        # it is not a trained-policy or collision-free-motion acceptance.
        env.reset()
        hold = env.scene.policy_action(env.scene.control_reset_action).expand(8, -1).clone()
        maximum_velocity = 0.
        maximum_position = 0.
        hand_contact_steps = 0
        for step in range(120):
            action = hold.clone()
            if step >= 60:
                action[:, :-1] += .05 * __import__('math').sin(step * .2)
            action = torch.maximum(torch.minimum(action, env.scene.action_high), env.scene.action_low)
            out = env.step(action)
            self.assertTrue(torch.isfinite(out.reward).all())
            self.assertTrue(torch.isfinite(env.scene.data.qpos[:]).all())
            self.assertTrue(torch.isfinite(env.scene.data.qvel[:]).all())
            maximum_velocity = max(maximum_velocity, env.scene.data.qvel[:].abs().max().item())
            maximum_position = max(maximum_position, env.scene.data.qpos[:].abs().max().item())
            hand_contact_steps += int(env.snapshot()['palm_collision_any'].any().item())
        print(json.dumps(dict(terminal_rollout_control_steps=120, worlds=8,
            maximum_absolute_qvel=maximum_velocity, maximum_absolute_qpos=maximum_position,
            control_steps_with_any_cross_hand_contact=hand_contact_steps)))


if __name__ == '__main__':
    unittest.main()
