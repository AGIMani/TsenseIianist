"""Run only inside a Slurm GPU allocation: graph, physical action and reset contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'Slurm CUDA allocation required')
class CouplingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        root = Path(__file__).resolve().parents[1]
        recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1', initial_buffer_time=0.,
                             preparation_time=1., preparation_mode='policy_v1')
        scene, task, _ = build(recipe)
        scene, task = without_arms(scene, task, scene.parent/'coupling_hand_only')
        cls.scene_path, cls.task_path, cls.recipe = scene, task, recipe
        clearance = json.loads((root/'configs/finger_clearance/generic_20260927.json').read_text())
        clearance.update(idle_finger_protection=False, active_press_permission=False)
        reward = json.loads((root/'configs/rewards/generic_finger_clearance_20260927.json').read_text())
        reward.setdefault('performance', {})['anticipatory_fingering_time'] = .15
        cls.arguments = dict(num_envs=2, physics_graph=True, action_profile='direct_joint_v1',
            disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'],
            finger_clearance=clearance, reward_config=reward,
            finger_coupling=dict(schema=1, mode='distal_60deg_v1', reduce_observation=True),
            control_limits=dict(schema=1, mode='target_clip_slew_v1', channels={
                'R_Wrist_TX': {'max_velocity': 1.}, 'R_Wrist_TY': {'max_velocity': 1.},
                'L_Wrist_TX': {'max_velocity': 1.}, 'L_Wrist_TY': {'max_velocity': 1.}}))
        cls.env = PianoTaskEnv(scene, task, **cls.arguments)

    def test_graph_exact_targets_and_changed_live_inputs(self):
        env = self.env; scene = env.scene; coupler = scene.finger_coupler
        env.reset()
        physical = scene.control_reset_action.expand(2, -1).clone()
        for offset in (.0, .03, .2):
            physical[:, coupler.indices[:, 0]] += offset
            with scene.scope():
                expected, expected_info = coupler.apply(physical, actual_qpos=scene.data.qpos[:], body_xmat=scene.data.xmat[:])
                actual, actual_info = scene._couple_finger_targets(physical)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            for name in expected_info:
                torch.testing.assert_close(actual_info[name], expected_info[name], atol=1e-6, rtol=1e-6)
        self.assertEqual(scene.action_size, 31)
        self.assertEqual(env.observation_size, 1199)
        self.assertEqual((scene.model.nq, scene.model.nu), (134, 46))

    def test_physical_followers_energy_observation_and_partial_reset(self):
        env = self.env; scene = env.scene
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (2, 1199))
        policy = scene.policy_action(scene.control_reset_action).expand(2, -1)
        result = env.step(policy)
        applied = result.info['applied_physical_action']
        torch.testing.assert_close(scene.data.ctrl[:, scene.action_ids], applied[:, scene.physical_ids], atol=0, rtol=0)
        torch.testing.assert_close(env.logic.last_action, applied, atol=0, rtol=0)
        base = env.logic.observation()[:, env._coupling_observation_ids]
        torch.testing.assert_close(result.observation[:, :base.shape[1]], base)
        self.assertEqual(tuple(result.info['coupling_active'].shape), (2, 8))
        self.assertEqual(tuple(env.snapshot()['signed_power'].shape), (2, 50))
        qpos = scene.data.qpos[1].clone(); ctrl = scene.data.ctrl[1].clone()
        env.reset([0])
        torch.testing.assert_close(scene.data.qpos[1], qpos, atol=0, rtol=0)
        torch.testing.assert_close(scene.data.ctrl[1], ctrl, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(result.reward).all())

    def test_full_observation_optional_and_incompatible_controls_rejected(self):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        args = dict(self.arguments)
        args['finger_coupling'] = dict(args['finger_coupling'], reduce_observation=False)
        env = PianoTaskEnv(self.scene_path, self.task_path, **args)
        self.assertEqual((env.action_size, env.observation_size), (31, 1231))
        args['control_limits'] = dict(schema=1, mode='target_clip_slew_v1', channels={'R_Index_PIP': {'max_velocity': 1.}})
        with self.assertRaisesRegex(ValueError, 'PIP/DIP'):
            PianoTaskEnv(self.scene_path, self.task_path, **args)
        args = dict(self.arguments)
        args['finger_clearance'] = dict(args['finger_clearance'], idle_finger_protection=True)
        with self.assertRaisesRegex(ValueError, 'projection controls'):
            PianoTaskEnv(self.scene_path, self.task_path, **args)

    def test_real_31_action_sac_replay_and_checkpoint_roundtrip_both_observations(self):
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch import checkpoint
        from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
        from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
        for reduced, expected_size in ((True, 1199), (False, 1231)):
            with self.subTest(reduce_observation=reduced):
                args = dict(self.arguments)
                args['finger_coupling'] = dict(args['finger_coupling'], reduce_observation=reduced)
                env = self.env if reduced else PianoTaskEnv(self.scene_path, self.task_path, **args)
                scene = env.scene
                self.assertEqual((env.action_size, env.observation_size), (31, expected_size))
                config = SACConfig(actor_hidden=(32, 32), critic_hidden=(32, 32), critic_dropout=0.)
                agent = SAC(expected_size, 31, config, device=str(env.device), seed=17,
                            action_profile='direct_joint_v1')
                observation = env.reset()
                action = agent.act(observation)
                result = env.step(action)
                replay = Replay(expected_size, 31, 16, 2, device=str(env.device),
                                action_profile='direct_joint_v1')
                replay.insert(Transition(observation, action, result.reward, result.discount,
                                         result.observation, result.terminated, result.truncated))
                metrics = agent.update(replay.sample())
                self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
                self.assertEqual(tuple(replay.data.action.shape), (16, 31))
                self.assertEqual(tuple(scene.data.ctrl[:, scene.action_ids].shape), (2, 46))
                metadata = dict(scene_sha256=sha256(self.scene_path/'manifest.json'),
                    task_sha256=sha256(self.task_path/'task.json'), stage=1,
                    action_profile='direct_joint_v1', action_contract=scene.action_contract,
                    disable_wrist_tz=True, disable_wrist_yaw=True, wrist_tz_physics='joint_removed_v1',
                    reward_config=env.reward_config, preparation=self.recipe['preparation'],
                    preparation_contract=scene.bundle.preparation,
                    control_limits=scene.control_limits_options, control_limits_contract=scene.control_limits,
                    finger_clearance=scene.finger_clearance_options, finger_clearance_contract=scene.finger_clearance,
                    finger_coupling=scene.finger_coupling_options, finger_coupling_contract=scene.finger_coupling,
                    coupling_observation_contract=env.coupling_observation_contract)
                generator = torch.Generator(device=env.device).manual_seed(11)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'coupled.pt'
                    checkpoint.save(path, agent, metadata, generator)
                    restored = checkpoint.load_agent(path, scene_sha256=metadata['scene_sha256'],
                        task_sha256=metadata['task_sha256'], obs_size=expected_size, action_size=31,
                        device=str(env.device), action_contract=scene.action_contract,
                        preparation_contract=scene.bundle.preparation,
                        control_limits_contract=scene.control_limits,
                        finger_clearance_contract=scene.finger_clearance,
                        finger_coupling_contract=scene.finger_coupling,
                        coupling_observation_contract=env.coupling_observation_contract,
                        **checkpoint.reward_options(path))
                    torch.testing.assert_close(restored.act(result.observation, deterministic=True),
                        agent.act(result.observation, deterministic=True), atol=0, rtol=0)
                    self.assertEqual(restored.updates, agent.updates)
                    checkpoint.load(path, restored, metadata, exploration_rng=generator)
                    changed = dict(metadata, coupling_observation_contract=dict(
                        env.coupling_observation_contract, reduce_observation=not reduced))
                    with self.assertRaisesRegex(ValueError, 'coupling'):
                        checkpoint.load(path, restored, changed)


if __name__ == '__main__':
    unittest.main()
