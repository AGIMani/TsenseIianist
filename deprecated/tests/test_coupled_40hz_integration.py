"""Real 40/400 Hz scene contract; run via submit.sh validate inside Slurm."""
import json
from pathlib import Path
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'Slurm CUDA allocation required')
class Coupled40HzIntegrationTest(unittest.TestCase):
    def test_full_observation_time_contract_and_reward_scale(self):
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
        from tests.test_hpc_40hz_entry import arguments
        root = Path(__file__).resolve().parents[1]
        # Preserve this original grid-contract regression; exact timing has its
        # own integration tests, even when the launch wrapper defaults change.
        args, _ = parser().parse_known_args(arguments('--no-exact-note-timing'))
        cfg = effective_config(args)
        recipe = make_recipe(control_hz=40, physics_hz=400, lookahead=20,
            task_profile='tsensei_collision_v1', drive_profile='shadow_v1', initial_buffer_time=0.,
            preparation_time=1., preparation_lift=.025, preparation_mode='policy_v1',
            release_tail_time=.25, score_validation='strict_v1')
        scene, task, _ = build(recipe)
        manifest = json.loads((scene/'manifest.json').read_text())
        self.assertEqual(manifest['contact_settings']['piano']['solref'][0], .005)
        self.assertEqual(manifest['physics_dt'], .0025)
        self.assertEqual(manifest['control_dt'], .025)
        scene, task = without_arms(scene, task, scene.parent/'hands_only')
        env = PianoTaskEnv(scene, task, num_envs=2, physics_graph=True,
            action_profile='direct_joint_v1', disable_wrist_tz=True, disable_wrist_yaw=True,
            preparation=recipe['preparation'], collision_diagnostics=True,
            reward_config=cfg['reward_config'], finger_clearance=cfg['finger_clearance'],
            finger_coupling=cfg['finger_coupling'], control_limits=cfg['control_limits'])
        self.assertEqual((env.action_size, env.observation_size), (31, 2121))
        self.assertEqual((env.preparation_steps, env.music_end_step, env.release_tail_steps), (40, 355, 10))
        self.assertEqual(len(env.logic.goal), 365)
        self.assertFalse(env.performance['anticipatory_current_priority'])
        self.assertLessEqual(float(env.future_seconds.max()), .150001)
        torch.testing.assert_close(env.scene.control_limiter.delta,
            torch.tensor([row['max_velocity']*.025 for row in env.scene.control_limits['channels'].values()],
                         device=env.device))
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (2, 2121))
        # The interval solver must honor changing inputs inside the CUDA graph.
        coupler = env.scene.finger_coupler
        self.assertTrue(coupler.range_mode)
        physical = env.scene.control_reset_action.expand(2, -1).clone()
        for offset in (0., .03, .2):
            physical[:, coupler.indices[:, 0]] += offset
            with env.scene.scope():
                expected, expected_info = coupler.apply(physical, actual_qpos=env.scene.data.qpos[:],
                                                        body_xmat=env.scene.data.xmat[:])
                actual, actual_info = env.scene._couple_finger_targets(physical)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            for name in expected_info:
                torch.testing.assert_close(actual_info[name], expected_info[name], atol=1e-6, rtol=1e-6)
        action = env.scene.policy_action(env.scene.control_reset_action).expand(2, -1)
        # Cross the preparation boundary and cover actual music frames.
        for _ in range(45):
            result = env.step(action)
            weighted = env.reward_manager.weighted_terms
            expected = sum(value*.5 for name,value in weighted.items()
                           if name != 'humanoid_finger_gesture_reward')
            expected += weighted['humanoid_finger_gesture_reward']
            torch.testing.assert_close(result.reward, expected)
            self.assertTrue(torch.isfinite(result.observation).all())
            angle = result.info['coupling_actual_angle_deg']
            expected_error = (60-angle).clamp_min(0)+(angle-80).clamp_min(0)
            torch.testing.assert_close(result.info['coupling_actual_error_deg'], expected_error)
            torch.testing.assert_close(env.scene.data.ctrl[:, env.scene.action_ids],
                result.info['applied_physical_action'][:, env.scene.physical_ids], atol=0, rtol=0)
        untouched = env.scene.data.qpos[1].clone()
        env.reset([0])
        torch.testing.assert_close(env.scene.data.qpos[1], untouched, atol=0, rtol=0)
