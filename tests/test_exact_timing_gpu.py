"""Full source-to-GPU timing validation; execute only through Slurm validate."""
import os
from pathlib import Path
import unittest
import numpy as np
import torch


@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required')
class ExactTimingGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Slurm validation requires allocated CUDA GPUs')
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        cls.recipe=make_recipe(control_hz=40,physics_hz=400,lookahead=20,
            task_profile='tsensei_collision_v1',drive_profile='shadow_v1',initial_buffer_time=0.,
            preparation_time=1.,preparation_lift=.025,preparation_mode='policy_v1',
            release_tail_time=.25,score_validation='strict_v1',exact_note_timing=True)
        scene,task,_=build(cls.recipe)
        cls.scene,cls.task=without_arms(scene,task,scene.parent/'hands_only')

    def test_full_episode_variants_and_exact_physical_counts(self):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config,evaluate
        root=Path(__file__).resolve().parents[1]
        for extra in ((),('--staged-fingering-reward',)):
            with self.subTest(extra=extra):
                args=parser().parse_args(['--exact-note-timing', '--finger-coupling',
                    '--finger-coupling-angle-ranges',str(root/'configs/finger_coupling/angle_ranges_60_80.json'),
                    '--reward-config',str(root/'configs/rewards/default_20261006.json'),*extra])
                cfg=effective_config(args)
                env=PianoTaskEnv(self.scene,self.task,num_envs=2,physics_graph=True,
                    action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
                    preparation=self.recipe['preparation'],collision_diagnostics=True,
                    reward_config=cfg['reward_config'],finger_clearance=cfg['finger_clearance'],
                    finger_coupling=cfg['finger_coupling'],control_limits=cfg['control_limits'])
                self.assertEqual((env.action_size,env.observation_size),(31,2297))
                events=env.bundle.exact_note_events
                self.assertEqual(len(events),34)
                torch.testing.assert_close(env.logic.goal, torch.tensor(env.bundle.goal, device=env.device))
                torch.testing.assert_close(env.logic.fingers, torch.tensor(env.bundle.fingers, device=env.device))
                self.assertAlmostEqual(events[:,0].min(),1.)
                self.assertEqual(env.reset().shape,(2,2297))
                action=env.scene.policy_action(env.scene.control_reset_action).expand(2,-1)
                class ConstantAgent:
                    device=env.device
                    def act(self,obs,deterministic=True):
                        return action
                metrics=evaluate(ConstantAgent(),env,episodes=2)
                self.assertEqual(metrics['completed_episodes'],2)
                self.assertTrue(metrics['substep_key_diagnostics_valid'])
                self.assertTrue(metrics['note_timing_available'])
                self.assertEqual(metrics['note_timing_reference_count'],68)
                self.assertEqual(metrics['note_timing_contract']['schema'],'exact_note_events_v1')
                expected=int(env.exact_score.goal_history[:,:,:88].sum().item())
                for row in metrics['episode_statistics']:
                    self.assertEqual(row['substep_true_positive_count']+row['substep_false_negative_count'],expected)
                    self.assertTrue(np.isfinite(row['return']))
                    self.assertEqual(row['finger_clearance_projection_frames'],0)
                env.reset()
                env.step(action)
                untouched=env.scene.data.qpos[1].clone()
                env.reset([0])
                torch.testing.assert_close(env.scene.data.qpos[1],untouched,atol=0,rtol=0)
                # Exercise a different local score clock while graphs are reused.
                result=env.step(action)
                self.assertEqual(result.info['score_index'].tolist(),[0,1])
                self.assertTrue(torch.isfinite(result.observation).all())
