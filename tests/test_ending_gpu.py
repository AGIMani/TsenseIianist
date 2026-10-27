"""Slurm-only integration: real prefix terminal, local reset and policy handoff."""
import os
import unittest
import json
from pathlib import Path
import numpy as np

@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'),'Requires a Slurm GPU allocation')
class EndingGPUChecks(unittest.TestCase):
    def test_actual_terminal_reset_budget_and_first_control_step(self):
        import torch
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[1]
        recipe=make_recipe(height=.09,ty=.04,yaw=.2617993877991494,control_hz=20,physics_hz=1000,
            midi=str(root/'training_mjlab/inputs/turkish_march_advanced/a32_contact_refined_20261024/turkish-march-advanced_fingered.proto'),
            initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',
            task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
            preparation_mode='first_note_v1',preparation_time=2,preparation_lift=0,ending_time=2)
        scene,task,_=build(recipe);scene,task=without_arms(scene,task,scene.parent/'no_arms')
        cfg=effective_config(parser().parse_args(['--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
            '--action-profile','direct_joint_v1','--finger-coupling','--no-coupling-reduce-observation',
            '--finger-coupling-angle-ranges',str(root/'configs/finger_coupling/angle_ranges_10_80.json'),
            '--preparation-mode','first_note_v1','--preparation-time','2','--preparation-lift','0','--ending-time','2',
            '--reward-config',str(root/'configs/rewards/default_20261006.json'),
            '--hard-joint-limits','--wrist-tx-max-velocity','1.99','--wrist-ty-max-velocity','1.99',
            '--wrist-tx-max-acceleration','1.99','--wrist-ty-max-acceleration','1.99',
            '--wrist-pitch-max-velocity','3.1416','--wrist-pitch-max-acceleration','15.708']))
        env=PianoTaskEnv(scene,task,num_envs=2,device='cuda:0',physics_graph=False,
            disable_wrist_tz=True,disable_wrist_yaw=True,
            **{k:cfg[k] for k in ('preparation','finger_coupling','control_limits','reward_config','action_profile')})
        b=env.scene.bundle
        env.reset()
        self.assertEqual(env.preparation_steps,0)
        self.assertFalse(b.preparation['included_in_replay'])
        self.assertTrue(b.preparation['report']['valid'])
        self.assertEqual(env.logic.length,env.bundle.manifest['length'])
        np.testing.assert_array_equal(b.reset_qpos,b.preparation_trace['qpos'][-1])
        np.testing.assert_array_equal(env.scene.data.qpos.cpu().numpy()[0],b.reset_qpos.astype(np.float32))
        np.testing.assert_array_equal(env.scene.data.qvel.cpu().numpy()[0],b.reset_qvel.astype(np.float32))
        np.testing.assert_allclose(env.reward_manager.neutral.cpu().numpy(),b.neutral_action[:50],atol=1e-7)
        action=env.scene.policy_action(env.scene.control_reset_action)[None].repeat(2,1)
        limiter=env.scene.target_motion_limiter
        torch.testing.assert_close(limiter.previous[0],limiter.reset_target,atol=0,rtol=0)
        self.assertEqual(torch.count_nonzero(limiter.velocity).item(),0)
        previous=limiter.previous.clone()
        result=env.step(action)
        speed=(limiter.previous-previous)/limiter.dt
        self.assertTrue((speed.abs()<=limiter.max_velocity+1e-7).all())
        self.assertTrue((speed.abs()/limiter.dt<=limiter.max_acceleration+1e-6).all())
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertEqual(env.scene.control_steps.tolist(),[1,1])
        from tsensei_v3_2_pianist.mjlab_env.ending import generate,append_trace
        from tsensei_v3_2_pianist.mjlab_env.first_note_preparation import PreparationFailure
        import mujoco
        output=Path(os.environ.get('ENDING_GPU_OUTPUT','results/ending_validation'))
        output.mkdir(parents=True,exist_ok=True)
        initial=env.scene.cpu_snapshot()
        initial.qacc_warmstart[:]=env.scene.data.qacc_warmstart[0].cpu().numpy()
        saved=env.scene.data.qpos.clone()
        try:
            contract,trace=generate(b,initial,b.preparation['control_context'],2.)
        except PreparationFailure as error:
            contract,trace=error.contract,error.trace
            np.savez_compressed(output/'ending.npz',**trace)
            (output/'ending.json').write_text(json.dumps(contract,indent=2)+'\n')
            raise
        np.savez_compressed(output/'ending.npz',**trace)
        (output/'ending.json').write_text(json.dumps(contract,indent=2)+'\n')
        mujoco.mj_saveModel(b.model,str(output/'model.mjb'))
        torch.testing.assert_close(env.scene.data.qpos[:],saved,atol=0,rtol=0)
        self.assertTrue(contract['report']['valid'])
        for name in ('qpos','qvel','ctrl'):
            np.testing.assert_array_equal(trace[name][0],getattr(initial,name))
        np.testing.assert_allclose(trace['ctrl'][-1],b.origin_ctrl,atol=1e-12)
        self.assertFalse(contract['included_in_replay'])
        self.assertEqual(env.scene.control_steps.tolist(),[1,1])
        raw={k:trace[k][:1] for k in ('qpos','qvel','mocap_pos','mocap_quat')}
        raw['sustain']=np.zeros(1)
        self.assertEqual(len(append_trace(raw,trace,b.model.opt.timestep)['qpos']),len(trace['time']))
