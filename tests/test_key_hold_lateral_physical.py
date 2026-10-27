"""A3.2 runtime integration on remote Slurm; no policy training."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TERM='key_hold_lateral_motion_penalty'

@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'),reason='Physical/GPU validation requires remote Slurm')
def test_a32_default_runtime_actual_velocity_and_reset():
    import mujoco
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
    torch.set_num_threads(1)
    recipe=make_recipe(world_x=.16,right_world_y=.15,left_world_y=-.15,world_z=.09,
        palm_base_yaw=.2617993877991494,control_hz=20,physics_hz=1000,
        midi=str(ROOT/'training_mjlab/inputs/turkish_march_advanced/a32_repeat_static_guarded_20261025/turkish-march-advanced_fingered.proto'),
        initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',
        task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
        preparation_mode='first_note_v1',preparation_time=2,preparation_lift=0,ending_time=2)
    scene,task,_=build(recipe)
    scene,task=without_arms(scene,task,scene.parent/'key_hold_no_arms')
    cfg=effective_config(parser().parse_args(['--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
        '--action-profile','direct_joint_v1','--finger-coupling','--no-coupling-reduce-observation',
        '--finger-coupling-angle-ranges',str(ROOT/'configs/finger_coupling/angle_ranges_10_80.json'),
        '--preparation-mode','first_note_v1','--preparation-time','2','--preparation-lift','0','--ending-time','2',
        '--reward-config',str(ROOT/'configs/rewards/default_20261006.json'),
        '--hard-joint-limits','--wrist-tx-max-velocity','1.99','--wrist-ty-max-velocity','1.99',
        '--wrist-tx-max-acceleration','1.99','--wrist-ty-max-acceleration','1.99',
        '--wrist-pitch-max-velocity','3.1416','--wrist-pitch-max-acceleration','15.708']))
    env=PianoTaskEnv(scene,task,num_envs=2,device='cuda:0',physics_graph=False,
        disable_wrist_tz=True,disable_wrist_yaw=True,
        **{k:cfg[k] for k in ('preparation','finger_coupling','control_limits','reward_config','action_profile')})
    env.reset(); s=env.scene; collector=s.reward_sampling.key_hold_lateral
    assert collector is not None and s.decimation==50
    assert env.preparation_steps==0
    action=s.policy_action(s.control_reset_action).expand(2,-1).clone()
    oracle=mujoco.MjData(s.model); max_error=0.;rewards=[]
    for _ in range(3):
        result=env.step(action)
        speed=collector.velocity_history; held=collector.contact_history
        expected=-.5*(held*((speed.abs()-.01)/.09).clamp(0,1).square()).amax(-1).mean(-1)
        torch.testing.assert_close(result.info['reward_weighted_terms'][TERM],expected)
        assert torch.isfinite(speed).all() and speed.shape==(2,50,2)
        assert ((expected>=-.5)&(expected<=0)).all()
        rewards.append(expected.tolist())
        for world in range(2):
            oracle.qpos[:]=s.data.qpos[world].cpu().numpy()
            oracle.qvel[:]=s.data.qvel[world].cpu().numpy()
            mujoco.mj_forward(s.model,oracle)
            for side,name in enumerate(('right_/base_link','left_/base_link')):
                jac=np.zeros((3,s.model.nv));rot=np.zeros_like(jac)
                mujoco.mj_jacBody(s.model,oracle,jac,rot,s.model.body(name).id)
                actual=float(speed[world,-1,side]);reference=float((jac@oracle.qvel)[1])
                max_error=max(max_error,abs(actual-reference))
                assert actual==pytest.approx(reference,abs=2e-5)
    other_speed=collector.velocity_history[1].clone()
    env.reset(torch.tensor([0],device='cuda:0'))
    assert collector.velocity_history[0].count_nonzero()==0
    torch.testing.assert_close(collector.velocity_history[1],other_speed)
    report=dict(status='passed',worlds=2,control_hz=20,physics_hz=1000,steps=3,
        actual_velocity_max_error_m_s=max_error,rewards=rewards,contract=collector.contract,
        preparation_valid=s.bundle.preparation['report']['valid'],training_started=False,
        scope='A3.2 default preparation/runtime/reset; synthetic real-contact tests separately validate nonzero gate')
    out=Path(os.environ['TSENSEI_TEST_REPORT_DIR']) if 'TSENSEI_TEST_REPORT_DIR' in os.environ else ROOT/'logs/key_hold_physical'
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'),reason='Rigid-arm/GPU validation requires remote Slurm')
def test_rigid_arm_mocap_applied_motion_not_dynamic_joint_velocity():
    from types import SimpleNamespace
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp
    from tests.test_optional_arm_follow import options,PREP
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
    from tsensei_v3_2_pianist.mjlab_env.substep_rewards import SubstepRewardCollector
    torch.set_num_threads(1)
    scene,task,_=build(make_recipe(initial_buffer_time=0,preparation_time=1.,preparation_lift=.025,
        preparation_mode='policy_v1',task_profile='tsensei_collision_v1',drive_profile='shadow_v1',exact_note_timing=True))
    bundle=SceneBundle(scene,disable_wrist_tz=False,disable_wrist_yaw=True,preparation=PREP,arm_follow=options())
    model=bundle.model; initial=bundle.reset_cpu()
    # Test the real assembly/follower from its actual initialized state. The
    # unrelated MJLab constructor zero-mocap contact overflow is documented.
    key_ids=bundle.ids(mujoco.mjtObj.mjOBJ_GEOM,json.loads((task/'task.json').read_text())['key_geoms'])
    with wp.ScopedDevice('cuda:0'), torch.cuda.stream(wp.stream_to_torch(wp.get_stream('cuda:0'))):
        m=mw.put_model(model);d=mw.put_data(model,initial,nworld=2,nconmax=1024,njmax=4096)
        follower=SceneArmFollower(bundle,num_envs=2,device='cuda:0')
        view=SimpleNamespace(mocap_pos=wp.to_torch(d.mocap_pos),mocap_quat=wp.to_torch(d.mocap_quat))
        sampling=SubstepRewardCollector(num_envs=2,decimation=10,device='cuda:0',key_qadr=bundle.key_qadr,
            key_limits=model.jnt_range[bundle.key_ids],action_ids=bundle.action_ids,
            geom_pair_mask=np.zeros((model.ngeom,model.ngeom),dtype=np.int32),
            key_hold_model=model,key_hold_geom_ids=key_ids)
        collector=sampling.key_hold_lateral
        assert collector.has_mocap
        ids=[model.body(side+'/base_link').id for side in ('right_','left_')]
        physical=torch.tensor(bundle.neutral_action,device='cuda:0').expand(2,-1).clone()
        physical[:,bundle.manifest['action_names'].index('R_Palm_Base_Ty')]+=.02
        sim=SimpleNamespace(wp_data=d)
        mw.forward(m,d);peak=0.
        for step in range(3):
            if step==2:
                follower.reset(torch.tensor([0],device='cuda:0'));follower.publish(view);mw.forward(m,d)
                sampling.clear(torch.tensor([0],device='cuda:0'))
            sampling.clear();sampling.begin_control(sim)
            previous=wp.to_torch(d.xpos)[:,ids].clone();expected=[]
            for substep in range(10):
                times=torch.full((2,), (step*10+substep+1)*model.opt.timestep,device='cuda:0')
                follower.step(follower.requests(physical,times));follower.publish(view);mw.forward(m,d)
                sampling.sample(sim,substep)
                current=wp.to_torch(d.xpos)[:,ids].clone()
                expected.append((current[:,:,1]-previous[:,:,1])/model.opt.timestep)
                previous=current
            reference=torch.stack(expected,1)
            torch.testing.assert_close(collector.velocity_history,reference,atol=2e-5,rtol=1e-4)
            peak=max(peak,float(reference.abs().max()))
        assert peak>1e-5, 'Arm fixture must exercise nonzero actual palm speed'
