"""Real physical collection and task reward integration, Slurm allocation only."""
import json
import os
from pathlib import Path
import pytest
import torch
import mujoco

@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Slurm GPU allocation required')
def test_current_model_physical_substeps_and_partial_reset():
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    root=Path(__file__).resolve().parents[1]
    recipe=make_recipe(midi=str(root/'tests/fixtures/exact_fingering_015.proto'), control_hz=20, physics_hz=200, lookahead=10,
        task_profile='tsensei_collision_v1', collision_profile='convex_hull_v1',
        drive_profile='shadow_v1', initial_buffer_time=0., preparation_time=1.,
        preparation_mode='policy_v1', release_tail_time=.25,
        score_validation='strict_v1', exact_note_timing=True,
        preparation_lift=.025, wrist_pitch_range_scale=1.)
    scene,task,_=build(recipe)
    scene,task=without_arms(scene,task,scene.parent/'hands_only')
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/rewards/default_20261006.json').read_text())
    limits=dict(schema=1,mode='target_clip_slew_v1',joint_state_limits='hard_v1',
        channels={f'{side}_Palm_Base_{axis}':dict(
            max_velocity=3.1416 if axis=='Pitch' else .99,
            max_acceleration=15.708 if axis=='Pitch' else 1.99)
            for side in ('R','L') for axis in ('Tx','Ty','Pitch')})
    coupling=dict(schema=1,mode='distal_angle_range_v1',reduce_observation=False,
        angle_ranges_deg=json.loads((root/'configs/finger_coupling/angle_ranges_45_80.json').read_text()))
    for graph in (False,True):
        env=PianoTaskEnv(scene,task,num_envs=2,physics_graph=graph,
            action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=recipe['preparation'],reward_config=cfg,
            finger_clearance=None,collision_diagnostics=False,
            control_limits=limits,finger_coupling=coupling)
        assert len(env.bundle.exact_note_events)==269
        assert int((env.logic.fingers.sum(-1)>1).sum())==39
        assert env.exact_score.fingers.sum(-1).max()<=1
        for i,name in enumerate(env.scene.bundle.manifest['action_names']):
            if name.endswith('_Palm_Base_Pitch'):
                assert env.scene.physical_low[i].item()==pytest.approx(0.)
                assert env.scene.physical_high[i].item()==pytest.approx(.5)
            if name.endswith('_MCP'):
                assert env.scene.physical_low[i].item()==pytest.approx(0.)
                assert env.scene.physical_high[i].item()==pytest.approx(1.4)
        assert all(v==[45.,80.] for v in env.scene.finger_coupling_options['angle_ranges_deg'].values())
        model=env.scene.model
        for side in ('right','left'):
            assert model.joint(f'{side}_/forearm_pitch').range.tolist()==pytest.approx([0.,.5])
        assert int(model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER)
        for side in ('right','left'):
            for axis in ('tx','ty'):
                joint=model.joint(f'{side}_/forearm_{axis}').id
                dof=int(model.jnt_dofadr[joint])
                actuator_ids=[i for i in range(model.nu) if int(model.actuator_trnid[i,0])==joint]
                assert len(actuator_ids)==1
                actuator=actuator_ids[0]
                assert model.dof_armature[dof]==pytest.approx(.02)
                assert model.dof_M0[dof]==pytest.approx(.972)
                assert model.dof_damping[dof]==pytest.approx(126. if axis=='tx' else 250.)
                assert model.actuator_gainprm[actuator,0]==pytest.approx(38400.)
                assert model.actuator_forcerange[actuator].tolist()==[-200.,200.]
        action=env.scene.policy_action(env.logic.reset_action).expand(2,-1)
        # Covers preparation -> original first note boundary with actual kernels.
        for step in range(24):
            result=env.step(action)
            h=env.scene.reward_sampling.key_state_history
            active=env.scene.reward_sampling.key_activation_history
            assert h.shape==(2,10,88)
            assert torch.isfinite(h).all()
            finger_expected=[]
            collector=env.scene.reward_sampling
            for world in range(2):
                samples=[]
                for j,keys in enumerate(result.info['finger_keys_substeps'][world]):
                    values=[]
                    for finger,key in enumerate(keys.tolist()):
                        if key<0:continue
                        distance=torch.linalg.vector_norm(collector.fingertip_history[world,j,finger]-collector.key_target_history[world,j,key])
                        values.append(torch.exp(torch.log(torch.tensor(.1,device=h.device))*(torch.clamp(distance-.01,min=0)/.1)**2))
                    samples.append(torch.stack(values).mean() if values else torch.zeros((),device=h.device))
                finger_expected.append(torch.stack(samples).mean())
            torch.testing.assert_close(result.info['reward_terms']['fingering_reward'],torch.stack(finger_expected))
            assert collector.fingertip_history.shape==(2,10,10,3)
            expected=[]
            for world in range(2):
                per_sample=[]
                goals=env.exact_score.goal_history[result.info['score_index'][world]]
                for j in range(10):
                    on=goals[j,:88].bool()
                    x=1-h[world,j]
                    depth=torch.exp(torch.log(torch.tensor(.1,device=h.device))*((torch.maximum(-x,x-.05).clamp_min(0)/.5)**2))
                    positive=(depth*on).sum()/on.sum().clamp_min(1)
                    wrong=(active[world,j]&~on).sum()
                    per_sample.append(.6*positive+.4*torch.exp(-1.25*wrong.float()))
                expected.append(torch.stack(per_sample).mean())
            torch.testing.assert_close(result.info['reward_terms']['key_press_reward'],torch.stack(expected))
        before=env.scene.reward_sampling.fingertip_history[1].clone()
        env.reset([0])
        assert env.scene.reward_sampling.fingertip_history[0].count_nonzero()==0
        torch.testing.assert_close(before,env.scene.reward_sampling.fingertip_history[1])
        assert env.scene.reward_sampling.key_state_history[0].count_nonzero()==0
        result=env.step(action)
        assert result.info['score_index'].tolist()==[0,24]
        assert torch.isfinite(result.reward).all()
