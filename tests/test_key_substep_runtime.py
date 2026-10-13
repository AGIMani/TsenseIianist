"""Real physical collection and task reward integration, Slurm allocation only."""
import json
import os
from pathlib import Path
import pytest
import torch

@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Slurm GPU allocation required')
def test_current_model_physical_substeps_and_partial_reset():
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    recipe=make_recipe(control_hz=20, physics_hz=200, lookahead=10,
        task_profile='tsensei_collision_v1', collision_profile='convex_hull_v1',
        drive_profile='shadow_v1', initial_buffer_time=0., preparation_time=1.,
        preparation_mode='policy_v1', release_tail_time=.25,
        score_validation='strict_v1', exact_note_timing=True)
    scene,task,_=build(recipe)
    scene,task=without_arms(scene,task,scene.parent/'hands_only')
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/rewards/default_20261006.json').read_text())
    limits=dict(schema=1,mode='target_clip_slew_v1',joint_state_limits='hard_v1',
        channels={f'{side}_Wrist_{axis}':dict(
            max_velocity=3.1416 if axis=='Pitch' else .99,
            max_acceleration=15.708 if axis=='Pitch' else 1.99)
            for side in ('R','L') for axis in ('TX','TY','Pitch')})
    coupling=dict(schema=1,mode='distal_angle_range_v1',reduce_observation=False,
        angle_ranges_deg=json.loads((root/'configs/finger_coupling/angle_ranges_60_80.json').read_text()))
    for graph in (False,True):
        env=PianoTaskEnv(scene,task,num_envs=2,physics_graph=graph,
            action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=recipe['preparation'],reward_config=cfg,
            finger_clearance=None,collision_diagnostics=False,
            control_limits=limits,finger_coupling=coupling)
        action=env.scene.policy_action(env.logic.reset_action).expand(2,-1)
        # Covers preparation -> original first note boundary with actual kernels.
        for step in range(24):
            result=env.step(action)
            h=env.scene.reward_sampling.key_state_history
            active=env.scene.reward_sampling.key_activation_history
            assert h.shape==(2,10,88)
            assert torch.isfinite(h).all()
            expected=[]
            for world in range(2):
                per_sample=[]
                goals=env.exact_score.goal_history[result.info['score_index'][world]]
                for j in range(10):
                    on=goals[j,:88].bool()
                    x=1-h[world,j]
                    depth=torch.exp(torch.log(torch.tensor(.1,device=h.device))*((torch.maximum(-x,x-.05).clamp_min(0)/.5)**2))
                    positive=((.5*depth+.5*active[world,j].float())*on).sum()/on.sum().clamp_min(1)
                    wrong=(active[world,j]&~on).sum()
                    per_sample.append(.6*positive+.4*torch.exp(-1.25*wrong.float()))
                expected.append(torch.stack(per_sample).mean())
            torch.testing.assert_close(result.info['reward_terms']['key_press_reward'],torch.stack(expected))
        env.reset([0])
        assert env.scene.reward_sampling.key_state_history[0].count_nonzero()==0
        result=env.step(action)
        assert result.info['score_index'].tolist()==[0,24]
        assert torch.isfinite(result.reward).all()
