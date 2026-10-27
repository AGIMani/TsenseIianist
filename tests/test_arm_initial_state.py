"""A3.2 runtime integration on remote Slurm; no policy training."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TERM='key_hold_lateral_motion_penalty'

@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'),reason='Physical/GPU validation requires remote Slurm')
def test_approved_arm_initial_state_and_current_default_step():
    import mujoco
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
    torch.set_num_threads(1)
    recipe=make_recipe(world_x=.16,right_world_y=.15,left_world_y=-.15,world_z=.09,
        palm_base_yaw=.2617993877991494,control_hz=20,physics_hz=1000,
        midi=str(ROOT/'training_mjlab/inputs/turkish_march_advanced/a32_repeat_static_guarded_20261025/turkish-march-advanced_fingered.proto'),
        initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',
        task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
        preparation_mode='first_note_v1',preparation_time=0,preparation_lift=0,ending_time=0)
    scene,task,_=build(recipe)
    import math
    angles=json.loads((ROOT/'configs/preview/current_arm_training_20261011.json').read_text())['arm_follow']['initial_q_rad']
    flags=['--arms','--arm-initial-right-deg',*[str(math.degrees(x)) for x in angles['right']],
           '--arm-initial-left-deg',*[str(math.degrees(x)) for x in angles['left']]]
    cfg=effective_config(parser().parse_args([*flags,'--disable-wrist-yaw',
        '--action-profile','wuji_cartesian_v1','--finger-coupling','--no-coupling-reduce-observation',
        '--finger-coupling-angle-ranges',str(ROOT/'configs/finger_coupling/angle_ranges_10_80.json'),
        '--preparation-mode','first_note_v1','--preparation-time','0','--preparation-lift','0','--ending-time','0',
        '--reward-config',str(ROOT/'configs/rewards/default_20261006.json'),
        '--hard-joint-limits','--wrist-tx-max-velocity','1.99','--wrist-ty-max-velocity','1.99',
        '--wrist-tx-max-acceleration','1.99','--wrist-ty-max-acceleration','1.99',
        '--wrist-pitch-max-velocity','0','--wrist-pitch-max-acceleration','0']))
    env=PianoTaskEnv(scene,task,num_envs=2,device='cuda:0',physics_graph=False,
        disable_wrist_tz=False,disable_wrist_yaw=True,
        **{k:cfg[k] for k in ('preparation','finger_coupling','control_limits','reward_config','action_profile','arm_follow')})
    env.reset(); s=env.scene; collector=s.reward_sampling.key_hold_lateral
    assert collector is not None and s.decimation==50
    assert s.sim.cfg.nconmax == 1024 and s.sim.cfg.njmax == 4096
    np.testing.assert_allclose(s.sim.mj_data.mocap_pos, s.bundle.reset_mocap_pos, atol=1e-12)
    np.testing.assert_allclose(s.sim.mj_data.qpos, s.bundle.reset_qpos, atol=1e-12)
    assert env.preparation_steps==0
    action=s.policy_action(s.control_reset_action).expand(2,-1).clone()
    rewards=[]
    for _ in range(3):
        result=env.step(action)
        speed=collector.velocity_history; held=collector.contact_history
        expected=-.5*(held*((speed.abs()-.01)/.09).clamp(0,1).square()).amax(-1).mean(-1)
        torch.testing.assert_close(result.info['reward_weighted_terms'][TERM],expected)
        assert torch.isfinite(speed).all() and speed.shape==(2,50,2)
        assert ((expected>=-.5)&(expected<=0)).all()
        rewards.append(expected.tolist())
        assert torch.isfinite(result.observation).all()
        assert s.arm_follower is not None and collector.has_mocap
    other_speed=collector.velocity_history[1].clone()
    env.reset(torch.tensor([0],device='cuda:0'))
    assert collector.velocity_history[0].count_nonzero()==0
    torch.testing.assert_close(collector.velocity_history[1],other_speed)
    report=dict(status='passed',worlds=2,control_hz=20,physics_hz=1000,steps=3,
        rewards=rewards,contract=collector.contract,
        preparation_disabled=s.bundle.preparation is None,training_started=False,
        scope='Approved bilateral M6 angles, current A3.2 default runtime with preparation/ending disabled; reset and 3 control steps')
    out=Path(os.environ['TSENSEI_TEST_REPORT_DIR']) if 'TSENSEI_TEST_REPORT_DIR' in os.environ else ROOT/'logs/exper043_validation'
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
