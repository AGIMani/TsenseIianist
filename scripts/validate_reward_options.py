#!/usr/bin/env python3
"""GPU reward integration acceptance, using explicitly labelled synthetic mappings.

Fixtures are arithmetic test inputs, not accepted hardware calibration. This
script never edits configs/rewards or launches a long training run.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, FINGER_NAMES, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output directory exists')
    a.output.mkdir(parents=True)
    # Fully explicit synthetic fixture; NOT saved as a user-facing preset.
    c=template('stage1');c['action_units']='physical'
    for t in c['terms'].values():t['enabled']=True
    for name in ('energy_reward','forearm_reward','self_collision_reward'):c['terms'][name]['enabled']=False
    c['terms']['finger_speed_reward']['weight']=.3
    c['terms']['natural_posture_reward']['params']['reference']='scene_reset'
    c['terms']['collision_reward']['params']['categories']=['self','inter_hand','piano_case']
    c['terms']['wrist_pitch_reward']['params']['directions']={'R_Palm_Base_Pitch':1,'L_Palm_Base_Pitch':-1}
    c['terms']['forward_press_reward']['params']['angle_map']={
        f:dict(joints={f+'_MCP':1.},offset=0.) for f in FINGER_NAMES}
    c['terms']['key_press_reward']['params'].update(chord_bonus=True,chord_scope='whole_key')
    validate(c)
    (a.output/'test_fixture_only.json').write_text(json.dumps(c,indent=2)+'\n')
    env=PianoTaskEnv(ROOT/'training_mjlab/outputs/scene',ROOT/'training_mjlab/outputs/task_twinkle',
                     num_envs=2,reward_config=c)
    manager=env.reward_manager
    cpu=RewardManager(c,env.scene.bundle.manifest,env.bundle.manifest,num_envs=2,device='cpu')
    obs=env.reset();cpu.reset(torch.arange(2));rng=torch.Generator(device='cuda:0').manual_seed(728)
    rows=[];max_error=0.
    for i in range(6):
        if i==3:
            saved=manager.previous[1].clone();saved2=manager.previous2[1].clone()
            env.reset([0]);cpu.reset(torch.tensor([0]))
            torch.testing.assert_close(manager.previous[1],saved)
            torch.testing.assert_close(manager.previous2[1],saved2)
        action=torch.rand((2,51),device=env.device,generator=rng)*2-1
        result=env.step(action)
        torch.testing.assert_close(result.reward,torch.stack(list(result.info['reward_terms'].values())).sum(0))
        torch.testing.assert_close(env.logic.last_reward,result.reward)
        snapshot={k:v.cpu() for k,v in env.snapshot().items()}
        index=result.info['score_index']
        base,_=env.logic.reward_terms(env.snapshot(),index)
        physical=env.scene.low+(action+1)*.5*(env.scene.high-env.scene.low)
        expected=cpu({k:v.cpu() for k,v in base.items()},physical.cpu(),snapshot,
                      env.logic.goal[index].cpu(),env.logic.fingers[index].cpu())
        for name,value in expected.items():
            actual=result.info['reward_terms'][name].cpu()
            error=(actual-value).abs().max().item();max_error=max(max_error,error)
            torch.testing.assert_close(actual,value,atol=3e-6,rtol=3e-5)
        assert torch.isfinite(result.observation).all() and torch.isfinite(result.reward).all()
        rows.append(dict(step=i+1,score_index=index.cpu().tolist(),
                         reward=result.reward.cpu().tolist(),
                         terms={k:v.cpu().tolist() for k,v in result.info['reward_terms'].items()}))
        obs=result.observation
    # Equivalent explicit current config must select the original arithmetic path.
    current=template()
    legacy=PianoTaskEnv(ROOT/'training_mjlab/outputs/scene',ROOT/'training_mjlab/outputs/task_twinkle',reward_config=current)
    assert legacy.reward_manager is None and legacy.scene.self_collision is None
    result=legacy.step(torch.zeros(1,51,device='cuda:0'))
    assert len(result.info['reward_terms'])==5
    report=dict(status='passed',gpu=torch.cuda.get_device_name(),steps=6,num_envs=2,
        enabled_terms=env.reward_term_names,max_cpu_gpu_term_error=max_error,
        partial_reset_score_indices=rows[3]['score_index'],legacy_fast_path=True,
        mapping_status='SYNTHETIC TEST FIXTURE ONLY; not a user mapping or deployable preset',rows=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
