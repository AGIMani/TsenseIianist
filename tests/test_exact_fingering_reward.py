"""Regression: same finger changes key inside a 20 Hz action, without double targets."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tsensei_v3_2_pianist.mjlab_env.exact_score import ExactScore
from tsensei_v3_2_pianist.mjlab_env.task_logic import fingering_substep_reward, MusicTaskLogic
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tests.test_mjlab_task import fixture, snapshot


def score(events, length=4):
    return ExactScore(np.array(events,float),np.empty((0,2)),length=length,
        control_dt=.05,physics_dt=.005,lookahead_steps=10,anticipation_horizon=.15)


def test_switch_within_action_scores_only_the_current_key_and_actual_sample():
    s=score([[0,.023,10,1],[.027,.049,20,1]])
    keys=s.finger_key_history[:1]
    assert keys[0,:,1].tolist()==[10,10,10,10,-1,20,20,20,20,-1]
    data=dict(exact_finger_key_history=keys, fingertip_history=torch.zeros(1,10,10,3),
              key_target_history=torch.zeros(1,10,88,3))
    data['key_target_history'][:,:,20,0]=.11
    # Perfectly follow the exact active key even though a single control step
    # contains two notes: eight active physical samples out of ten earn 1.
    data['fingertip_history'][:,5:9,1,0]=.11
    assert fingering_substep_reward(data,.01).item()==pytest.approx(.8)
    # Only the last physical sample changes: no endpoint substitution allowed.
    data['fingertip_history'][:,-1]=100
    assert fingering_substep_reward(data,.01).item()==pytest.approx(.8)
    # Hold the old key; the next four samples score .1 rather than 1.
    data['fingertip_history'][:,5:9,1,0]=0
    assert fingering_substep_reward(data,.01).item()==pytest.approx(.44)
    assert s.fingers[0,1,10] and not s.fingers[0,1,20]
    assert s.future[0,1,20] and s.future_seconds[0,1].item()==pytest.approx(.027)


def test_active_finger_mean_then_time_mean_and_partial_indices():
    s=score([[0,.05,10,1],[.015,.035,20,6]])
    data=dict(exact_finger_key_history=s.finger_key_history[[0,1,0]],
        fingertip_history=torch.zeros(3,10,10,3),key_target_history=torch.zeros(3,10,88,3))
    data['fingertip_history'][:,:,6,0]=.11
    # Nine active samples; four contain two fingers: 5*1 + 4*(1+.1)/2.
    torch.testing.assert_close(fingering_substep_reward(data,.01),torch.tensor([.72,0,.72]))
    with pytest.raises(KeyError):fingering_substep_reward({'fingertips':torch.zeros(1,10,3)},.01)


def test_current_observation_is_exact_and_future_is_not_merged():
    s=score([[0,.023,10,1],[.027,.049,20,1]])
    bundle=fixture(history=False,forearm=False)
    logic=MusicTaskLogic(bundle,num_envs=1,device='cpu')
    logic.observation_goal,logic.observation_fingers=s.goal,s.fingers
    frame=logic.frame(snapshot(1))
    torch.testing.assert_close(frame[:,:10],s.fingers[0].sum(-1)[None])
    torch.testing.assert_close(frame[:,10:99],s.goal[0][None])
    assert frame[0,10+10]==1 and frame[0,10+20]==0
    assert s.future[0,1,20]==1


def test_removed_formulas_and_previous_checkpoint_contract_rejected():
    for mode in ('legacy','paper','staged'):
        cfg=template();cfg['terms']['fingering_reward']['params']={'mode':mode}
        with pytest.raises(ValueError):validate(cfg)
    cfg=template();cfg['contract']='physical_key_travel_exact_v1'
    with pytest.raises(ValueError):validate(cfg)


def test_original_015_events_without_rewriting_fingers_or_times():
    data=json.loads((Path(__file__).parent/'fixtures/exact_fingering_015.json').read_text())
    events=np.array(data['events'])
    assert len(events)==269 and data['retained_grid_conflicts']==39
    length=int(np.ceil(events[:,1].max()/.05))+5
    s=score(events,length)
    np.testing.assert_array_equal(s.events,events)
    # Reconstruct all active physical events independently and verify the
    # compact table preserves every note's original finger, onset and release.
    times=(np.arange(length*10)+1)*.005
    assignments=np.zeros((len(times),10,88),dtype=np.uint8)
    for start,end,key,finger in events:
        active=(times>=start-1e-10)&(times<end-1e-10)
        assert active.any()
        assignments[active,int(finger),int(key)]=1
    assert assignments.sum(-1).max()==1
    keys=np.where(assignments.any(-1),assignments.argmax(-1),-1)
    np.testing.assert_array_equal(s.finger_key_history.reshape(-1,10),keys)
