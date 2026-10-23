"""Continuous correct-key travel with exact substep timing and real wrong keys."""
import math
import numpy as np
from types import SimpleNamespace
import pytest
import torch
from tsensei_v3_2_pianist.mjlab_env.task_logic import key_press_substep_reward
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate


def case(samples=10, device='cpu'):
    target=torch.zeros(1,89,device=device);target[:,10]=1
    travel=torch.zeros(1,samples,88,device=device)
    return target, dict(key_state_history=travel,
        key_activation_history=torch.zeros_like(travel,dtype=torch.bool),
        exact_goal_history=target[:,None,:].expand(-1,samples,-1).clone())


def test_continuous_correct_key_travel_has_no_activation_jump():
    target,s=case();s['key_state_history'][:,:,10]=.9
    depth=math.exp(math.log(.1)*(.05/.5)**2)
    expected=.6*depth+.4
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(expected)
    s['key_activation_history'][:,:,10]=True
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(expected)


@pytest.mark.parametrize('travel', [0., .5, .55, .9, .95, 1.])
def test_continuous_travel_matches_original_gaussian_formula(travel):
    target,s=case();s['key_state_history'][:,:,10]=travel
    depth=math.exp(math.log(.1)*(max(0.,1-travel-.05)/.5)**2)
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.6*depth+.4)


def test_wrong_key_uses_activation_not_travel_or_control_endpoint():
    target,s=case();s['key_state_history'][:,:,10]=1.
    s['key_state_history'][:,:,30]=1.
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(1.)
    s['key_activation_history'][:,4,30]=True
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.9+.1*(.6+.4*math.exp(-1.25)))


def test_transient_wrong_key_cannot_disappear_at_endpoint():
    target,s=case();s['key_state_history'][:,:,10]=1;s['key_activation_history'][:,:,10]=True
    s['key_activation_history'][:,4,30]=True
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.9+.1*(.6+.4*math.exp(-1.25)))


def test_only_last_substep_pressed_has_low_score():
    target,s=case();s['key_state_history'][:,-1,10]=1;s['key_activation_history'][:,-1,10]=True
    empty_depth=math.exp(math.log(.1)*( .95/.5)**2)
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.1+.9*(.6*empty_depth+.4))


def test_original_on_off_inside_control_frame_and_silence():
    target,s=case();s['exact_goal_history'].zero_()
    s['exact_goal_history'][:,2:7,10]=1
    s['key_state_history'][:,2:7,10]=1;s['key_activation_history'][:,2:7,10]=True
    # Five perfect music samples, five correctly silent prep/tail samples.
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.7)
    target.zero_() # policy-frame goal cannot overwrite original boundaries
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.7)


@pytest.mark.parametrize('control_dt', [.05, .025, .01])
def test_original_subframe_events_drive_reward_and_f1_on_the_same_clock(control_dt):
    from tsensei_v3_2_pianist.mjlab_env.exact_score import ExactScore
    from tsensei_v3_2_pianist.rl_torch.note_timing_metrics import NoteTimingMetrics
    # Neither boundary lies on the 20Hz or 200Hz grid. The four true physical
    # samples are .015/.020/.025/.030; these event times must not be rounded.
    events=np.array([[.012,.031,10,1]],dtype=np.float64)
    score=ExactScore(events,np.empty((0,2)),length=round(.1/control_dt),
        control_dt=control_dt,physics_dt=.005,lookahead_steps=1)
    history=score.goal_history[None,:,:,:88].bool()
    assert history.sum().item()==4
    rewards=[]
    for i in range(len(score.goal)):
        sample=dict(key_state_history=history[:,i].float(),
            key_activation_history=history[:,i],exact_goal_history=score.goal_history[None,i])
        # An intentionally wrong control-grid target cannot replace exact MIDI.
        rewards.append(key_press_substep_reward(sample,torch.zeros(1,89),.05))
    assert torch.stack(rewards).mean().item()==pytest.approx((4*1.+16*.4)/20)
    metrics=NoteTimingMetrics(events=events,physics_dt=.005,music_start=0.)
    metrics.sample(history.reshape(1,-1,88))
    measured=metrics.result()
    assert measured['exact_original_music_micro_f1']==1.
    assert measured['exact_original_music_true_positive_count']==4
    assert measured['note_timing_onset_mean_ms']==pytest.approx(3.)
    assert measured['note_timing_offset_mean_ms']==pytest.approx(4.)


def test_mean_not_sum_and_no_endpoint_fallback():
    for samples in (1,10,20):
        target,s=case(samples);s['key_state_history'][:,:,10]=1;s['key_activation_history'][:,:,10]=True
        assert key_press_substep_reward(s,target,.05).item()==pytest.approx(1.)
    with pytest.raises(KeyError):
        key_press_substep_reward({},target,.05)


def test_saved_runtime_contract_describes_the_current_formula_and_clock():
    from tsensei_v3_2_pianist.rl_torch.train import score_timing_contract
    from tsensei_v3_2_pianist.mjlab_env.exact_score import ExactScore
    score=ExactScore(np.array([[.012,.031,10,1]]),np.empty((0,2)),
        length=2,control_dt=.05,physics_dt=.005,lookahead_steps=1)
    env=SimpleNamespace(reward_config=template(),note_timing_contract=score.contract,
        bundle=SimpleNamespace(manifest=dict(control_dt=.05,physics_dt=.005,score_validation='strict_v1'),
            music_end_step=1,release_tail_steps=1))
    saved=score_timing_contract(env)
    assert saved['reward_contract']=='physical_key_fingering_exact_activation100_v1'
    assert saved['key_correct_signal']=='continuous_gaussian_key_travel'
    assert saved['key_wrong_signal']=='physical_key_activation_count'
    assert 'key_activation_weight' not in saved
    assert saved['reward_target']=='original_midi_physical_substeps'
    assert saved['reward_state']=='post_physics_substep_mean'
    assert saved['exact_events']['physical_sample_time']=='post_step'
    assert saved['exact_events']['interval']=='start_inclusive_end_exclusive'
    assert (saved['control_dt'],saved['physics_dt'])==(.05,.005)
    assert saved['exact_events'] is not score.contract
    target,sample=case();sample['key_state_history'][:,:,10]=.9
    continuous=.6*math.exp(math.log(.1)*(.05/.5)**2)+.4
    assert key_press_substep_reward(sample,target,.05).item()==pytest.approx(continuous)
    sample['key_activation_history'][:,:,10]=True
    assert key_press_substep_reward(sample,target,.05).item()==pytest.approx(continuous)


@pytest.mark.parametrize('contract', ['shadow_control_frame_distinct_fingers_v1',
    'physical_key_activation50_v1'])
def test_old_contract_and_legacy_key_formula_are_rejected(contract):
    c=template();c['contract']=contract
    with pytest.raises(ValueError,match='incompatible'):validate(c)
    c=template();c['terms']['key_press_reward']['params']['mode']='legacy'
    with pytest.raises(ValueError,match='only supports'):validate(c)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Slurm CUDA only')
def test_cuda_capture_updates_physical_history():
    target,s=case(device='cuda:0')
    for _ in range(3): key_press_substep_reward(s,target,.05)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):value=key_press_substep_reward(s,target,.05)
    s['key_state_history'][:,:,10]=1;s['key_activation_history'][:,:,10]=True
    graph.replay();torch.cuda.synchronize()
    assert value.item()==pytest.approx(1.)
    s['key_activation_history'][:,4,30]=True
    graph.replay();torch.cuda.synchronize()
    assert value.item()==pytest.approx(.9+.1*(.6+.4*math.exp(-1.25)))
