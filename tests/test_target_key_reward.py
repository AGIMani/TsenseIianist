"""Job1724 key formula: physical samples, true activation, original boundaries."""
import math
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


def test_half_actual_activation_and_half_depth():
    target,s=case();s['key_state_history'][:,:,10]=.9
    depth=math.exp(math.log(.1)*(.05/.5)**2)
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.3*depth+.4)
    s['key_activation_history'][:,:,10]=True
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.3*depth+.3+.4)


def test_transient_wrong_key_cannot_disappear_at_endpoint():
    target,s=case();s['key_state_history'][:,:,10]=1;s['key_activation_history'][:,:,10]=True
    s['key_activation_history'][:,4,30]=True
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.9+.1*(.6+.4*math.exp(-1.25)))


def test_only_last_substep_pressed_has_low_score():
    target,s=case();s['key_state_history'][:,-1,10]=1;s['key_activation_history'][:,-1,10]=True
    empty_depth=math.exp(math.log(.1)*( .95/.5)**2)
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.1+.9*(.3*empty_depth+.4))


def test_original_on_off_inside_control_frame_and_silence():
    target,s=case();s['exact_goal_history'].zero_()
    s['exact_goal_history'][:,2:7,10]=1
    s['key_state_history'][:,2:7,10]=1;s['key_activation_history'][:,2:7,10]=True
    # Five perfect music samples, five correctly silent prep/tail samples.
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.7)
    target.zero_() # policy-frame goal cannot overwrite original boundaries
    assert key_press_substep_reward(s,target,.05).item()==pytest.approx(.7)


def test_mean_not_sum_and_no_endpoint_fallback():
    for samples in (1,10,20):
        target,s=case(samples);s['key_state_history'][:,:,10]=1;s['key_activation_history'][:,:,10]=True
        assert key_press_substep_reward(s,target,.05).item()==pytest.approx(1.)
    with pytest.raises(KeyError):
        key_press_substep_reward({},target,.05)


def test_old_contract_and_legacy_key_formula_are_rejected():
    c=template();c['contract']='shadow_control_frame_distinct_fingers_v1'
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
