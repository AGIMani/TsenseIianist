"""Idle target L1: current plus 20 future score frames, independent of anticipation."""
from copy import deepcopy

import pytest
import torch

from tests.test_paper_rewards import only, scene_fixture
from tests.test_reward_defaults_20261006 import GESTURE, evaluate_posture
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, gesture_protection_mask


def test_current_and_twentieth_future_frame_exempt_but_twenty_first_does_not():
    score = torch.zeros(23, 10, 88)
    score[0, 0, 30] = 1
    score[1, 1, 31] = 1
    score[20, 5, 32] = 1  # Left thumb, exactly the inclusive future boundary.
    score[21, 9, 33] = 1  # Left pinky, beyond the boundary at frame zero.
    protected = gesture_protection_mask(score, 20)
    assert protected[0].nonzero().flatten().tolist() == [0, 1, 5]
    assert protected[1].nonzero().flatten().tolist() == [1, 5, 9]
    assert not protected[22].any()  # No wrap to the score's opening notes.
    result, _ = evaluate_posture(torch.ones(2, 51), score[[0, 22]],
        snapshot=dict(gesture_protected_fingers=protected[[0, 22]]))
    torch.testing.assert_close(result, torch.tensor([-2.45, -3.6]))


def test_preparation_rest_and_local_reset_use_each_world_score_index():
    score = torch.zeros(48, 10, 88)
    score[20:24, 2, 20] = 1  # First note follows 20 silent preparation frames.
    score[46, 7, 30] = 1
    protected = gesture_protection_mask(score, 20)
    assert protected[0, 2]  # Already exempt on the very first preparation step.
    assert not protected[24].any()  # Rest with the next note still 22 steps away.
    assert protected[26, 7]
    assert not protected[-1].any()
    indices = torch.tensor([26, 47])
    before, manager = evaluate_posture(torch.ones(2, 51), score[indices],
        snapshot=dict(gesture_protected_fingers=protected[indices]))
    torch.testing.assert_close(before, torch.tensor([-3.25, -3.6]))
    indices[0] = 0
    manager.reset(torch.tensor([0]))
    after = manager({}, torch.ones(2, 51),
        dict(gesture_protected_fingers=protected[indices]), torch.zeros(2, 89), score[indices])[GESTURE]
    torch.testing.assert_close(after, before)
    assert protected[indices][0, 2] and not protected[indices][0, 7]


@pytest.mark.parametrize('steps', [None, True, 0, -1, 20.0, '20'])
def test_horizon_is_explicit_positive_integer_and_old_current_only_config_rejected(steps):
    config = only(GESTURE)
    if steps is None:
        del config['terms'][GESTURE]['params']['lookahead_steps']
    else:
        config['terms'][GESTURE]['params']['lookahead_steps'] = steps
    with pytest.raises(ValueError, match='lookahead_steps'):
        validate(config)


@pytest.mark.parametrize('mask', [None, torch.zeros(1, 10), torch.zeros(1, 9, dtype=torch.bool)])
def test_required_future_mask_cannot_silently_fall_back_to_current_only(mask):
    manager = RewardManager(only(GESTURE), scene_fixture(), {}, num_envs=1, device='cpu')
    sample = {} if mask is None else dict(gesture_protected_fingers=mask)
    with pytest.raises(ValueError, match='gesture_protected_fingers'):
        manager({}, torch.ones(1, 51), sample, torch.zeros(1, 89), torch.zeros(1, 10, 88))


def test_future_protection_does_not_depend_on_anticipatory_observation_or_reward():
    score = torch.zeros(22, 10, 88)
    score[20, :, 10] = 1
    mask = gesture_protection_mask(score, 20)[[0]]
    for seconds in (0., .15):
        cfg = deepcopy(only(GESTURE))
        cfg['performance'] = dict(anticipatory_fingering_time=seconds)
        result, manager = evaluate_posture(torch.ones(1, 51), cfg=cfg,
            snapshot=dict(gesture_protected_fingers=mask, anticipatory_mask=~mask))
        assert manager.gesture_lookahead_steps == 20
        assert result.item() == 0.


@pytest.mark.parametrize('side', ['R', 'L'])
@pytest.mark.parametrize('finger', ['Index', 'Mid', 'Ring', 'Pinky'])
def test_each_swing_has_half_the_mcp_penalty(side, finger):
    scene = scene_fixture()
    manager = RewardManager(only(GESTURE), scene, {}, num_envs=2, device='cpu')
    action = torch.zeros(2, 51)
    action[0, manager.actions.index(f'{side}_{finger}_SW')] = .2
    action[1, manager.actions.index(f'{side}_{finger}_MCP')] = .2
    reward, _ = evaluate_posture(action)
    torch.testing.assert_close(reward, torch.tensor([-.01, -.02]))


@pytest.mark.parametrize('value', [None, True, -.1, float('inf')])
def test_swing_scale_must_be_explicit_finite_nonnegative(value):
    cfg = only(GESTURE)
    if value is None:
        del cfg['terms'][GESTURE]['params']['swing_scale']
    else:
        cfg['terms'][GESTURE]['params']['swing_scale'] = value
    with pytest.raises(ValueError):
        validate(cfg)
