"""CPU checks of TX/TY issued limits and pre-limit demand reward end to end."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.test_target_motion_limits import NAMES, config, contract
from tests.test_paper_rewards import scene_fixture
from tsensei_v3_2_pianist.mjlab_env.control_limits import rate_overrides
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

TERM = 'joint_motion_limit_penalty'
TCP = [f'{s}_Palm_Base_{a}' for s in ('R', 'L') for a in ('Tx', 'Ty')]


def fixture(n=1, dt=.05, **params):
    c = template()
    for name, term in c['terms'].items():
        term['enabled'] = name == TERM
    c['terms'][TERM]['params'].update(tcp_velocity_limit=.95, tcp_acceleration_limit=2.,
                                    full_penalty_excess=.1, **params)
    scene = scene_fixture()
    scene['control_dt'] = dt
    manager = RewardManager(c, scene, {}, num_envs=n, device='cpu')
    action = torch.tensor(scene['neutral_action']).repeat(n, 1)
    return manager, action


def score(manager, issued, requested):
    return manager({}, issued, dict(requested_physical_action=requested),
                   torch.zeros(len(issued), 89), torch.zeros(len(issued), 10, 88))[TERM]


@pytest.mark.parametrize('dt', [.025, .05, .1])
def test_all_four_translation_channels_obey_both_hard_bounds_and_partial_reset(dt):
    resolved = contract(config(**{name: dict(max_velocity=.95, max_acceleration=2.) for name in TCP}), dt)
    limiter = TargetMotionLimiter(resolved, num_envs=4, device='cpu', reset_action=torch.zeros(51))
    for name in TCP:
        assert resolved['channels'][name]['acceleration_unit'] == 'm/s^2'
        assert limiter.contract['channels'][name]['position_unit'] == 'm'
    rng = torch.Generator().manual_seed(719)
    previous = torch.zeros(4, 4, dtype=torch.float64)
    velocity = previous.clone()
    peak_v = peak_a = 0.
    for k in range(1000):
        if k == 500:
            limiter.reset(torch.tensor([0, 2]))
            previous[[0, 2]] = 0
            velocity[[0, 2]] = 0
        request = torch.randn(4, 51, generator=rng) * 10
        if k < 180:
            request[:, limiter.indices] = 100.
        elif k < 360:
            request[:, limiter.indices] = -100.
        issued, _ = limiter.apply(request)
        pos = issued[:, limiter.indices].double()
        v = (pos-previous)/dt
        a = (v-velocity)/dt
        assert (v.abs() <= .95+1e-12).all()
        assert (a.abs() <= 2.+1e-12).all()
        assert (pos >= limiter.low).all() and (pos <= limiter.high).all()
        assert torch.equal(issued[:, 0], request[:, 0])
        peak_v = max(peak_v, v.abs().max().item())
        peak_a = max(peak_a, a.abs().max().item())
        previous, velocity = pos.clone(), v.clone()
    assert peak_v > .94 and peak_a > 1.99


@pytest.mark.parametrize('axis', TCP)
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('derivative', ['velocity', 'acceleration'])
def test_demand_threshold_five_percent_and_ten_percent(axis, sign, derivative):
    manager, action = fixture(4)
    i = manager.actions.index(axis)
    requested = action.clone()
    speeds = sign * .95 * torch.tensor([.99, 1., 1.05, 1.1])
    if derivative == 'velocity':
        # Constant prior issued velocity cancels acceleration for this check.
        manager.previous2[:, i] = manager.previous[:, i] - speeds * manager.dt
        requested[:, i] += speeds * manager.dt
    else:
        requested[:, i] += sign * 2. * torch.tensor([.99, 1., 1.05, 1.1]) * manager.dt**2
    actual = score(manager, action, requested)
    torch.testing.assert_close(actual, torch.tensor([0., 0., -2.5, -5.]), atol=2e-5, rtol=1e-5)


def test_raw_demand_is_penalized_while_issued_targets_are_safe_and_history_is_issued():
    manager, action = fixture(2)
    ids = manager.tcp_indices
    resolved = contract(config(**{name: dict(max_velocity=.95, max_acceleration=2.) for name in TCP}))
    limiter = TargetMotionLimiter(resolved, num_envs=2, device='cpu', reset_action=action[0])
    raw = action.clone(); raw[:, ids] = .2
    issued, _ = limiter.apply(raw)
    assert (issued[:, ids].abs() <= .005).all()  # 2 * .05^2, first step from rest.
    torch.testing.assert_close(score(manager, issued, raw), torch.full((2,), -5.))
    # Repeat the issued command: demand speed is zero, deceleration only 2 m/s².
    torch.testing.assert_close(score(manager, issued, issued), torch.zeros(2), atol=1e-5, rtol=0)
    # A 2.1 m/s² demand from rest incurs -2.5; old raw .2 is not history.
    raw = issued.clone(); raw[:, ids] += 2.1 * .05**2
    torch.testing.assert_close(score(manager, issued, raw), torch.full((2,), -2.5), atol=2e-5, rtol=1e-5)
    manager.reset(torch.tensor([0]))
    raw = issued.clone()
    torch.testing.assert_close(score(manager, issued, raw), torch.zeros(2), atol=1e-5, rtol=0)


def test_mixed_units_share_one_maximum_and_exclude_tz_pedal():
    manager, action = fixture(3, wrist_velocity_limit=3., wrist_acceleration_limit=15.)
    raw = action.clone()
    raw[:, manager.tcp_indices] += 2.1 * .05**2  # TCP +5%, -2.5.
    action[1, manager.groups['finger']] += 210. * .05**2  # finger +5% against template 200.
    action[2, manager.groups['wrist']] += 16.5 * .05**2  # wrist +10%, -5.
    raw[:, [manager.actions.index('R_Palm_Base_Tz'), manager.actions.index('L_Palm_Base_Tz'), 50]] = 100.
    torch.testing.assert_close(score(manager, action, raw), torch.tensor([-2.5, -2.5, -5.]), atol=2e-5, rtol=1e-5)


def test_tcp_requires_request_and_explicit_complete_positive_threshold_pair():
    manager, action = fixture()
    for requested in (None, torch.zeros(1, 50)):
        with pytest.raises(ValueError, match='requested_physical_action'):
            score(manager, action, requested)
    for value in (None, 0, -1, True, float('nan'), float('inf')):
        for field in ('tcp_velocity_limit', 'tcp_acceleration_limit'):
            c = deepcopy(manager.config)
            c['terms'][TERM]['params'][field] = value
            with pytest.raises(ValueError):
                validate(c)
    old = deepcopy(manager.config)
    del old['terms'][TERM]['params']['tcp_velocity_limit']
    del old['terms'][TERM]['params']['tcp_acceleration_limit']
    with pytest.raises(ValueError, match='params must contain'):
        validate(old)


def test_cli_persists_bilateral_hard_limits_and_demand_thresholds():
    args = parser().parse_args(['--inspect-only', '--wrist-tx-max-velocity', '.95',
        '--wrist-ty-max-velocity', '.95', '--wrist-tx-max-acceleration', '2',
        '--wrist-ty-max-acceleration', '2', '--tcp-target-velocity-limit', '.95',
        '--tcp-target-acceleration-limit', '2', '--joint-motion-limit-penalty',
        '--joint-motion-full-penalty-excess', '.1'])
    c = effective_config(args)
    for name in TCP:
        assert c['control_limits']['channels'][name] == dict(max_velocity=.95, max_acceleration=2.)
    p = c['reward_config']['terms'][TERM]['params']
    assert (p['tcp_velocity_limit'], p['tcp_acceleration_limit'], p['full_penalty_excess']) == (.95, 2., .1)
    disabled = rate_overrides(c['control_limits'], SimpleNamespace(wrist_tx_max_acceleration=0.))
    assert 'max_acceleration' not in disabled['channels']['R_Palm_Base_Tx']
    assert disabled['channels']['L_Palm_Base_Ty']['max_acceleration'] == 2.


@pytest.mark.parametrize('transforms', [False, True])
def test_real_task_step_forwards_unclipped_tcp_demand_to_reward(transforms):
    # CPU scene double: real task.step and real reward manager, no GPU physics.
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    manager, neutral = fixture()
    raw = neutral.clone(); raw[:, manager.tcp_indices] += .2
    issued = neutral.clone(); issued[:, manager.tcp_indices] += .005
    scene = SimpleNamespace(tensor=torch.as_tensor, has_action_transform=transforms,
        finger_clearance_guard=None, finger_coupler=None, joint_limiter=None,
            arm_collision_tracker=None,
        policy_preparation=False, data=SimpleNamespace(overflow=torch.zeros(1), qvel=torch.zeros(1)),
        physical_action=lambda action: action, last_requested_physical_action=raw,
        last_applied_physical_action=issued, last_control_clip_mask=torch.zeros_like(raw, dtype=torch.bool),
        last_control_slew_mask=torch.zeros_like(raw, dtype=torch.bool), step=lambda *a, **kw: None)
    def advance(action, snapshot, *, extra_terms=None, reward_transform):
        terms = reward_transform({}, action, snapshot, torch.zeros(1, 89), torch.zeros(1, 10, 88))
        return SimpleNamespace(observation=torch.zeros(1, 1), reward=terms[TERM], info={'reward_terms': terms})
    env = SimpleNamespace(scene=scene, logic=SimpleNamespace(done=torch.zeros(1, dtype=torch.bool), advance=advance),
        reward_manager=manager, _update_timing_clearance=lambda: None, snapshot=lambda **kw: {},
        _with_preparation=lambda x: x, exact_score=None, collision_diagnostics=False,
        gesture_protected_fingers=None)
    result = PianoTaskEnv.step(env, raw)
    torch.testing.assert_close(result.reward, torch.tensor([-5.]))
    expected = issued if transforms else raw
    torch.testing.assert_close(manager.previous, expected[:, :50])


@pytest.mark.parametrize('dt', [.025, .05, .1])
def test_tcp_budget_is_per_control_step_and_weight_applies_once(dt):
    manager, action = fixture(dt=dt)
    manager.config['scale_by_dt'] = True
    manager.config['reference_dt'] = .05
    manager.terms[TERM]['weight'] = .5
    raw = action.clone(); raw[:, manager.tcp_indices] += 10.
    torch.testing.assert_close(score(manager, action, raw), torch.tensor([-2.5]))
