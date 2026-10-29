"""Light CPU checks for current Roll layout, drive, observation and rate limits."""
import argparse
import torch
import numpy as np
from tsensei_v3_2_pianist.drive_settings import load
import runpy
from types import SimpleNamespace
from pathlib import Path
constants = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[1]/'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py')))
from tsensei_v3_2_pianist.mjlab_env.action_mapping import _direct_contract, active_indices, physical_bounds
from tsensei_v3_2_pianist.mjlab_env.control_limits import add_rate_arguments, rate_overrides, resolve
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter
from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FOLLOWER_INDICES
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tests.test_paper_rewards import only, scene_fixture
from tests.test_mjlab_task import fixture


def test_layout_preserves_fingers_and_exposes_both_roll_axes():
    rows = _direct_contract()['channels']
    names = [r['name'] for r in rows]
    assert len(names) == 53
    assert names[25] == 'R_Palm_Base_Roll' and names[51] == 'L_Palm_Base_Roll'
    assert all(names[i].endswith(('_PIP', '_DIP')) for i in FOLLOWER_INDICES)
    for size in (33, 35, 37, 49, 51, 53):
        active = active_indices(size, 'direct_joint_v1')
        assert len(active) == size and {25, 51}.issubset(active)
        if size < 40:
            assert not set(active) & set(FOLLOWER_INDICES)
    for size in (31, 47):
        active = active_indices(size, 'wuji_cartesian_v1')
        assert len(active) == size
        assert not any(names[i].endswith(('_Pitch', '_Yaw', '_Roll')) for i in active)
    assert constants.PALM_BASE_DOFS['palm_base_roll'][1:3] == ((0, 0, 1), (-.3, .5))
    assert constants.PALM_BASE_DOFS['palm_base_pitch'][2] == (-.3, .6)
    for r in rows:
        if r['name'].endswith('_Pitch'):
            assert (r['physical_min'], r['physical_max']) == (-.3, .6)
    low, high = physical_bounds('direct_joint_v1', names,
        [r['physical_min'] for r in rows], [r['physical_max'] for r in rows])
    assert [low[i] for i in (25, 51)] == [-.3, -.3]
    assert [high[i] for i in (25, 51)] == [.5, .5]


def test_pitch_roll_drive_and_reward_group():
    drive = load('shadow_v1')
    for axis in ('pitch', 'roll'):
        assert {k: drive[axis][k] for k in ('kp','damping','force_limit')} == dict(kp=8., damping=.5, force_limit=18.)
    manager = RewardManager(only(), scene_fixture(), fixture().manifest, num_envs=1, device='cpu')
    wrist_names = {manager.actions[i] for i in manager.groups['wrist']}
    assert {'R_Palm_Base_Roll','L_Palm_Base_Roll'} <= wrist_names
    assert len(manager.groups['finger']) == 40


def test_roll_reversals_at_50hz_and_partial_reset():
    parser = argparse.ArgumentParser()
    add_rate_arguments(parser)
    args = parser.parse_args(['--wrist-roll-max-velocity','3.1416','--wrist-roll-max-acceleration','15.708'])
    args.disable_wrist_tz = args.disable_wrist_yaw = True
    options = rate_overrides(None, args)
    rows = _direct_contract()['channels']; names = [r['name'] for r in rows]
    reset = torch.zeros(53)
    config = resolve(options, action_names=names, active_indices=active_indices(33,'direct_joint_v1'),
        physical_low=[r['physical_min'] for r in rows], physical_high=[r['physical_max'] for r in rows],
        reset_action=reset, control_dt=.02)
    limiter = TargetMotionLimiter(config, num_envs=2, device='cpu', reset_action=reset)
    previous = limiter.previous.clone(); velocity = limiter.velocity.clone()
    for step in range(300):
        request = torch.zeros(2,53)
        request[:,limiter.indices] = (.5 if (step//32)%2 else -.3)
        applied,_ = limiter.apply(request)
        value = applied[:,limiter.indices].double()
        speed = (value-previous)/.02; acceleration = (speed-velocity)/.02
        assert value.min() >= -.3 and value.max() <= .5
        assert speed.abs().max() <= 3.1416+1e-12
        assert acceleration.abs().max() <= 15.708+1e-12
        previous,velocity = value.clone(),speed.clone()
    other = limiter.previous[1].clone()
    limiter.reset(torch.tensor([0]))
    assert torch.equal(limiter.previous[0],torch.zeros(2,dtype=torch.float64))
    assert torch.equal(limiter.previous[1],other)


def test_current_palm_translation_ranges():
    import json
    root=Path(__file__).resolve().parents[1]
    for profile in ('direct_joint_v1','wuji_cartesian_v1'):
        rows=json.loads((root/f'configs/actions/{profile}.json').read_text())['channels']
        for axis,bounds in [('ty',(-.05,.05)),('tz',(-.6,.46))]:
            assert constants.PALM_BASE_DOFS['palm_base_'+axis][2]==bounds
            selected=[r for r in rows if r['name'].endswith('_Palm_Base_'+axis.title())]
            assert len(selected)==2
            assert all((r['physical_min'],r['physical_max'])==bounds for r in selected)
