"""Versioned policy coordinates and explicit per-joint physical target ranges."""
import json
import math
from pathlib import Path


PROFILES = ('legacy', 'mixed_v1', 'direct_joint_v1')
_REMOVED = {49: (22, 47), 47: (22, 24, 47, 49)}
_FOLLOWERS = tuple(base+i for base in (0, 25) for i in (6, 7, 10, 11, 14, 15, 18, 19))
_DIRECT_PATH = Path(__file__).resolve().parents[2] / 'configs/actions/direct_joint_v1.json'


def active_indices(action_size):
    if action_size in (31, 33, 35):
        return [i for i in active_indices(action_size+16) if i not in _FOLLOWERS]
    if action_size == 51:
        return list(range(51))
    if action_size in _REMOVED:
        removed = _REMOVED[action_size]
        return [i for i in range(51) if i not in removed]
    raise ValueError(f'Unsupported TsenseI action size: {action_size}')


def _direct_contract():
    data = json.loads(_DIRECT_PATH.read_text())
    channels = data.get('channels', ())
    if data.get('schema_version') != 1 or data.get('name') != 'direct_joint_v1' or len(channels) != 51:
        raise ValueError(f'Invalid direct action contract: {_DIRECT_PATH}')
    names = [row.get('name') for row in channels]
    if len(set(names)) != 51 or names[-1] != 'sustain':
        raise ValueError('Direct action contract requires 51 unique canonical channels ending in sustain')
    for row in channels:
        values = [row.get(key) for key in ('policy_min', 'policy_max', 'physical_min', 'physical_max')]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError(f'Nonfinite direct action range: {row.get("name")}')
        if not values[0] < values[1] or not values[2] < values[3]:
            raise ValueError(f'Non-increasing direct action range: {row.get("name")}')
    return data


def policy_bounds(profile='legacy', action_size=51):
    """Return explicit policy-coordinate lower and upper bounds."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown action profile: {profile}')
    if profile == 'direct_joint_v1':
        indices = active_indices(action_size)
        rows = _direct_contract()['channels']
        return ([rows[i]['policy_min'] for i in indices],
                [rows[i]['policy_max'] for i in indices])
    center = shifts(profile, action_size)
    return ([value - 1 for value in center], [value + 1 for value in center])


def physical_bounds(profile, action_names, mechanical_low, mechanical_high, *, range_scales=None):
    """Return full 51-channel commanded physical targets and validate the scene."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown action profile: {profile}')
    range_scales = {} if range_scales is None else range_scales
    if (not isinstance(range_scales, dict) or set(range_scales) - {'R_Wrist_Pitch', 'L_Wrist_Pitch'}
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 < v <= 1 for v in range_scales.values())):
        raise ValueError('Invalid explicit wrist Pitch action range scales')
    low, high = list(mechanical_low), list(mechanical_high)
    if len(action_names) != 51 or len(low) != 51 or len(high) != 51:
        raise ValueError('Physical action contract requires the canonical 51-channel layout')
    if profile != 'direct_joint_v1':
        return low, high
    rows = _direct_contract()['channels']
    expected_names = [row['name'] for row in rows]
    if list(action_names) != expected_names:
        raise ValueError(f'Scene action names do not match {profile} canonical order')
    target_low = [row['physical_min'] for row in rows]
    target_high = [row['physical_max'] for row in rows]
    for i, name in enumerate(expected_names):
        target_low[i] *= range_scales.get(name, 1.)
        target_high[i] *= range_scales.get(name, 1.)
        if (target_low[i] < low[i] - 1e-9 or target_high[i] > high[i] + 1e-9
                or target_low[i] >= target_high[i]):
            raise ValueError(f'{profile} target for {name} exceeds scene mechanical range')
    return target_low, target_high


def resolved_contract(profile, action_size, action_names=None, mechanical_low=None, mechanical_high=None, *, range_scales=None):
    """Serializable active-channel contract for run metadata and audit tables."""
    indices = active_indices(action_size)
    policy_low, policy_high = policy_bounds(profile, action_size)
    result = {'name': profile, 'schema_version': 1, 'action_size': action_size}
    if action_names is None:
        result.update(policy_min=policy_low, policy_max=policy_high)
        return result
    physical_low, physical_high = physical_bounds(
        profile, action_names, mechanical_low, mechanical_high, range_scales=range_scales)
    result['channels'] = [dict(index=j, full_index=i, name=action_names[i],
                               policy_min=policy_low[j], policy_max=policy_high[j],
                               physical_min=physical_low[i], physical_max=physical_high[i])
                          for j, i in enumerate(indices)]
    return result


def shifts(profile='legacy', action_size=51):
    if profile not in PROFILES:
        raise ValueError(f'Unknown action profile: {profile}')
    if profile == 'legacy':
        if isinstance(action_size, bool) or not isinstance(action_size, int) or action_size < 1:
            raise ValueError('action_size must be a positive integer')
        return [0.] * action_size
    if profile == 'direct_joint_v1':
        low, high = policy_bounds(profile, action_size)
        return [(a + b) * .5 for a, b in zip(low, high)]
    if action_size != 51:
        full = shifts(profile, 51)
        return [full[i] for i in active_indices(action_size)]
    result = [0.] * 51
    if profile == 'mixed_v1':
        if action_size != 51:
            raise ValueError('mixed_v1 requires the 51-action TsenseI layout')
        for base in (0, 25):
            for i in range(20):
                if i not in (5, 9, 13, 17):
                    result[base+i] = 1.
            result[base+21] = 1.
            result[base+22] = -1.
    return result
