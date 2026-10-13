"""Opt-in physical target clipping/slew; never rewrites policy or joint ranges.

Positions use metres for Wrist_TX/TY/TZ and radians for rotational channels.
max_velocity means metres/s or radians/s in the same physical joint coordinate.
Explicit wrist max_acceleration enables a final discrete target-motion projection;
the scene applies it after all other target transforms, in m/s^2 or rad/s^2.
Target slew does not limit actual joint velocity or collision force. The optional
joint_state_limits=hard_v1 additionally bounds robot positions after integration.
"""
from copy import deepcopy
import math


MODE = 'target_clip_slew_v1'


def add_rate_arguments(parser):
    import argparse
    parser.add_argument('--hard-joint-limits', action=argparse.BooleanOptionalAction, default=None,
                        help='Project robot joint states into model/target limits after every physics substep')
    for axis in ('tx', 'ty', 'tz', 'pitch', 'yaw'):
        unit = 'm/s' if axis in ('tx', 'ty', 'tz') else 'rad/s'
        parser.add_argument(f'--wrist-{axis}-max-velocity', type=float, default=None,
                            help=f'Both wrists target rate in {unit}; 0 removes rate, omitted preserves JSON')
    for axis in ('tx', 'ty', 'pitch', 'yaw'):
        unit = 'm/s^2' if axis in ('tx', 'ty') else 'rad/s^2'
        parser.add_argument(f'--wrist-{axis}-max-acceleration', type=float, default=None,
                            help=f'Both wrists final target acceleration in {unit}; requires max velocity; '
                                 '0 removes acceleration, omitted preserves JSON')
    parser.add_argument('--finger-max-velocity', type=float, default=None,
                        help='All finger target rates in rad/s; 0 removes rates, omitted preserves JSON')
    parser.add_argument('--joint-max-velocity', action='append', default=[], metavar='NAME=VALUE',
                        help='Per-joint target rate, overriding group options; repeatable, 0 removes rate')


def rate_overrides(config, args):
    """JSON < bilateral/finger groups < named joints; zero removes only rate."""
    options = canonical(config)
    hard = getattr(args, 'hard_joint_limits', None)
    state_mode = (options or {}).get('joint_state_limits')
    if hard is not None:
        state_mode = 'hard_v1' if hard else None
    wrist = {axis: getattr(args, f'wrist_{axis}_max_velocity', None)
             for axis in ('tx', 'ty', 'tz', 'pitch', 'yaw')}
    acceleration = {axis: getattr(args, f'wrist_{axis}_max_acceleration', None)
                    for axis in ('tx', 'ty', 'pitch', 'yaw')}
    finger = getattr(args, 'finger_max_velocity', None)
    joints = getattr(args, 'joint_max_velocity', [])
    if (all(value is None for value in (*wrist.values(), *acceleration.values()))
            and finger is None and not joints and hard is None):
        return options
    from .action_mapping import _direct_contract, active_indices
    names = [row['name'] for row in _direct_contract()['channels']]
    size = (47 if getattr(args, 'disable_wrist_yaw', False) else
            49 if getattr(args, 'disable_wrist_tz', False) else 51)
    active = {names[i] for i in active_indices(size)} - {'sustain'}
    channels = deepcopy(options['channels']) if options else {}

    def apply(name, value, field='max_velocity'):
        if name not in names or name == 'sustain':
            raise ValueError(f'Unknown hand control channel: {name}')
        value = _number(value, name+'.'+field)
        if value < 0:
            raise ValueError(f'{name}: target rate must be nonnegative (0 disables)')
        if value > 0 and name not in active:
            raise ValueError(f'Control limits cannot address a removed/fixed channel: {name}')
        row = channels.setdefault(name, {})
        if value == 0:
            row.pop(field, None)
            if not row:
                del channels[name]
        else:
            row[field] = value

    for axis, value in wrist.items():
        if value is not None:
            label = axis.upper() if axis in ('tx', 'ty', 'tz') else axis.title()
            for side in ('R', 'L'):
                apply(f'{side}_Wrist_{label}', value)
    if finger is not None:
        for name in names[:-1]:
            if '_Wrist_' not in name:
                apply(name, finger)
    for item in joints:
        name, separator, raw = item.partition('=')
        if not separator:
            raise ValueError('--joint-max-velocity requires NAME=VALUE')
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError('--joint-max-velocity requires a numeric NAME=VALUE') from error
        apply(name, value)
    for axis, value in acceleration.items():
        if value is not None:
            label = axis.upper() if axis in ('tx', 'ty') else axis.title()
            for side in ('R', 'L'):
                apply(f'{side}_Wrist_{label}', value, 'max_acceleration')
    result = dict(schema=1, mode=MODE, channels=channels)
    if state_mode is not None:
        result['joint_state_limits'] = state_mode
    return canonical(result) if channels or state_mode else None


def _number(value, name, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise ValueError(f'{name} must be a finite '+('positive number' if positive else 'number'))
    return float(value)


def canonical(config):
    """Return canonical user options. None preserves the historical execution."""
    if config is None:
        return None
    if (not isinstance(config,dict) or not {'schema','mode','channels'} <= set(config)
            or set(config)-{'schema','mode','channels','joint_state_limits'}):
        raise ValueError('control_limits requires schema, mode and channels')
    if type(config['schema']) is not int or config['schema']!=1 or config['mode']!=MODE:
        raise ValueError('Unsupported control_limits schema/mode')
    channels=config['channels']
    if 'joint_state_limits' in config and config['joint_state_limits'] != 'hard_v1':
        raise ValueError('joint_state_limits must be hard_v1 or omitted')
    if not isinstance(channels,dict) or (not channels and 'joint_state_limits' not in config):
        raise ValueError('control_limits.channels must be a nonempty named mapping')
    if any(not isinstance(name,str) or not name for name in channels):
        raise ValueError('Control channel names must be nonempty strings')
    result={}
    for name,row in sorted(channels.items()):
        if not isinstance(name,str) or not name or not isinstance(row,dict) or not row:
            raise ValueError('Each control channel needs a name and at least one bound/rate')
        if set(row)-{'min','max','max_velocity','max_acceleration'}:
            raise ValueError(f'Unknown control limit fields for {name}')
        parsed={key:_number(value,name+'.'+key,positive=key in ('max_velocity','max_acceleration'))
                for key,value in row.items()}
        if 'max_acceleration' in parsed:
            if name not in tuple(f'{side}_Wrist_{axis}' for side in ('R','L')
                                 for axis in ('TX','TY','Pitch','Yaw')):
                raise ValueError('Target acceleration limits support only Wrist_TX/TY/Pitch/Yaw')
            if 'max_velocity' not in parsed:
                raise ValueError(f'{name}: max_acceleration requires max_velocity')
        if 'min' in parsed and 'max' in parsed and parsed['min']>=parsed['max']:
            raise ValueError(f'{name}: min must be less than max')
        result[name]={key:parsed[key] for key in sorted(parsed)}
    return dict(schema=1,mode=MODE,channels=result,
                **({'joint_state_limits': config['joint_state_limits']} if 'joint_state_limits' in config else {}))


def resolve(config, *, action_names, active_indices, physical_low, physical_high,
            reset_action, control_dt):
    """Resolve defaults against the exact saved action/reset contract."""
    options=canonical(config)
    if options is None:
        return None
    dt=_number(control_dt,'control_dt',positive=True)
    names=list(action_names)
    if len(names)!=51 or len(set(names))!=51 or names[-1]!='sustain':
        raise ValueError('Control limits require the complete named 51-channel contract')
    if any(len(values)!=51 for values in (physical_low,physical_high,reset_action)):
        raise ValueError('Control physical bounds/reset must have 51 values')
    active=set(active_indices)
    channels={}
    for name,row in options['channels'].items():
        if name not in names or name=='sustain':
            raise ValueError(f'Control limits require an active hand channel: {name}')
        index=names.index(name)
        if index not in active:
            raise ValueError(f'Control limits cannot address a removed/fixed channel: {name}')
        original_low=_number(float(physical_low[index]),name+'.physical_min')
        original_high=_number(float(physical_high[index]),name+'.physical_max')
        reset=_number(float(reset_action[index]),name+'.reset_target')
        lower=row.get('min',original_low);upper=row.get('max',original_high)
        if not original_low<=lower<upper<=original_high:
            raise ValueError(f'{name}: limits must be ordered and inside original physical targets')
        if not lower<=reset<=upper:
            raise ValueError(f'{name}: software range must include the actual reset control target')
        translation=name.endswith(('_Wrist_TX','_Wrist_TY','_Wrist_TZ'))
        channels[name]=dict(index=index,min=lower,max=upper,max_velocity=row.get('max_velocity'),
            position_unit='m' if translation else 'rad',velocity_unit='m/s' if translation else 'rad/s',
            original_min=original_low,original_max=original_high,reset_target=reset)
        if 'max_acceleration' in row:
            channels[name].update(max_acceleration=row['max_acceleration'],
                                 acceleration_unit='m/s^2' if translation else 'rad/s^2')
    return dict(schema=1,mode=MODE,control_dt=dt,channels=channels,
        **({'joint_state_limits': options['joint_state_limits']} if 'joint_state_limits' in options else {}),
        application='original policy mapping, static clip, previous applied target slew, zero-order hold',
        action_observation='actual applied full physical 51 channels; same observation dimensions')


class TargetLimiter:
    """Pure one-step transform: caller supplies previous actually issued ctrl."""
    def __init__(self, contract, *, device):
        import torch
        if contract is None:
            raise ValueError('Do not allocate a limiter for the identity contract')
        self.contract=deepcopy(contract)
        self.indices=torch.tensor([row['index'] for row in contract['channels'].values()],device=device,dtype=torch.long)
        rows=list(contract['channels'].values())
        self.low=torch.tensor([row['min'] for row in rows],device=device)
        self.high=torch.tensor([row['max'] for row in rows],device=device)
        self.delta=torch.tensor([float('inf') if row['max_velocity'] is None else
            row['max_velocity']*contract['control_dt'] for row in rows],device=device)

    def apply(self, requested, previous):
        import torch
        if requested.shape!=previous.shape or requested.shape[-1]!=51:
            raise ValueError('Target limiter needs equally shaped full physical actions')
        values=requested[...,self.indices]
        clipped=values.clamp(self.low,self.high)
        old=previous[...,self.indices]
        limited=clipped.clamp(old-self.delta,old+self.delta)
        applied=requested.clone();applied[...,self.indices]=limited
        clip_mask=torch.zeros_like(requested,dtype=torch.bool)
        slew_mask=torch.zeros_like(requested,dtype=torch.bool)
        clip_mask[...,self.indices]=clipped!=values
        slew_mask[...,self.indices]=limited!=clipped
        return applied,clip_mask,slew_mask
