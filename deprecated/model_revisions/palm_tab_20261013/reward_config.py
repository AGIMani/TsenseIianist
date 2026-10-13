"""Serializable reward choices. Unknown paper/model mappings are never guessed.

TODO(V32-PAPER-U01/U02/U05): author shaping, command units and model mappings
remain undisclosed. Null fields are deliberate activation gates, not defaults
to fill from synthetic test fixtures.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from . import performance_options

GROUPS = ('forearm', 'wrist', 'finger')
KINDS = ('power', 'velocity', 'acceleration')
COEFFICIENTS = {'power': (-.001, -.01, -.005),
                'velocity': (-.02, -.005, -.0002),
                'acceleration': (-.002, -.0003, -.00001)}
PAPER_WEIGHTS = {
    'stage1': {'power': (.2,.176,.2), 'velocity': (.2,.176,.2), 'acceleration': (.2,.176,.2)},
    'stage2_canon': {'power': (4.6,.1,1.), 'velocity': (2.6,.1,1.), 'acceleration': (6.4,.1,1.)},
    'stage2_other': {'power': (4.,.1,1.), 'velocity': (2.2,.1,1.), 'acceleration': (5.5,.1,1.)},
}
FINGER_NAMES = tuple(f'{side}_{finger}' for side in ('R','L')
                     for finger in ('Thumb','Index','Mid','Ring','Pinky'))
IDLE_TARGET_POSTURE_GESTURE = dict(coefficient=-.1,
                                  reference='scene_reset', lookahead_steps=20)
JOINT_MOTION_LIMIT_PENALTY = dict(
    finger_velocity_limit=10., finger_acceleration_limit=200.,
    wrist_velocity_limit=None, wrist_acceleration_limit=None,
    tcp_velocity_limit=None, tcp_acceleration_limit=None,
    full_penalty_excess=.2, maximum_penalty_per_step=5.)


def template(stage='current'):
    if stage not in ('current', *PAPER_WEIGHTS):
        raise ValueError(f'Unknown reward weight stage: {stage}')
    paper = stage != 'current'
    weight_stage = 'stage1' if not paper else stage
    def term(enabled=False, weight=1., **params):
        return dict(enabled=enabled, weight=weight, params=params)
    terms = {
        'key_press_reward': term(True, 2. if 'stage2' in stage else 1.,
            mode='paper' if paper else 'legacy', chord_bonus=False, chord_scope=None),
        'sustain_reward': term(not paper),
        'energy_reward': term(not paper),
        'fingering_reward': term(True, 1.5 if 'stage2' in stage else 1., mode='paper' if paper else 'legacy'),
        'forearm_reward': term(not paper),
        'self_collision_reward': term(False, .2, aggregation='substep_sum',
                                      maximum_penalty_per_step=5.),
        # Daxian travel-based wrong-key cost; opt-in, fixed original formula.
        'false_press_penalty': term(False),
        'humanoid_finger_gesture_reward': term(False, 1., **IDLE_TARGET_POSTURE_GESTURE),
        # Physical command differences, in rad/s and rad/s^2; no guessed wrist limits.
        'joint_motion_limit_penalty': term(False, 1., **JOINT_MOTION_LIMIT_PENALTY),
        'arm_motion_limit_penalty': term(False, 1., soft_limit_ratio=.95,
            full_penalty_excess=.1, maximum_penalty_per_step=5.),
    }
    for kind in KINDS:
        for i, group in enumerate(GROUPS):
            terms[f'{group}_{kind}_reward'] = term(False, PAPER_WEIGHTS[weight_stage][kind][i],
                                                   coefficient=COEFFICIENTS[kind][i])
    terms.update({
        'finger_speed_reward': term(False, .4 if stage=='stage2_canon' else .3 if stage=='stage2_other' else 0.,
                                    coefficient=-.05, limit=6.5),
        'wrist_pitch_reward': term(False, bonus=.05, directions=None),
        'forward_press_reward': term(False, coefficient=-.1, threshold=math.pi/2, angle_map=None),
    })
    return dict(contract='shadow_control_frame_current', weight_stage=stage, scale_by_dt=paper, action_units=None,
                regularization_shaping=None if 'stage2' in stage else 'identity', terms=terms)


def finite_number(value, path, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f'{path} must be a finite number')
    if (nonnegative and value < 0) or (positive and value <= 0):
        raise ValueError(f'{path} must be {"positive" if positive else "nonnegative"}')


def validate(config):
    """Validate structure and explicit user choices before GPU allocation."""
    try:
        json.dumps(config,allow_nan=False)
    except (ValueError,TypeError) as error:
        raise ValueError('Reward configuration must be finite, JSON-serializable data') from error
    config = deepcopy(config)
    reference = template()
    if (not isinstance(config, dict) or
            not set(reference) <= set(config) or
            set(config) - set(reference) - {'performance', 'reference_dt'}):
        raise ValueError(f'Reward config requires these fields: {list(reference)}; '
                         'optional: performance, reference_dt')
    if 'performance' in config:
        config['performance'] = performance_options.canonical(config['performance'])
    if config['contract'] != 'shadow_control_frame_current':
        raise ValueError('Reward config must use the current Shadow control-frame contract')
    if config['weight_stage'] not in ('current', *PAPER_WEIGHTS):
        raise ValueError('Unknown weight_stage')
    if type(config['scale_by_dt']) is not bool:
        raise ValueError('scale_by_dt must be true/false')
    if 'reference_dt' in config:
        finite_number(config['reference_dt'], 'reference_dt', positive=True)
        if not config['scale_by_dt']:
            raise ValueError('reference_dt requires scale_by_dt=true')
    if config['action_units'] not in (None,'physical','normalized'):
        raise ValueError('action_units must be null, physical or normalized')
    if config['regularization_shaping'] not in (None,'identity'):
        raise ValueError('Only explicit identity shaping is implemented; paper Stage2 shaping is undisclosed')
    if not isinstance(config['terms'],dict) or set(config['terms']) != set(reference['terms']):
        raise ValueError(f'Reward terms must match the template: {list(reference["terms"])}')
    enabled = set()
    for name, term in config['terms'].items():
        if not isinstance(term,dict) or set(term) != {'enabled','weight','params'}:
            raise ValueError(f'{name} requires enabled, weight and params')
        if type(term['enabled']) is not bool:
            raise ValueError(f'{name}.enabled must be true/false')
        finite_number(term['weight'], f'{name}.weight', nonnegative=name!='false_press_penalty')
        params = term['params']
        expected_params = reference['terms'][name]['params']
        gesture = name == 'humanoid_finger_gesture_reward'
        if name == 'self_collision_reward' and isinstance(params, dict):
            if params.get('aggregation') != 'substep_sum':
                raise ValueError('self_collision_reward requires capped substep_sum; '
                                 'deprecated collision configurations are not supported')
        if not isinstance(params,dict) or set(params) != set(expected_params):
            raise ValueError(f'{name}.params must contain {list(expected_params)}')
        if gesture and params.get('reference') != 'scene_reset':
            raise ValueError('Humanoid gesture reference must be scene_reset')
        if gesture and (type(params['lookahead_steps']) is not int or params['lookahead_steps'] <= 0):
            raise ValueError('Humanoid gesture lookahead_steps must be a positive integer')
        if name == 'joint_motion_limit_penalty':
            for key in ('finger_velocity_limit', 'finger_acceleration_limit', 'full_penalty_excess'):
                finite_number(params[key], f'{name}.{key}', positive=True)
            for group in ('wrist', 'tcp'):
                limits = (params[f'{group}_velocity_limit'], params[f'{group}_acceleration_limit'])
                if (limits[0] is None) != (limits[1] is None):
                    raise ValueError(f'joint_motion_limit_penalty {group} limits must both be null or positive')
                if limits[0] is not None:
                    for key in (f'{group}_velocity_limit', f'{group}_acceleration_limit'):
                        finite_number(params[key], f'{name}.{key}', positive=True)
            finite_number(params['maximum_penalty_per_step'],
                          f'{name}.maximum_penalty_per_step', nonnegative=True)
            if term['weight'] > 1. or params['maximum_penalty_per_step'] > 5.:
                raise ValueError('joint_motion_limit_penalty requires weight<=1 and maximum_penalty_per_step<=5')
        if name == 'arm_motion_limit_penalty':
            for key in ('soft_limit_ratio', 'full_penalty_excess'):
                finite_number(params[key], f'{name}.{key}', positive=True)
            if (params['soft_limit_ratio'] != .95 or params['full_penalty_excess'] != .1
                    or params['maximum_penalty_per_step'] != 5. or term['weight'] != 1.):
                raise ValueError('Arm penalty contract requires 95% SDK, 10% excess, weight=1, independent maximum=5')
        for key in ('coefficient','limit','bonus','threshold','maximum_penalty_per_step'):
            if key in params:
                finite_number(params[key],f'{name}.{key}',positive=key in ('limit','threshold'),
                              nonnegative=key in ('bonus','maximum_penalty_per_step'))
        if 'coefficient' in params and params['coefficient']>0:
            raise ValueError(f'{name}.coefficient must not be positive')
        if term['enabled']:
            enabled.add(name)
    terms = config['terms']
    for name in ('key_press_reward','fingering_reward'):
        if terms[name]['params']['mode'] not in ('legacy','paper'):
            raise ValueError(f'{name}.mode must be legacy or paper')
    experiments = performance_options.get(config)
    if (experiments['anticipatory_fingering_time'] and experiments['anticipatory_fingering_reward']
            or experiments['staged_fingering_reward']):
        if ('fingering_reward' not in enabled or
                terms['fingering_reward']['params']['mode'] != 'legacy'):
            raise ValueError('Fingering experiments require enabled legacy fingering_reward')
    key = terms['key_press_reward']['params']
    if type(key['chord_bonus']) is not bool or key['chord_scope'] not in (None,'on','whole_key'):
        raise ValueError('key_press_reward: chord_bonus is boolean, chord_scope is null/on/whole_key')
    if 'key_press_reward' in enabled and key['chord_bonus']:
        if key['mode']!='paper' or key['chord_scope'] is None:
            raise ValueError('Chord bonus requires paper mode and an explicit chord_scope (on or whole_key)')
    derivative = {f'{group}_{kind}_reward' for group in GROUPS for kind in ('velocity','acceleration')}
    if enabled & (derivative | {'finger_speed_reward'}):
        if config['action_units'] is None:
            raise ValueError('Fill action_units (physical or normalized) before enabling command/posture rewards')
    if 'finger_speed_reward' in enabled and config['action_units']!='physical':
        raise ValueError('finger_speed_reward uses rad/s and requires action_units=physical')
    regularizers = derivative | {f'{g}_power_reward' for g in GROUPS} | {'finger_speed_reward'}
    if enabled & regularizers and config['regularization_shaping'] is None:
        raise ValueError('Stage2 shaping is undisclosed: explicitly set regularization_shaping=identity to use unshaped formulas')
    if 'energy_reward' in enabled and enabled & {f'{g}_power_reward' for g in GROUPS}:
        raise ValueError('Disable legacy energy_reward before enabling positive-power rewards')
    directions=terms['wrist_pitch_reward']['params']['directions']
    if 'wrist_pitch_reward' in enabled:
        if not isinstance(directions,dict) or set(directions)!={'R_Wrist_Pitch','L_Wrist_Pitch'}:
            raise ValueError('Fill wrist_pitch_reward.directions for R_Wrist_Pitch and L_Wrist_Pitch')
        if any(type(v) not in (int,float) or v not in (-1,1) for v in directions.values()):
            raise ValueError('Wrist pitch directions must each be +1 or -1')
    angles=terms['forward_press_reward']['params']['angle_map']
    if 'forward_press_reward' in enabled:
        if not isinstance(angles,dict) or set(angles)!=set(FINGER_NAMES):
            raise ValueError(f'Fill forward_press_reward.angle_map for all ten fingers: {FINGER_NAMES}')
        for finger, mapping in angles.items():
            if not isinstance(mapping,dict) or set(mapping)!={'joints','offset'} or not isinstance(mapping['joints'],dict) or not mapping['joints']:
                raise ValueError(f'{finger} needs joints (action_name: coefficient) and offset (rad)')
            finite_number(mapping['offset'],finger+'.offset')
            for name,value in mapping['joints'].items():
                if not isinstance(name,str) or not name.startswith(finger+'_'):
                    raise ValueError(f'{finger}: joints must belong to this finger')
                finite_number(value, finger+'.'+name)
            if not any(mapping['joints'].values()):
                raise ValueError(f'{finger}: at least one nonzero joint coefficient is required')
    return deepcopy(config)


def canonical(config=None, self_collision_weight=0.):
    finite_number(self_collision_weight,'self_collision_weight',nonnegative=True)
    if config is not None and self_collision_weight:
        raise ValueError('Use reward_config OR --self-collision-weight, not both')
    if config is None:
        config = template()
        if self_collision_weight:
            config['terms']['self_collision_reward'].update(enabled=True,weight=self_collision_weight)
    return validate(config)


def load(path):
    return validate(json.loads(Path(path).read_text()))


def contact_options(config):
    categories = ('self',) if config['terms']['self_collision_reward']['enabled'] else ()
    options = dict(collision_categories=categories, ignored_body_pairs=[])
    term = config['terms']['self_collision_reward']
    if term['enabled'] and term['params'].get('aggregation') == 'substep_sum':
        options['self_collision_substep_sum'] = True
    return options
