"""Pure UI/audit adapter for current self-collision occurrence scoring.

No model loading or physics. This does not change the training reward runtime.
Inputs must be native contacts after the model's collision filters.
"""
import math


def validate_coupling_contract(options):
    """Reject source metadata that differs from this probe's audited coupling."""
    expected = dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False,
                    angle_ranges_deg={f'{side}_{digit}': [10., 80.]
                                      for side in ('left', 'right')
                                      for digit in ('index', 'mid', 'ring', 'pinky')})
    if options != expected:
        raise ValueError('Expected the audited eight-finger 10–80 degree coupling contract')


def finger_identity(body_name):
    for side in ('left', 'right'):
        prefix=f'{side}_/{side}_hand_'
        if not body_name.startswith(prefix):
            continue
        link=body_name[len(prefix):]
        for digit in ('thumb','index','mid','ring','pinky'):
            suffixes=(('rota_block_link','rotaback_link2','MCP_link','PIP_link') if digit=='thumb'
                      else ('MCP_link','swing_link','PIP_link','DIP_link'))
            if link in {f'{digit}_{suffix}' for suffix in suffixes}:
                return side,digit
    return None


def classify_contact(body1, body2, distance, *, same_weld=False, ignored=False):
    distance=float(distance)
    if not math.isfinite(distance):
        raise ValueError('Contact distance must be finite')
    side=lambda name:'right' if name.startswith('right_/') else 'left' if name.startswith('left_/') else None
    first,second=finger_identity(body1),finger_identity(body2)
    a,b=side(body1),side(body2)
    if a and b:
        category=('cross_hand' if a!=b else 'hand_palm_or_mount' if first is None or second is None
                  else 'same_finger' if first==second else 'same_hand_different_fingers')
    elif (a or b) and ('piano/white_key_' in body1 or 'piano/black_key_' in body1
                      or 'piano/white_key_' in body2 or 'piano/black_key_' in body2):
        category='hand_key'
    elif (a or b) and 'piano/base' in (body1,body2):
        category='hand_piano_case'
    else:
        category='other'
    reason=('separated' if distance>0 else 'same_body' if body1==body2 else 'same_weld' if same_weld
            else 'explicit_reward_ignore' if ignored else None if category=='same_hand_different_fingers'
            else category)
    return dict(body1=body1,body2=body2,finger_ids=[first,second],category=category,
                reward_counted=reason is None,reward_exclusion_reason=reason,
                distance_m=distance,depth_m=max(0.,-distance))


def score_contacts(contacts, *, weight=.1, cap=5., physics_hz=1000., control_hz=20.):
    values=(weight,cap,physics_hz,control_hz)
    if not all(math.isfinite(v) for v in values) or weight<0 or cap<0 or control_hz<=0 or physics_hz<control_hz:
        raise ValueError('Invalid scoring contract')
    ratio=physics_hz/control_hz
    if not math.isclose(ratio,round(ratio),abs_tol=1e-9):
        raise ValueError('Expected an integer substep count')
    pairs={tuple(sorted((c['body1'],c['body2']))) for c in contacts if c['reward_counted']}
    count=len(pairs);substeps=round(ratio)
    return dict(eligible_body_pair_count=count,eligible_body_pairs=[list(p) for p in sorted(pairs)],
                instant_substep_penalty=-weight*count,instant_substep_penalty_is_uncapped=True,
                held_control_step_penalty=-min(weight*count*substeps,cap),
                substeps_per_control_step=substeps,physics_hz=physics_hz,control_hz=control_hz,
                weight=weight,cap=cap,
                basis='Static pose held for every physics substep; estimate only, no integration or measured force',
                counting='Unique body pair per physics substep, distance <= 0, no force threshold')


def validate_request(request, controls):
    if not isinstance(request,dict) or set(request)-{'id','values','reset'}:
        raise ValueError('Expected id/values/reset JSON object')
    if type(request.get('reset',False)) is not bool:
        raise ValueError('reset must be boolean')
    values=request.get('values',{})
    if not isinstance(values,dict):
        raise ValueError('values must map independent action names to numbers')
    allowed={c['name']:c for c in controls};result={}
    for name,value in values.items():
        if name not in allowed:
            raise ValueError(f'Unknown or coupled follower control: {name}')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'Nonfinite or nonnumeric control: {name}')
        if not allowed[name]['min']<=value<=allowed[name]['max']:
            raise ValueError(f'Control outside bounds: {name}')
        result[name]=float(value)
    return result
