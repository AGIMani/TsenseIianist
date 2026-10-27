"""Kinematic preparation support, with fixed and policy-controlled modes.

The support motion is kinematic: it does not simulate an arm or its inertia.
No policy joint is added. Fixed preparation is outside the episode; policy
preparation receives ordinary music observations, rewards and replay entries.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MOUNTS = {side: f'preparation_{side}_mount' for side in ('right', 'left')}


def options(value):
    """Return the minimal, JSON-safe contract; None keeps historical reset."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {'duration', 'lift', 'trajectory', 'mode', 'hover', 'settle_time', 'position_tolerance', 'ending_duration'}:
        raise ValueError('Preparation requires duration, lift, trajectory and optional mode')
    duration, lift = float(value.get('duration', 1.)), float(value.get('lift', .02))
    trajectory = value.get('trajectory', 'quintic_v1')
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(lift) or lift < 0:
        raise ValueError('Preparation duration must be positive and lift nonnegative, both finite')
    if trajectory != 'quintic_v1':
        raise ValueError('Unknown preparation trajectory')
    if value.get('mode', 'fixed_v1') not in ('fixed_v1', 'policy_v1', 'first_note_v1'):
        raise ValueError('Unknown preparation mode')
    result = {'duration': duration, 'lift': lift, 'trajectory': trajectory}
    if value.get('mode') in ('policy_v1', 'first_note_v1'):
        result['mode'] = value['mode']
    if value.get('mode') == 'first_note_v1':
        if lift != 0:
            raise ValueError('First-note preparation uses physical joints; preparation lift must be zero')
        for name, default in (('hover', .012), ('settle_time', .5), ('position_tolerance', .005)):
            number = value.get(name, default)
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ValueError('First-note '+name+' must be finite and positive')
            result[name] = float(number)
    elif any(name in value for name in ('hover', 'settle_time', 'position_tolerance')):
        raise ValueError('Hover settings require first_note_v1')
    ending = float(value.get('ending_duration', 0.))
    if not math.isfinite(ending) or ending < 0 or (ending and value.get('mode') != 'first_note_v1'):
        raise ValueError('Ending duration must be finite, nonnegative and requires first_note_v1')
    if ending:
        result['ending_duration'] = ending
    return result


def is_policy(config):
    return config is not None and config.get('mode', 'fixed_v1') == 'policy_v1'


POLICY_OBSERVATION = {
    'name': 'preparation/remaining_fraction', 'shape': [1], 'range': [0., 1.],
    'placement': 'append after the stacked music observation',
    'meaning': '1 at raised initial, 0 at origin; configuration and phase determine support height and velocity',
}


def validate_timing(bundle, config):
    count = round(config['duration']/bundle.model.opt.timestep)
    if count < 1 or not math.isclose(count*bundle.model.opt.timestep, config['duration'], abs_tol=1e-10):
        raise ValueError('Preparation duration must contain whole physics steps')
    control_dt = bundle.manifest['control_dt']
    if not math.isclose(round(config['duration']/control_dt)*control_dt, config['duration'], abs_tol=1e-10):
        raise ValueError('Preparation duration must contain whole control frames')
    return count


def wrap_hands(xml, manifest):
    """Attach each complete hand to an identity mocap support at world level."""
    if 'm6' in manifest:
        raise ValueError('Preparation currently requires a hand-only scene; use --no-arms')
    world = xml.find('worldbody')
    if world is None:
        raise ValueError('Preparation scene has no worldbody')
    for side, name in MOUNTS.items():
        root_name = manifest['hands'][side]['body']
        matches = [body for body in world.findall('body')
                   if body.get('name') == root_name
                   or any(child.get('name') == root_name for child in body.iter('body'))]
        if len(matches) != 1:
            raise ValueError(f'Preparation cannot identify the complete {side} hand root')
        original = matches[0]
        if any(body.get('name') == name for body in xml.iter('body')):
            raise ValueError('Duplicate preparation mount')
        mount = ET.Element('body', name=name, mocap='true', pos='0 0 0', quat='1 0 0 0')
        index = list(world).index(original)
        world.remove(original)
        mount.append(original)
        world.insert(index, mount)


def lift_fraction(time_fraction):
    """Quintic descent: zero speed and acceleration at both endpoints."""
    u = np.clip(time_fraction, 0., 1.)
    return 1 - u**3 * (10 - 15*u + 6*u*u)


def _named_pose(bundle, data):
    model = bundle.model
    joints = {model.joint(i).name: float(data.qpos[model.jnt_qposadr[i]])
              for i in range(model.njnt)}
    joints.update(bundle.fixed_joint_values)
    return {
        'joint_positions': joints,
        'joint_velocities': {model.joint(i).name: float(data.qvel[model.jnt_dofadr[i]])
                             for i in range(model.njnt)},
        'actuator_targets': {model.actuator(i).name: float(data.ctrl[i]) for i in range(model.nu)},
        'hands': {side: {
            'body': hand['body'],
            'position': data.body(hand['body']).xpos.tolist(),
            'quaternion': data.body(hand['body']).xquat.tolist(),
        } for side, hand in bundle.manifest['hands'].items()},
        'mocap': {model.body(i).name: {
            'position': data.mocap_pos[mid].tolist(), 'quaternion': data.mocap_quat[mid].tolist(),
        } for i, mid in enumerate(model.body_mocapid) if mid >= 0},
        'sustain': 0.,
    }


def generate(bundle, config):
    """Compute once; return a complete trace and an unmodified physical terminal state.

    Uses a separate MjData and never advances a GPU training world. The caller
    copies the returned terminal state when resetting selected environments.
    """
    model = bundle.model
    dt = float(model.opt.timestep)
    count = validate_timing(bundle, config)
    data = mujoco.MjData(model)
    data.qpos[:] = bundle.origin_qpos
    data.qvel[:] = bundle.origin_qvel
    data.ctrl[:] = bundle.origin_ctrl
    data.mocap_pos[:] = bundle.origin_mocap_pos
    data.mocap_quat[:] = bundle.origin_mocap_quat
    mujoco.mj_forward(model, data)
    origin = _named_pose(bundle, data)
    mount_ids = np.array([model.body(name).mocapid[0] for name in MOUNTS.values()])
    if np.any(mount_ids < 0):
        raise ValueError('Preparation supports must be mocap bodies')
    base = data.mocap_pos[mount_ids].copy()
    data.mocap_pos[mount_ids, 2] = base[:, 2] + config['lift']
    mujoco.mj_forward(model, data)
    initial = _named_pose(bundle, data)
    rows = {name: [] for name in ('time', 'qpos', 'qvel', 'ctrl', 'mocap_pos',
                                  'mocap_quat', 'key_activation', 'hand_positions')}
    body_ids = [model.body(bundle.manifest['hands'][side]['body']).id for side in MOUNTS]

    def record(time):
        rows['time'].append(time)
        for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat'):
            rows[name].append(getattr(data, name).copy())
        rows['key_activation'].append(bundle.key_activation(data.qpos))
        rows['hand_positions'].append(data.xpos[body_ids].copy())

    record(0.)
    for index in range(1, count+1):
        data.mocap_pos[mount_ids, 2] = base[:, 2] + config['lift']*lift_fraction(index/count)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        record(index*dt)
    trace = {name: np.asarray(values) for name, values in rows.items()}
    if any(not np.isfinite(trace[name]).all() for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos')):
        raise FloatingPointError('Nonfinite state during deterministic preparation')
    joint_error = np.abs(data.qpos-bundle.origin_qpos)
    slide = model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE
    hinge = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    hand = np.array([model.joint(i).name.startswith(('right_/', 'left_/'))
                     for i in range(model.njnt)])
    report = {
        'physics_steps': count, 'physics_dt': dt,
        'max_joint_error_rad': float(joint_error[model.jnt_qposadr[hinge]].max(initial=0)),
        'max_slide_error_m': float(joint_error[model.jnt_qposadr[slide]].max(initial=0)),
        'max_hand_joint_error_rad': float(joint_error[model.jnt_qposadr[hand & hinge]].max(initial=0)),
        'max_hand_slide_error_m': float(joint_error[model.jnt_qposadr[hand & slide]].max(initial=0)),
        'max_abs_terminal_velocity': float(np.abs(data.qvel).max(initial=0)),
        'max_abs_velocity': float(np.abs(trace['qvel']).max(initial=0)),
        'activated_key_frames': int(trace['key_activation'].any(axis=1).sum()),
        'activated_keys': np.flatnonzero(trace['key_activation'].any(axis=0)).tolist(),
        'terminal_hand_position_error_m': {
            side: float(np.linalg.norm(data.body(hand['body']).xpos-origin['hands'][side]['position']))
            for side, hand in bundle.manifest['hands'].items()},
    }
    contract = dict(config, schema=1, method='kinematic_mocap_preparation_v1',
                    origin=origin, initial=initial, terminal=_named_pose(bundle, data),
                    report=report, policy_controls_preparation=False,
                    included_in_replay=False, included_in_score=False,
                    support_dynamics='kinematic; no arm dynamics or policy TZ joint')
    terminal = {name: getattr(data, name).copy() for name in
                ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'qacc_warmstart')}
    return contract, trace, terminal


def policy_initial(bundle, config):
    """Preserve origin and reset at raised initial; no precomputed policy motion."""
    validate_timing(bundle, config)
    model = bundle.model
    data = mujoco.MjData(model)
    for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
        getattr(data,name)[:] = getattr(bundle,'origin_'+name)
    mujoco.mj_forward(model,data)
    origin = _named_pose(bundle,data)
    for name in MOUNTS.values():
        index = model.body(name).mocapid[0]
        data.mocap_pos[index,2] += config['lift']
    mujoco.mj_forward(model,data)
    contract = dict(config, schema=1, method='policy_mocap_preparation_v1',
                    origin=origin, initial=_named_pose(bundle,data),
                    terminal_support_offset_world=[0.,0.,0.],
                    terminal_joint_positions='controlled by policy; never overwritten with origin',
                    policy_controls_preparation=True, included_in_replay=True,
                    included_in_score=True, score_prefix='empty notes and pedal, ordinary rewards',
                    policy_observation=dict(POLICY_OBSERVATION),
                    support_dynamics='kinematic; no arm dynamics or policy TZ joint')
    initial = {name:getattr(data,name).copy() for name in
               ('qpos','qvel','ctrl','mocap_pos','mocap_quat','qacc_warmstart')}
    return contract, None, initial


_SUPPORT_KERNEL = None


def support_kernel():
    """Lazy Warp import keeps the native-only Composer exporter independent."""
    global _SUPPORT_KERNEL, wp
    if _SUPPORT_KERNEL is not None:
        return _SUPPORT_KERNEL
    import warp as wp

    @wp.kernel
    def move_support(steps: wp.array(dtype=wp.int64),
                     positions: wp.array2d(dtype=wp.vec3),
                     origins: wp.array(dtype=wp.vec3),
                     right_id: int, left_id: int, decimation: int,
                     substep: int, preparation_steps: int, lift: float):
        world = wp.tid()
        elapsed = steps[world]*wp.int64(decimation)+wp.int64(substep)
        u = wp.min(wp.float32(elapsed)/wp.float32(preparation_steps),1.0)
        z = lift*(1.0-u*u*u*(10.0-15.0*u+6.0*u*u))
        right, left = origins[right_id], origins[left_id]
        positions[world,right_id] = wp.vec3(right[0],right[1],right[2]+z)
        positions[world,left_id] = wp.vec3(left[0],left[1],left[2]+z)

    _SUPPORT_KERNEL = move_support
    return _SUPPORT_KERNEL
