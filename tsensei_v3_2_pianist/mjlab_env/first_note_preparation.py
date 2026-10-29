"""Deterministic first-note hover commands followed by unprojected physics.

IK uses separate scratch data. Recorded qpos/qvel are never assigned IK values
or clipped after integration. This physical prefix is outside policy replay.
"""
import math
import numpy as np


def first_notes(events):
    events = np.asarray(events, dtype=float)
    if (events.ndim != 2 or events.shape[1:] != (4,) or not len(events)
            or not np.isfinite(events).all() or (events[:, 0] < 0).any()
            or (events[:, 1] <= events[:, 0]).any()
            or (events[:, 2:] != np.floor(events[:, 2:])).any()
            or ((events[:, 2] < 0) | (events[:, 2] >= 88)).any()
            or ((events[:, 3] < 0) | (events[:, 3] >= 10)).any()):
        raise ValueError('First-note preparation requires finite exact notes with complete fingering')
    selected = []
    for side in range(2):
        hand = events[(events[:, 3] // 5) == side]
        if not len(hand):
            continue
        onset = hand[:, 0].min()
        chord = hand[np.abs(hand[:, 0] - onset) <= 1e-9]
        if len(set(chord[:, 3])) != len(chord):
            raise ValueError('First chord assigns multiple notes to the same finger')
        if len(set(chord[:, 2])) != len(chord):
            raise ValueError('First chord contains duplicate keys')
        selected.extend(sorted(chord.tolist(), key=lambda row: row[3]))
    return selected


def motion_duration(delta, velocity, acceleration, *, minimum, control_dt):
    delta, velocity, acceleration = map(lambda x: np.asarray(x, dtype=float), (delta, velocity, acceleration))
    if (not np.isfinite(delta).all() or not np.isfinite(velocity).all()
            or not np.isfinite(acceleration).all() or (velocity <= 0).any()
            or (acceleration <= 0).any() or not math.isfinite(minimum) or minimum <= 0
            or not math.isfinite(control_dt) or control_dt <= 0):
        raise ValueError('Finite positive duration and motion bounds required')
    # Maxima of the quintic derivative and second derivative, respectively.
    duration = max(minimum, float(np.max(1.875 * abs(delta) / velocity)),
                   float(np.max(np.sqrt((10 / math.sqrt(3)) * abs(delta) / acceleration))))
    return math.ceil(duration / control_dt) * control_dt


def quintic_targets(start, end, times, duration):
    u = np.clip(np.asarray(times, dtype=float) / duration, 0., 1.)
    blend = u**3 * (10 - 15*u + 6*u*u)
    return np.asarray(start) + blend[..., None] * (np.asarray(end) - start)


def departure_targets(start, end, times, duration):
    """Seven-segment S curve with continuous velocity and acceleration."""
    u = np.clip(np.asarray(times, dtype=float) / duration, 0., 1.)
    blend = np.zeros_like(u)
    position = velocity = acceleration = elapsed = 0.
    for width, jerk in zip((.04, .42, .04, 0., .04, .42, .04),
                           (25., 0., -25., 0., -25., 0., 25.)):
        local = np.clip(u-elapsed, 0., width)
        blend += velocity*local + .5*acceleration*local**2 + jerk*local**3/6
        position += velocity*width + .5*acceleration*width**2 + jerk*width**3/6
        velocity += acceleration*width + .5*jerk*width**2
        acceleration += jerk*width
        elapsed += width
    blend = np.where(u >= 1., 1., blend/position)
    return np.asarray(start) + blend[..., None]*(np.asarray(end)-start)


class PreparationFailure(ValueError):
    def __init__(self, message, contract, trace):
        super().__init__(message)
        self.contract, self.trace = contract, trace


def generate(bundle, config, context):
    import mujoco
    import torch
    from scipy.optimize import least_squares
    from .action_mapping import physical_bounds
    from .finger_coupling import FingerCoupler
    from ..finger_resistance import Resistance
    from ..fingering_targets import target_for_key, key_reference
    from .preparation import _named_pose
    if not context or context.get('action_profile') != 'direct_joint_v1' or not context.get('finger_coupling'):
        raise ValueError('First-note preparation requires direct_joint_v1 and explicit finger coupling')
    if 'm6' in bundle.manifest or bundle.model.nmocap:
        raise ValueError('First-note preparation currently validated only for hand-only scenes without mocap')
    source = bundle.manifest.get('first_note_preparation')
    if not source:
        raise ValueError('First-note preparation requires source-compiled exact score metadata')
    from .preparation import validate_timing
    validate_timing(bundle, config)
    validate_timing(bundle, dict(config, duration=config['settle_time']))
    rows = first_notes(source['notes'])
    model = bundle.model
    low, high = map(np.asarray, physical_bounds(context['action_profile'], bundle.manifest['action_names'],
        bundle.low, bundle.high, range_scales=bundle.manifest.get('action_range_scales')))
    ids = np.asarray(bundle.active_action_indices)
    aids = np.asarray(bundle.action_ids)
    joints = model.actuator_trnid[aids, 0]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    origin = bundle.neutral_action.copy()
    origin[ids] = bundle.origin_ctrl[aids]
    if ((origin[ids] < low[ids]-1e-8) | (origin[ids] > high[ids]+1e-8)).any():
        raise ValueError('Original preparation command outside action bounds')
    coupler = FingerCoupler(bundle, device='cpu', options=context['finger_coupling'], physical_low=low, physical_high=high)
    follower = set(coupler.indices[:, 1:].reshape(-1).tolist())
    independent = np.array([i for i in ids if i not in follower])
    active_hands = {int(row[3]) // 5 for row in rows}
    active_fingers = {int(row[3]) for row in rows}
    independent = np.array([i for i in independent if i // 26 in active_hands
                            and (i % 26 >= 20 or (i // 26)*5 + (i % 26)//4 in active_fingers)])
    scratch = bundle.reset_cpu()
    tips = [model.site(name).id for side in ('right','left') for name in bundle.manifest['hands'][side]['sites']]
    selected = np.array([int(row[3]) for row in rows])
    keyids = [model.geom(source['key_geoms'][int(row[2])]).id for row in rows]
    wanted = np.array([target_for_key(model, scratch, gid, task_profile=source['task_profile']) for gid in keyids])
    for i, gid in enumerate(keyids):
        sid, size = key_reference(model, gid)
        wanted[i, 2] = scratch.site_xpos[sid, 2] + size[2] + config['hover']
    names = bundle.manifest['action_names']

    def coupled(command, data):
        with torch.no_grad():
            result, info = coupler.apply(torch.as_tensor(command[None], dtype=torch.float32),
                actual_qpos=torch.as_tensor(data.qpos[None], dtype=torch.float32),
                body_xmat=torch.as_tensor(data.xmat[None], dtype=torch.float32))
        solved = command.copy()
        follower_ids = coupler.indices[:, 1:].reshape(-1).numpy()
        solved[follower_ids] = result[0].numpy()[follower_ids]
        return solved, info

    def pose(values):
        command = origin.copy(); command[independent] = values
        scratch.qpos[:] = bundle.origin_qpos
        scratch.qpos[qadr] = command[ids]
        mujoco.mj_forward(model, scratch)
        command, info = coupled(command, scratch)
        scratch.qpos[qadr] = command[ids]
        mujoco.mj_forward(model, scratch)
        return command, info

    solve_low, solve_high = low[independent].copy(), high[independent].copy()
    # Avoid resting the final wrist on a compliant joint stop. This only
    # constrains the scratch IK target, never projects actual joint states.
    for col, index in enumerate(independent):
        if names[index].endswith(('_Pitch', '_Roll')):
            solve_low[col] += .01
            solve_high[col] -= .01
    scale = solve_high - solve_low
    initial = np.maximum(solve_low+1e-8, np.minimum(solve_high-1e-8, origin[independent]))

    def residual(values):
        _, info = pose(values)
        errors = [(scratch.site_xpos[np.array(tips)[selected]] - wanted).ravel()]
        # Unused fingers retain their original joint targets and move with the
        # wrist; pinning their world positions would splay them into neighbours.
        errors.append(.001*(values-initial)/scale)
        errors.append(.001*info['coupling_target_error_deg'][0].numpy())
        return np.concatenate(errors)

    def jacobian(values):
        columns = []
        for col, index in enumerate(independent):
            epsilon = 1e-5 if '_Palm_Base_T' in names[index] else 1e-4
            lower, upper = values.copy(), values.copy()
            lower[col] = max(solve_low[col], values[col]-epsilon)
            upper[col] = min(solve_high[col], values[col]+epsilon)
            columns.append((residual(upper)-residual(lower))/(upper[col]-lower[col]))
        return np.stack(columns, axis=1)

    best = None
    for shift in (0., .15, -.15):
        seed = np.maximum(solve_low+1e-8, np.minimum(solve_high-1e-8, initial+shift*scale))
        fit = least_squares(residual, seed, jac=jacobian, bounds=(solve_low, solve_high), max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        target, info = pose(fit.x)
        error = np.linalg.norm(scratch.site_xpos[np.array(tips)[selected]]-wanted, axis=1)
        if best is None or error.max() < best[0]:
            best = (float(error.max()), target.copy(), error.copy())
        if best[0] <= .002:
            break
    error_max, target, ik_error = best
    if error_max > config['position_tolerance']:
        raise ValueError(f'First-note hover IK unreachable within {config["position_tolerance"]} m: {ik_error.tolist()}')

    velocity = np.full(53, 3.); acceleration = np.full(53, 15.)
    for i, name in enumerate(names):
        if '_Palm_Base_T' in name:
            velocity[i], acceleration[i] = .3, .6
        elif '_Palm_Base_' not in name:
            velocity[i], acceleration[i] = 3., 15.
    for i, name in enumerate(names):
        if i not in ids:
            continue  # Fixed/baked wrist axes are not actuator targets.
        row = (context.get('control_limits') or {}).get('channels', {}).get(name, {})
        velocity[i] = min(velocity[i], row.get('max_velocity', velocity[i]))
        acceleration[i] = min(acceleration[i], row.get('max_acceleration', acceleration[i]))
        if target[i] < row.get('min', low[i])-1e-8 or target[i] > row.get('max', high[i])+1e-8:
            raise ValueError(f'First-note target outside configured control bounds: {name} value={target[i]} range={row.get("min",low[i])},{row.get("max",high[i])}')
    # Leave the lower wrist stop before the longer lateral approach. Both
    # phases have zero target velocity/acceleration at their join; physics
    # continues without resetting or projecting any state.
    departure = origin.copy()
    for i in ids:
        if i // 26 in active_hands and names[i].endswith('_Pitch'):
            departure[i] = min(high[i]-.01, max(origin[i], low[i]+.008))
    departure_minimum = max(.05, float(np.max(2*abs(departure-origin)/velocity)),
        float(np.max(np.sqrt(abs(departure-origin)/(.23*acceleration)))))
    departure_duration = math.ceil(departure_minimum/bundle.manifest['control_dt'])*bundle.manifest['control_dt']
    duration = motion_duration(target-departure, velocity, acceleration,
        minimum=config['duration'], control_dt=bundle.manifest['control_dt'])
    dt = float(model.opt.timestep)
    count = round((departure_duration + duration + config['settle_time']) / dt)
    times = np.arange(count+1)*dt
    requests = quintic_targets(departure, target, times-departure_duration, duration)
    departing = times <= departure_duration
    requests[departing] = departure_targets(origin, departure, times[departing], departure_duration)
    data = bundle.reset_cpu()
    initial_pose = _named_pose(bundle, data)
    resistance = Resistance(model)
    trace = {name: [] for name in ('time','qpos','qvel','ctrl','mocap_pos','mocap_quat','key_activation','fingertips','contacts','coupling_actual_error_deg','coupling_target_error_deg')}
    hand_geoms = np.array([(model.geom(i).name or '').startswith(('right_/','left_/')) for i in range(model.ngeom)])
    key_geoms = np.array([(model.geom(i).name or '').startswith('piano/') and 'key_geom_' in (model.geom(i).name or '') for i in range(model.ngeom)])

    def record(t, target_error=None):
        trace['time'].append(t)
        trace['coupling_target_error_deg'].append(np.zeros(8) if target_error is None else target_error)
        for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
            trace[name].append(getattr(data,name).copy())
        trace['key_activation'].append(bundle.key_activation(data.qpos))
        trace['fingertips'].append(data.site_xpos[tips].copy())
        pairs = np.asarray(data.contact.geom[:data.ncon])
        count_contact = sum(int((hand_geoms[a] and key_geoms[b]) or (hand_geoms[b] and key_geoms[a])) for a,b in pairs)
        trace['contacts'].append(count_contact)
        info = coupler.actual_diagnostics(torch.as_tensor(data.xmat[None],dtype=torch.float32))
        trace['coupling_actual_error_deg'].append(info['coupling_actual_error_deg'][0].numpy().copy())

    try:
        record(0.)
        for step in range(1,count+1):
            command, target_info = coupled(requests[step],data)
            data.ctrl[aids] = command[ids]
            resistance.update(data)
            mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            record(times[step],target_info['coupling_target_error_deg'][0].numpy().copy())
            if not all(np.isfinite(getattr(data,name)).all() for name in ('qpos','qvel','ctrl')):
                raise FloatingPointError('Nonfinite first-note preparation physics')
    except Exception as error:
        partial = {name:np.asarray(value) for name,value in trace.items()}
        contract = dict(config, method='first_note_physical_hover_v1',source=source,
            report=dict(valid=False,failures=['integration failure'],error=str(error),
                        positions_modified=False,recorded_samples=len(trace['time']),physics_dt=dt))
        raise PreparationFailure('First-note preparation integration failed: '+str(error),contract,partial) from error
    trace = {name:np.asarray(value) for name,value in trace.items()}
    terminal_error = np.linalg.norm(data.site_xpos[np.array(tips)[selected]]-wanted,axis=1)
    actual = trace['qpos'][:,qadr]
    ranges = model.jnt_range[joints]
    violation = np.maximum(ranges[:,0]-actual,actual-ranges[:,1]).max(initial=0.)
    report = dict(ik_error_m=ik_error.tolist(),terminal_error_m=terminal_error.tolist(),
        max_terminal_velocity=float(abs(data.qvel[vadr]).max()),
        activated_key_samples=int(trace['key_activation'].any(axis=1).sum()),
        hand_key_contact_samples=int(np.count_nonzero(trace['contacts'])),
        max_joint_limit_violation=float(violation),
        max_actual_coupling_error_deg=float(trace['coupling_actual_error_deg'].max()),
        terminal_actual_coupling_error_deg=float(trace['coupling_actual_error_deg'][-1].max()),
        max_target_coupling_error_deg=float(trace['coupling_target_error_deg'].max()),
        positions_modified=False, physics_steps=count, physics_dt=dt)
    failures=[]
    from .control_limits import canonical, resolve
    from .joint_limits import resolve_robot_limits
    controls = resolve(canonical(context.get('control_limits')),action_names=names,
        active_indices=getattr(bundle,'policy_action_indices',ids),physical_low=low,physical_high=high,
        reset_action=target,control_dt=bundle.manifest['control_dt'])
    reset_bounds = resolve_robot_limits(model,key_ids=bundle.key_ids,action_ids=aids,
        active_indices=ids,physical_low=low,physical_high=high,control_contract=controls)
    incompatible = []
    for row in reset_bounds['joints']:
        value = float(np.float32(data.qpos[row['qpos_address']]))
        velocity_at_boundary = float(data.qvel[row['dof_address']])
        if (value < row['float32_min'] or value > row['float32_max']
                or (value <= row['float32_min'] and velocity_at_boundary < 0)
                or (value >= row['float32_max'] and velocity_at_boundary > 0)):
            incompatible.append(row['name'])
    report['reset_projection_required_joints'] = incompatible
    if incompatible:failures.append('terminal incompatible with unmodified GPU reset')
    if report['terminal_actual_coupling_error_deg'] > coupler.TOLERANCE_DEG:
        failures.append('terminal actual coupling violation')
    if report['max_target_coupling_error_deg'] > coupler.TOLERANCE_DEG:
        failures.append('infeasible coupled command')
    if terminal_error.max() > config['position_tolerance']:failures.append('terminal position error')
    if report['activated_key_samples']:failures.append('actual key activation during preparation')
    if report['hand_key_contact_samples']:failures.append('hand/key contact during preparation')
    # Transient physical excursions remain measurements, never projected away.
    # Arrival/reset validity is checked separately above. This is a hover
    # acceptance report, not a claim of physical hard-limit compliance.
    report['joint_excursions'] = []
    for col,joint in enumerate(joints):
        excess = np.maximum(ranges[col,0]-actual[:,col],actual[:,col]-ranges[col,1])
        bad = np.flatnonzero(excess > 0.)
        if len(bad):
            report['joint_excursions'].append(dict(name=model.joint(joint).name,
                first_time=float(times[bad[0]]),last_time=float(times[bad[-1]]),samples=int(len(bad)),
                maximum=float(excess[bad].max()),minimum_position=float(actual[:,col].min()),
                maximum_position=float(actual[:,col].max())))
    report['physical_hard_limits_satisfied'] = not report['joint_excursions']
    report['validation_scope'] = 'hover arrival, no key contact, feasible commands, unmodified terminal handoff'
    report['warnings'] = ['Actual transient joint excursions retained; see joint_excursions'] if report['joint_excursions'] else []
    command_v = np.diff(trace['ctrl'][:, aids], axis=0)/dt
    command_a = np.diff(trace['ctrl'][:, aids], n=2, axis=0)/dt**2
    report['command_motion'] = {names[i]: dict(max_velocity=float(abs(command_v[:,col]).max()),
        max_acceleration=float(abs(command_a[:,col]).max()), velocity_limit=float(velocity[i]),
        acceleration_limit=float(acceleration[i])) for col,i in enumerate(ids)}
    if (abs(command_v).max(axis=0) > velocity[ids]+1e-5).any():failures.append('command velocity bound')
    if (abs(command_a).max(axis=0) > acceleration[ids]+1e-3).any():failures.append('command acceleration bound')
    if report['max_terminal_velocity'] > .05:failures.append('terminal still moving')
    report.update(valid=not failures,failures=failures)
    contract = dict(config, method='first_note_physical_hover_v1',duration=float(times[-1]),
        requested_duration=config['duration'],motion_duration=duration,departure_duration=departure_duration,
        departure_targets=departure.tolist(),departure_trajectory='continuous_acceleration_s_curve_v1',source=source,
        targets_m=wanted.tolist(),selected_fingers=selected.tolist(),
        control_context=context,origin=initial_pose,initial=initial_pose,terminal=_named_pose(bundle,data),
        report=report,policy_controls_preparation=False,included_in_replay=False,included_in_score=False,
        reset_semantics='cached actual physical terminal; full original prefix retained',
        support_dynamics='fixed hand mounting; no mocap movement or qpos projection')
    terminal={name:getattr(data,name).copy() for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','qacc_warmstart')}
    if failures:
        raise PreparationFailure('First-note preparation failed: '+', '.join(failures),contract,trace)
    return contract,trace,terminal
