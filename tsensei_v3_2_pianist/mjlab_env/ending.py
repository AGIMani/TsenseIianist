"""Prescribed physical release/return after the scored episode, outside SAC."""
import numpy as np
from .first_note_preparation import motion_duration, quintic_targets, PreparationFailure


def plan(start, origin, wrists, velocity, acceleration, minimum, control_dt, physics_dt, settle):
    start, origin = np.asarray(start), np.asarray(origin)
    released = origin.copy()
    released[wrists] = start[wrists]
    release = motion_duration(released-start, velocity, acceleration, minimum=.5, control_dt=control_dt)
    duration = motion_duration(origin-released, velocity, acceleration, minimum=minimum, control_dt=control_dt)
    total = release+duration+settle
    if not np.isclose(round(total/physics_dt)*physics_dt,total,rtol=0,atol=1e-9):
        raise ValueError('Ending duration must contain whole physics steps')
    time = np.arange(round(total/physics_dt)+1)*physics_dt
    commands = quintic_targets(released,origin,time-release,duration)
    mask = time <= release
    commands[mask] = quintic_targets(start,released,time[mask],release)
    return time,commands,release


def append_trace(arrays, trace, dt):
    """Join policy to native return without duplicating or modifying the seam."""
    if not np.allclose(trace['time'],np.arange(len(trace['time']))*dt,rtol=0,atol=1e-8):
        raise ValueError('Ending physics times differ from recording')
    result = dict(arrays)
    for name in ('qpos','qvel','mocap_pos','mocap_quat'):
        if name not in arrays:continue
        if not np.array_equal(arrays[name][-1],trace[name][0]):
            raise ValueError('Ending discontinuity at '+name)
        result[name] = np.concatenate((arrays[name],trace[name][1:]),axis=0)
    result['sustain'] = np.concatenate((arrays['sustain'],np.zeros(len(trace['time'])-1)))
    return result


def generate(bundle, initial, context, minimum):
    """Continue actual state with native MuJoCo, never reset/project after t=0.

    Return targets are the original reference, including the documented reset
    coupling exception. They are not recomputed as first-note hover targets.
    """
    import mujoco
    import torch
    from .action_mapping import physical_bounds
    from .finger_coupling import FingerCoupler
    from ..finger_resistance import Resistance
    from .preparation import _named_pose
    model = bundle.model
    if not bundle.preparation or bundle.preparation.get('mode') != 'first_note_v1' or 'm6' in bundle.manifest:
        raise ValueError('Ending requires validated hand-only first_note_v1 preparation')
    data = mujoco.MjData(model)
    for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','qacc_warmstart'):
        getattr(data,name)[:] = getattr(initial,name)
    data.time = initial.time
    mujoco.mj_forward(model,data)
    ids=np.asarray(bundle.active_action_indices); aids=np.asarray(bundle.action_ids)
    names=[bundle.manifest['action_names'][i] for i in ids]
    wrists=np.array(['_Palm_Base_' in n for n in names])
    translation=np.array(['_Palm_Base_T' in n for n in names])
    velocity=np.where(translation,.3,3.); acceleration=np.where(translation,.6,15.)
    for i,name in enumerate(names):
        limit=(context.get('control_limits') or {}).get('channels',{}).get(name,{})
        velocity[i]=min(velocity[i],limit.get('max_velocity',velocity[i]))
        acceleration[i]=min(acceleration[i],limit.get('max_acceleration',acceleration[i]))
    dt=float(model.opt.timestep)
    times,commands,release=plan(data.ctrl[aids],bundle.origin_ctrl[aids],wrists,velocity,acceleration,
        minimum,bundle.manifest['control_dt'],dt,.5)
    joints=model.actuator_trnid[aids,0]; qadr=model.jnt_qposadr[joints]; vadr=model.jnt_dofadr[joints]
    resistance=Resistance(model)
    low,high=physical_bounds(context['action_profile'],bundle.manifest['action_names'],
        bundle.low,bundle.high,range_scales=bundle.manifest.get('action_range_scales'))
    coupler=FingerCoupler(bundle,device='cpu',options=context['finger_coupling'],physical_low=low,physical_high=high)
    trace={name:[] for name in ('time','qpos','qvel','ctrl','mocap_pos','mocap_quat','key_activation','coupling_actual_error_deg')}
    contract=dict(method='physical_return_origin_v1',target='original pose before preparation',
        duration=float(times[-1]),release_duration=release,minimum_return_duration=minimum,
        settle_time=.5,physics_dt=dt,included_in_replay=False,included_in_score=False,
        physics_backend='native MuJoCo continuation of actual terminal state',
        origin_ctrl=bundle.origin_ctrl.tolist(),initial=_named_pose(bundle,data),
        coupling='original reset/reference targets; no projection of actual coupling angles')
    def record(t):
        trace['time'].append(t)
        for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
            trace[name].append(getattr(data,name).copy())
        trace['key_activation'].append(bundle.key_activation(data.qpos))
        info=coupler.actual_diagnostics(torch.as_tensor(data.xmat[None],dtype=torch.float32))
        trace['coupling_actual_error_deg'].append(info['coupling_actual_error_deg'][0].numpy().copy())
    try:
        record(0.)
        for i in range(1,len(times)):
            data.ctrl[aids]=commands[i]
            resistance.update(data)
            mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            record(times[i])
            if not all(np.isfinite(getattr(data,n)).all() for n in ('qpos','qvel','ctrl')):
                raise FloatingPointError('Nonfinite ending physics')
    except Exception as error:
        contract['report']=dict(valid=False,positions_modified=False,error=str(error))
        raise PreparationFailure('Ending integration failed: '+str(error),contract,
            {k:np.asarray(v) for k,v in trace.items()}) from error
    trace={k:np.asarray(v) for k,v in trace.items()}
    actual=trace['qpos'][:,qadr]; ranges=model.jnt_range[joints]
    excursions=[]
    for col,joint in enumerate(joints):
        excess=np.maximum(ranges[col,0]-actual[:,col],actual[:,col]-ranges[col,1])
        bad=np.flatnonzero(excess>0)
        if len(bad):
            excursions.append(dict(name=model.joint(joint).name,maximum=float(excess[bad].max()),
                first_time=float(times[bad[0]]),last_time=float(times[bad[-1]]),samples=len(bad)))
    error=abs(data.qpos[qadr]-bundle.origin_qpos[qadr])
    active=trace['key_activation'].astype(bool)
    failures=[]
    if (error[wrists]>np.where(translation[wrists],.005,.01)).any():failures.append('original hand position/orientation arrival error')
    finger_residual={n:float(e) for n,e,w in zip(names,error,wrists) if not w and e>.05}
    if abs(data.qvel[vadr]).max()>.1:failures.append('terminal still moving')
    if active[-1].any():failures.append('keys still active')
    release_keys=np.flatnonzero(active[round(release/dt)])
    if len(release_keys):failures.append('keys still active before wrist return')
    report=dict(valid=not failures,failures=failures,positions_modified=False,
        joint_excursions=excursions,physical_hard_limits_satisfied=not excursions,
        arrival_scope='original wrist position within 5mm, rotation within .01rad, released keys and settled motion; all finger residuals retained',
        finger_reference_residuals_over_005rad=finger_residual,
        warnings=['Physical finger posture differs from original reset reference'] if finger_residual else [],
        terminal_position_error={n:float(e) for n,e in zip(names,error)},
        max_actual_coupling_error_deg=float(trace['coupling_actual_error_deg'].max()),
        terminal_actual_coupling_error_deg=float(trace['coupling_actual_error_deg'][-1].max()),
        max_actual_joint_velocity=float(abs(trace['qvel'][:,vadr]).max()),
        max_actual_joint_acceleration=float(abs(np.diff(trace['qvel'][:,vadr],axis=0)/dt).max()),
        max_terminal_velocity=float(abs(data.qvel[vadr]).max()),
        terminal_active_keys=np.flatnonzero(active[-1]).tolist(),
        active_keys_before_wrist_return=release_keys.tolist(),
        actual_note_on_count=int((active[1:] & ~active[:-1]).sum()),
        actual_note_on_count_after_release=int(((active[1:] & ~active[:-1])[times[1:]>release]).sum()),
        physics_steps=len(times)-1,physics_dt=dt)
    contract.update(report=report,terminal=_named_pose(bundle,data))
    if failures:raise PreparationFailure('Ending failed: '+', '.join(failures),contract,trace)
    return contract,trace
