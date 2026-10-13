"""Presentation timing for fixed initialization and policy-controlled preparation.

Only historical fixed preparation gets a separate video/audio prefix. Policy
preparation belongs to the actual GPU rollout, including its real key strikes.
"""
import numpy as np


def timeline(bundle, task_manifest=None):
    contract = getattr(bundle, 'preparation', None)
    mode = (contract or {}).get('mode', 'fixed_v1' if contract else 'none')
    policy_active = mode == 'policy_v1' or bool((contract or {}).get('policy_controls_preparation'))
    trace = getattr(bundle, 'preparation_trace', None)
    prefix = None if policy_active else trace
    environment = (task_manifest or {}).get('contract', {}).get('environment', {})
    return dict(mode=mode, policy_active=policy_active, prefix_trace=prefix,
                prefix_seconds=0. if prefix is None else float(prefix['time'][-1]),
                policy_preparation_seconds=float(contract['duration']) if policy_active else 0.,
                release_tail_steps=int((task_manifest or {}).get('release_tail_steps', 0)),
                music_end_step=(task_manifest or {}).get('music_end_step'),
                music_start_policy_seconds=float(environment.get('initial_buffer_time', 0.)))


def phase_at(policy_index, control_dt, timing):
    preparation_frames = round(timing['policy_preparation_seconds'] / control_dt)
    buffer_frames = round(timing['music_start_policy_seconds'] / control_dt)
    if timing.get('release_tail_steps', 0) and policy_index >= timing['music_end_step']:
        phase = 'release_tail'
    elif policy_index < preparation_frames:
        phase = 'preparation'
    elif policy_index < buffer_frames:
        phase = 'buffer'
    else:
        phase = 'music'
    return phase, policy_index - buffer_frames


def frame_indices(trace, control_dt, *, include_endpoint=False):
    times = np.asarray(trace['time'])
    count = round(float(times[-1]) / control_dt)
    if not np.isclose(count * control_dt, times[-1]):
        raise ValueError('Preparation must contain whole presentation frames')
    wanted = np.arange(count + int(include_endpoint)) * control_dt
    indices = np.searchsorted(times, wanted - 1e-9)
    if not np.allclose(times[indices], wanted, atol=1e-8):
        raise ValueError('Preparation trace does not contain presentation times')
    return indices


def snapshot(model, trace, index, data=None):
    import mujoco
    data = mujoco.MjData(model) if data is None else data
    for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat'):
        if name in trace:
            getattr(data, name)[:] = trace[name][index]
    data.time = float(trace['time'][index])
    mujoco.mj_forward(model, data)
    return data


def audio_prefix(bundle, key_qadr):
    trace = timeline(bundle)['prefix_trace']
    if trace is None:
        return np.empty((0, 88)), np.empty(0)
    # Audio samples describe substep ends. The t=0 state is not an extra sample.
    q = np.asarray(trace['qpos'])[1:, key_qadr]
    pedal = np.asarray(trace.get('sustain', np.zeros(len(trace['time']))))[1:]
    return q, pedal
