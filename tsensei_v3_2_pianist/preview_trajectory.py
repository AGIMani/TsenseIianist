# 2026-10-25: The hand base_link frame is named palm_base; exported pose coordinates explicitly name world.
"""Offline differential IK and time scaling, without stepping arm dynamics."""
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SIDES = ('right', 'left')
DEFAULT_LIMITS = {'linear_speed_m_s': 1., 'linear_acceleration_m_s2': 2.,
                  'angular_speed_rad_s': 1.5, 'angular_acceleration_rad_s2': 4.}


def jacobians(arms, side):
    p = np.zeros((3, arms.model.nv))
    r = np.zeros_like(p)
    mujoco.mj_jacSite(arms.model, arms.data, p, r, arms.sides[side]['sid'])
    return p, r


def differential_track(arms, positions, rotations):
    """Seed one bounded IK pose, then follow with damped Jacobian increments.

    No pointwise velocity clipping: original timing requirements are measured
    before applying a common time scale to the entire continuous joint path.
    """
    count = len(positions)
    if positions.shape != (count, 2, 3) or rotations.shape != (count, 2, 3, 3):
        raise ValueError('Expected N x two arms x position/rotation')
    weight = float(arms.config['orientation_weight_m'])
    rng = np.random.default_rng(17)
    arms.targets = {s: (positions[0, i], rotations[0, i]) for i, s in enumerate(SIDES)}
    for side in SIDES:
        arm = arms.sides[side]
        low, high = arm['limits'].T
        def residual(q):
            arms.data.qpos[arm['qadr']] = q
            mujoco.mj_forward(arms.model, arms.data)
            return arms._error(side)
        best = None
        for seed in [np.asarray(arms.config['seed_q_rad'][side])] + [rng.uniform(low*.8, high*.8) for _ in range(12)]:
            fit = least_squares(residual, np.clip(seed, low+1e-8, high-1e-8), bounds=(low, high),
                                max_nfev=200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                best = fit
            if np.linalg.norm(fit.fun) < 1e-7:
                break
        if np.linalg.norm(best.fun) > 1e-5:
            raise ValueError(f'{side}: initial TCP pose unreachable with this mount')
        arms.data.qpos[arm['qadr']] = best.x
    q = np.empty((count, arms.model.nq))
    errors = np.empty((count, 2, 2))
    for index in range(count):
        arms.targets = {s: (positions[index, i], rotations[index, i]) for i, s in enumerate(SIDES)}
        for i, side in enumerate(SIDES):
            arm = arms.sides[side]
            for _ in range(18):
                mujoco.mj_forward(arms.model, arms.data)
                error = arms._error(side)
                if np.linalg.norm(error) < 1e-8:
                    break
                jp, jr = jacobians(arms, side)
                j = np.vstack((jp[:, arm['dof']], weight*jr[:, arm['dof']]))
                delta = j.T @ np.linalg.solve(j @ j.T + np.eye(6)*1e-8, error)
                # Bound each numerical correction, then enforce vendor joint range.
                delta *= min(1., .08/max(float(np.max(np.abs(delta))), 1e-12))
                arms.data.qpos[arm['qadr']] = np.clip(arms.data.qpos[arm['qadr']] + delta,
                                                     arm['limits'][:, 0], arm['limits'][:, 1])
            mujoco.mj_forward(arms.model, arms.data)
            error = arms._error(side)
            errors[index, i] = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])/weight
        q[index] = arms.data.qpos
    if errors[:, :, 0].max() > .0001 or errors[:, :, 1].max() > .001:
        raise ValueError(f'IK tracking failed: max position {errors[:, :, 0].max():.6g} m, angle {errors[:, :, 1].max():.6g} rad')
    return q, errors


def path_dynamics(arms, q, qd, qdd):
    """World TCP velocity/acceleration via J qdot and J qddot + Jdot qdot."""
    vectors = np.empty((len(q), 2, 4, 3))
    tcp = np.empty((len(q), 2, 3))
    epsilon = 1e-5  # directional derivative time increment, seconds
    for k in range(len(q)):
        arms.data.qpos[:] = q[k]
        mujoco.mj_forward(arms.model, arms.data)
        base = {s: np.vstack(jacobians(arms, s)) for s in SIDES}
        for i, side in enumerate(SIDES):
            tcp[k, i] = arms.data.site_xpos[arms.sides[side]['sid']]
        shifted = []
        for sign in (-1, 1):
            arms.data.qpos[:] = q[k] + sign*epsilon*qd[k]
            mujoco.mj_forward(arms.model, arms.data)
            shifted.append({s: np.vstack(jacobians(arms, s)) for s in SIDES})
        for i, side in enumerate(SIDES):
            velocity = base[side] @ qd[k]
            acceleration = base[side] @ qdd[k] + ((shifted[1][side]-shifted[0][side])/(2*epsilon)) @ qd[k]
            vectors[k, i] = velocity[:3], acceleration[:3], velocity[3:], acceleration[3:]
    return tcp, vectors


def scale_path(time, q, arms, limits=None, audit_hz=1000):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or not all(np.isfinite(v) and v > 0 for v in limits.values()):
        raise ValueError('Four positive finite TCP limits are required')
    # C2 path, with stationary endpoints. Retiming also covers start/stop ramps.
    spline = CubicSpline(time, q, bc_type=((1, np.zeros(q.shape[1])), (1, np.zeros(q.shape[1]))))
    samples = np.linspace(time[0], time[-1], int(np.ceil((time[-1]-time[0])*audit_hz))+1)
    position, velocity, acceleration = (spline(samples, derivative) for derivative in (0, 1, 2))
    for side in SIDES:
        arm = arms.sides[side]
        if np.any(position[:, arm['qadr']] < arm['limits'][:, 0]-1e-8) or np.any(position[:, arm['qadr']] > arm['limits'][:, 1]+1e-8):
            raise ValueError('Interpolated path exceeds vendor joint limits: ' + side)
    tcp, vectors = path_dynamics(arms, position, velocity, acceleration)
    norms = np.linalg.norm(vectors, axis=-1)
    bounds = np.array([limits[key] for key in DEFAULT_LIMITS])
    maximum = norms.max(axis=(0, 1))
    factor = max(maximum[0]/bounds[0], np.sqrt(maximum[1]/bounds[1]),
                 maximum[2]/bounds[2], np.sqrt(maximum[3]/bounds[3]))
    for side in SIDES:
        arm = arms.sides[side]
        factor = max(factor, float(np.max(np.abs(velocity[:, arm['dof']])/arm['velocity'])))
    factor = max(1., 1.02*factor)
    limited_vectors = vectors / np.array([factor, factor**2, factor, factor**2])[None, None, :, None]
    # For a uniform time scale, these relations are exact for the entire spline.
    limited_norms = np.linalg.norm(limited_vectors, axis=-1)
    if np.any(limited_norms.max(axis=(0, 1)) > bounds*(1+1e-9)):
        raise AssertionError('Time scaling did not satisfy the sampled TCP limits')
    report = {'time_scale': float(factor), 'original_duration_s': float(time[-1]),
              'limited_duration_s': float(time[-1]*factor), 'limits': limits,
              'audit_hz_in_original_time': audit_hz, 'extra_scale_margin': .02,
              'path_interpolation': 'C2 cubic spline with zero endpoint joint velocities',
              'palm_base_frame': 'hand base_link origin; saved Link7-to-hand tool transform; not the flange origin',
              'maximum_original': {}, 'maximum_limited': {}, 'maximum_joint_speed_rad_s': {}}
    for i, side in enumerate(SIDES):
        report['maximum_original'][side] = dict(zip(DEFAULT_LIMITS, norms[:, i].max(0).tolist()))
        report['maximum_limited'][side] = dict(zip(DEFAULT_LIMITS, limited_norms[:, i].max(0).tolist()))
        report['maximum_joint_speed_rad_s'][side] = {
            'original': np.abs(velocity[:, arms.sides[side]['dof']]).max(0).tolist(),
            'limited': (np.abs(velocity[:, arms.sides[side]['dof']]).max(0)/factor).tolist(),
            'vendor_limits': arms.sides[side]['velocity'].tolist()}
    return spline, report, {'time': samples, 'q': position, 'qd': velocity, 'qdd': acceleration,
                            'tcp_position': tcp, 'tcp_vectors': vectors, 'limited_tcp_vectors': limited_vectors}
