"""Fixed-time M6 trajectory optimization with continuous joint motion bounds.

The cubic B-spline and its first two derivatives have nonnegative partition-
of-unity bases. Bounding their coefficients bounds every instant, including
between recorded frames. Wrist errors use absolute FK targets, not integration.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

SIDES = ('right', 'left')


def spline_basis(time, knot_dt=.01):
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all() or
            np.any(np.diff(time) <= 0) or not np.isfinite(knot_dt) or knot_dt <= 0):
        raise ValueError('Require at least three finite increasing timestamps and positive knot spacing')
    # Preserve the requested grid (typically four physics substeps per span).
    # Distributing a short final span across every knot shifts the control grid.
    duration = time[-1]-time[0]
    if duration < 4*knot_dt:
        nodes = np.linspace(time[0], time[-1], 5)
    else:
        nodes = time[0]+np.arange(int(np.ceil(duration/knot_dt)))*knot_dt
        nodes = np.r_[nodes[nodes < time[-1]-1e-12], time[-1]]
    knots = np.r_[np.repeat(nodes[0], 4), nodes[1:-1], np.repeat(nodes[-1], 4)]
    count = len(knots)-4
    basis = sparse.csc_matrix(BSpline.design_matrix(time, knots, 3))
    difference = sparse.diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count))
    velocity = sparse.diags(3/(knots[4:count+3]-knots[1:count])) @ difference
    difference = sparse.diags([-np.ones(count-2), np.ones(count-2)], [0, 1], shape=(count-2, count-1))
    acceleration = sparse.diags(2/(knots[4:count+2]-knots[2:count])) @ difference @ velocity
    return knots, basis, sparse.csc_matrix(velocity), sparse.csc_matrix(acceleration)


def certify_coefficients(coefficients, velocity_matrix, acceleration_matrix, position_bounds,
                         velocity_limits, acceleration_limits):
    c = np.asarray(coefficients)
    v, a = velocity_matrix @ c, acceleration_matrix @ c
    if not all(np.isfinite(x).all() for x in (c, v, a)):
        raise ValueError('Nonfinite planned trajectory')
    margin = np.minimum(c-position_bounds[:, 0], position_bounds[:, 1]-c).min(axis=0)
    v_peak, a_peak = np.abs(v).max(axis=0), np.abs(a).max(axis=0)
    passed = bool(np.all(margin >= -1e-9) and np.all(v_peak <= velocity_limits) and
                  np.all(a_peak <= acceleration_limits))
    return dict(passed=passed, minimum_position_margin_rad=margin.tolist(),
                velocity_bound_rad_s=v_peak.tolist(), acceleration_bound_rad_s2=a_peak.tolist(),
                velocity_limit_fraction=(v_peak/velocity_limits).tolist(),
                acceleration_limit_fraction=(a_peak/acceleration_limits).tolist(),
                method='B-spline derivative coefficient convex-hull bound over every interval; not a grid-only check')


def pose_errors(arms, q, positions, rotations):
    errors = np.empty((len(q), 2, 2))
    for k, angles in enumerate(q):
        arms.data.qpos[:] = angles
        mujoco.mj_forward(arms.model, arms.data)
        for i, side in enumerate(SIDES):
            sid = arms.sides[side]['sid']
            errors[k, i] = (
                np.linalg.norm(arms.data.site_xpos[sid]-positions[k, i]),
                np.linalg.norm(Rotation.from_matrix(
                    rotations[k, i] @ arms.data.site_xmat[sid].reshape(3, 3).T).as_rotvec()))
    return errors


def solve_fixed_time(arms, time, seed, positions, rotations, limits, *,
                     position_tolerance_m=.0001, rotation_tolerance_rad=np.deg2rad(.5), knot_dt=.01):
    """Fit a C2 path at original times; fail if motion or pose checks do not pass.

    SDK bounds have a 2% numerical margin in the optimization. Initial/final
    joint coordinates are retained, with zero velocity and acceleration, so
    the path joins stationary poses without a velocity jump.
    """
    try:
        import osqp
    except ImportError as exc:
        raise RuntimeError('Constrained IK requires preview/requirements-replay.txt (osqp)') from exc
    time = np.asarray(time, dtype=float)
    seed = np.asarray(seed, dtype=float)
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    if (seed.shape != (len(time), arms.model.nq) or arms.model.nq != 14 or
            positions.shape != (len(time), 2, 3) or rotations.shape != (len(time), 2, 3, 3) or
            not all(np.isfinite(x).all() for x in (seed, positions, rotations)) or
            not all(np.isfinite(x) and x > 0 for x in (position_tolerance_m, rotation_tolerance_rad))):
        raise ValueError('Invalid M6 targets, seed or wrist error tolerances')
    knots, B, D1, D2 = spline_basis(time, knot_dt)
    m, n = B.shape[1], len(time)
    h = float(np.median(np.diff(np.unique(knots))))
    A0 = sparse.kron(sparse.vstack([sparse.eye(m), h*D1, h*h*D2]), sparse.eye(7), format='csc')
    B7 = sparse.kron(B, sparse.eye(7), format='csc')
    branch_cost = B7.T @ B7
    orientation_weight = .03*(position_tolerance_m/.0001)/(rotation_tolerance_rad/np.deg2rad(.5))
    weight = np.array([1., 1., 1., *([orientation_weight]*3)])
    coefficients = np.empty((m, 14))
    side_reports = {}
    all_bounds, all_v, all_a = np.empty((14, 2)), np.empty(14), np.empty(14)

    for side_index, side in enumerate(SIDES):
        arm = arms.sides[side]
        ids = arm['qadr']
        original = seed[:, ids]
        rows = [limits['joints'][f'{side}_J{j}'] for j in range(1, 8)]
        vl = np.array([row['velocity'] for row in rows])
        al = np.array([row['acceleration'] for row in rows])
        all_bounds[ids], all_v[ids], all_a[ids] = arm['limits'], vl, al
        low = np.r_[np.tile(arm['limits'][:, 0], m), np.tile(-.98*h*vl, m-1), np.tile(-.98*h*h*al, m-2)]
        high = np.r_[np.tile(arm['limits'][:, 1], m), np.tile(.98*h*vl, m-1), np.tile(.98*h*h*al, m-2)]
        low[:21] = high[:21] = np.tile(original[0], 3)
        low[m*7-21:m*7] = high[m*7-21:m*7] = np.tile(original[-1], 3)
        current, c = original.copy(), None
        iterations = []
        accepted = False
        # First improve the linearization with a motion-bounded Cartesian fit;
        # subsequent passes also impose absolute wrist error tubes.
        for iteration in range(6):
            blocks, desired = [], []
            jp, jr = np.zeros((3, 14)), np.zeros((3, 14))
            for k in range(n):
                arms.data.qpos[:] = seed[k]
                arms.data.qpos[ids] = current[k]
                mujoco.mj_forward(arms.model, arms.data)
                sid = arm['sid']
                error = np.r_[positions[k, side_index]-arms.data.site_xpos[sid],
                              Rotation.from_matrix(rotations[k, side_index] @
                                                   arms.data.site_xmat[sid].reshape(3, 3).T).as_rotvec()]
                mujoco.mj_jacSite(arms.model, arms.data, jp, jr, sid)
                J = np.vstack([jp[:, arm['dof']], jr[:, arm['dof']]])*weight[:, None]
                blocks.append(sparse.csc_matrix(J))
                desired.append(J @ current[k] + error*weight)
            G = sparse.block_diag(blocks, format='csc') @ B7
            y = np.concatenate(desired)
            P = sparse.csc_matrix(2*(G.T @ G + 1e-5*branch_cost))
            objective = -2*(G.T @ y + 1e-5*B7.T @ original.ravel())
            A, lower, upper = A0, low, high
            if iteration:
                tube = np.tile(np.r_[np.full(3, .95*position_tolerance_m/np.sqrt(3)),
                                     np.full(3, .95*orientation_weight*rotation_tolerance_rad)], n)
                A = sparse.vstack([A0, G], format='csc')
                trust_low, trust_high = low.copy(), high.copy()
                trust_low[:m*7] = np.maximum(trust_low[:m*7], c.ravel()-.1)
                trust_high[:m*7] = np.minimum(trust_high[:m*7], c.ravel()+.1)
                lower, upper = np.r_[trust_low, y-tube], np.r_[trust_high, y+tube]
            model = osqp.OSQP()
            model.setup(P=P, q=objective, A=A, l=lower, u=upper, eps_abs=1e-9, eps_rel=1e-7,
                        eps_prim_inf=1e-10, eps_dual_inf=1e-10, max_iter=200000, time_limit=90,
                        polishing=True, verbose=False)
            if c is not None:
                model.warm_start(x=c.ravel())
            result = model.solve(raise_error=False)
            if result.x is None or result.info.status not in ('solved', 'solved inaccurate'):
                raise ValueError(f'{side} constrained IK did not converge: {result.info.status}; '
                                 'original tempo and requested limits were not relaxed')
            c = result.x.reshape(m, 7)
            bounded = np.clip(c, arm['limits'][:, 0], arm['limits'][:, 1])
            bound_roundoff = float(np.max(np.abs(c-bounded)))
            if bound_roundoff > 1e-6:
                raise ValueError(f'{side} solver exceeded a joint bound by {bound_roundoff} rad')
            c = bounded
            # Remove only equality-constraint roundoff at the stationary endpoints;
            # the complete trajectory is independently certified again below.
            c[:3], c[-3:] = original[0], original[-1]
            candidate = seed.copy()
            candidate[:, ids] = B @ c
            errors = pose_errors(arms, candidate, positions, rotations)[:, side_index]
            certificate = certify_coefficients(c, D1, D2, arm['limits'], vl, al)
            row = dict(iteration=iteration, solver_status=result.info.status, solver_iterations=result.info.iter,
                       coefficient_bound_roundoff_rad=bound_roundoff,
                       solver_seconds=result.info.run_time, maximum_position_error_m=float(errors[:, 0].max()),
                       maximum_rotation_error_rad=float(errors[:, 1].max()), motion_certificate=certificate)
            iterations.append(row)
            print(f'{side} constrained IK {iteration}: position {errors[:,0].max()*1000:.4g} mm, '
                  f'rotation {np.rad2deg(errors[:,1].max()):.4g} deg; motion={certificate["passed"]}', flush=True)
            if not certificate['passed']:
                raise ValueError(f'{side} candidate failed independent continuous motion bounds: {certificate}')
            if (certificate['passed'] and errors[:, 0].max() <= position_tolerance_m and
                    errors[:, 1].max() <= rotation_tolerance_rad):
                accepted = True
                break
            current = candidate[:, ids]
        if not accepted:
            raise ValueError(f'{side} cannot meet the requested wrist errors in this optimization: '
                             f'{errors[:,0].max()*1000:.6g} mm, {np.rad2deg(errors[:,1].max()):.6g} deg. '
                             'No raw or retimed fallback was exported.')
        coefficients[:, ids] = c
        side_reports[side] = iterations
    path = BSpline(knots, coefficients, 3, extrapolate=False)
    q = path(time)
    errors = pose_errors(arms, q, positions, rotations)
    certificate = certify_coefficients(coefficients, D1, D2, all_bounds, all_v, all_a)
    if not certificate['passed']:
        raise ValueError('Combined trajectory failed continuous motion bounds')
    report = dict(schema=1, mode='fixed_time_constrained_bspline_v1', original_timing_preserved=True,
                  time_scale=1., position_tolerance_m=position_tolerance_m,
                  rotation_tolerance_rad=rotation_tolerance_rad, knot_dt_s=h,
                  interpolation='C2 cubic B-spline; renderer evaluates this same spline',
                  endpoint_conditions='original joint positions; zero velocity and acceleration at both ends',
                  motion_certificate=certificate, reference_limits=limits, solver_iterations=side_reports,
                  maximum_error_m_rad=errors.max(axis=0).tolist(),
                  rms_error_m_rad=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                  initial_error_m_rad=errors[0].tolist(), terminal_error_m_rad=errors[-1].tolist(),
                  error_definition='absolute FK wrist target error at every recorded physics timestamp; no integrated drift',
                  limitations=['Kinematic motion constraints; no torque, collision or hardware execution validation.'])
    return path, q, errors, report


def save_path(path, spline, report):
    path = Path(path)
    np.savez_compressed(path, knots=spline.t, coefficients=spline.c, degree=np.array(spline.k))
    path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


def load_path(path):
    with np.load(path, allow_pickle=False) as data:
        return BSpline(data['knots'], data['coefficients'], int(data['degree']), extrapolate=False)
