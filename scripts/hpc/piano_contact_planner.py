"""Small CPU-only mathematical primitives for offline contact planning.

This module neither loads a model nor changes any physical state. Model
callbacks belong to the remote runner and operate on scratch FK data only.
"""
from types import SimpleNamespace

import numpy as np


def idle_pinky_coordinates(action_names, active_indices, selected):
    """Return controlled coordinates by canonical names; followers stay free."""
    names=list(action_names);active=list(active_indices);result={}
    for prefix,finger in (('R',4),('L',9)):
        if finger in selected:continue
        for joint,value in (('MCP',.490541428),('SW',.01000039)):
            result[active.index(names.index(f'{prefix}_Pinky_{joint}'))]=value
    return result


def terminal_top_error(vertices, center, rotation, half_sizes, *, edge_margin=.005,
                       vertical_offset=0.):
    """Lowest real terminal vertex against the finite, possibly tilted key top.

    This is an IK objective, never evidence of physical contact. Requiring the
    lowest vertex inside the finite key avoids a side contact or a fingertip
    hanging beyond the black-key front edge. Native contacts validate execution.
    """
    local=(np.asarray(vertices)-np.asarray(center))@np.asarray(rotation)
    half=np.asarray(half_sizes)
    point=local[int(np.argmin(local[:,2]))]
    extent=max(float(half[0])-edge_margin,0.)
    return np.array([point[0]-np.clip(point[0],-extent,extent),point[1],
                     point[2]-half[2]-vertical_offset]),point


def fingertip_residual(actual, wanted, weights, *, active, x_bounds=None,
                       idle_clearance=True):
    """Weighted task error; an idle height is a floor, never an attractor."""
    error = np.asarray(actual, dtype=float) - np.asarray(wanted, dtype=float)
    if idle_clearance:
        for finger in range(len(error)):
            if finger not in active:
                error[finger, 2] = min(error[finger, 2], 0.)
    for finger, (low, high) in (x_bounds or {}).items():
        if high < low:
            raise ValueError('Empty longitudinal key surface interval')
        error[finger, 0] = actual[finger, 0] - np.clip(actual[finger, 0], low, high)
    return error * np.asarray(weights)


def solve_bounded_pose(initial, low, high, independent, residual, *, max_nfev=35,
                       difference_steps=None):
    """Solve independent coordinates while preserving exact frozen coordinates.

Use explicit absolute finite differences: float32 coupling callbacks erase
SciPy's default ~1e-8 perturbations. Every callback remains inside the limits.
The final callback restores the scratch model to the returned pose.
"""
    from scipy.optimize import least_squares
    low, high = np.asarray(low, float), np.asarray(high, float)
    pose = np.clip(np.asarray(initial, float), low, high)
    independent = np.asarray(independent, dtype=int)
    if (pose.ndim != 1 or low.shape != pose.shape or high.shape != pose.shape
            or not np.isfinite(np.r_[pose, low, high]).all() or np.any(low > high)):
        raise ValueError('Expected finite ordered pose limits')
    indices = independent[(high - low)[independent] > 1e-7]
    if not len(indices):
        errors = np.asarray(residual(pose), float)
        return SimpleNamespace(pose=pose, success=True, nfev=1, status=0,
                               max_abs_residual=float(np.max(np.abs(errors), initial=0)))
    steps = (np.full(len(pose), 1e-4) if difference_steps is None
             else np.broadcast_to(np.asarray(difference_steps, float), pose.shape))
    if not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError('Finite-difference steps must be positive')

    def evaluate(x):
        q = pose.copy()
        q[indices] = np.clip(x, low[indices], high[indices])
        return np.asarray(residual(q), dtype=float)

    def jacobian(x):
        baseline = evaluate(x)
        columns = []
        for j, index in enumerate(indices):
            moved = x.copy()
            delta = min(steps[index], high[index] - x[j])
            if delta < min(steps[index] / 2, (high[index] - low[index]) / 2):
                delta = max(-steps[index], low[index] - x[j])
            moved[j] += delta
            columns.append((evaluate(moved) - baseline) / delta)
        return np.stack(columns, axis=1)

    result = least_squares(evaluate, pose[indices], jac=jacobian,
                           bounds=(low[indices], high[indices]), max_nfev=max_nfev,
                           ftol=1e-7, xtol=1e-7, gtol=1e-8, x_scale='jac')
    pose[indices] = result.x
    errors = evaluate(result.x)
    return SimpleNamespace(pose=pose, success=bool(result.success), nfev=result.nfev,
                           status=result.status,
                           max_abs_residual=float(np.max(np.abs(errors), initial=0)))
