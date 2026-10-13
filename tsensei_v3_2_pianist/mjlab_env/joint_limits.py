"""Explicit robot position bounds shared by simulation and trajectory export.

Historical physical recordings remain measurements: bounded exports must be
labelled as derived commands, never substituted for the physical trace.
"""
import numpy as np


def inward_bounds(lower, upper, dtype=np.float32):
    """Represent bounds inside the exact double interval, even after rounding."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != upper.shape or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower >= upper)):
        raise ValueError('Joint ranges must be finite and increasing')
    lo, hi = lower.astype(dtype), upper.astype(dtype)
    lo = np.where(lo.astype(float) < lower, np.nextafter(lo, np.asarray(np.inf, dtype=dtype)), lo)
    hi = np.where(hi.astype(float) > upper, np.nextafter(hi, np.asarray(-np.inf, dtype=dtype)), hi)
    if np.any(lo >= hi):
        raise ValueError('Joint range cannot be represented in simulation precision')
    return lo, hi


def resolve_robot_limits(model, *, key_ids=(), action_ids=(), active_indices=(),
                         physical_low=(), physical_high=(), control_contract=None):
    """Intersect compiled URDF/model, action ranges and explicit position limits.

Piano keys are excluded: this option governs robot motion, not piano dynamics.
Removed wrist DOFs have no joint address and are not recreated or projected.
"""
    import mujoco
    actions = {}
    controls = {} if control_contract is None else {
        row['index']: row for row in control_contract['channels'].values()}
    for aid, index in zip(action_ids, active_indices):
        jid = int(model.actuator_trnid[aid, 0])
        row = controls.get(int(index), {})
        bounds = (max(float(physical_low[index]), row.get('min', -np.inf)),
                  min(float(physical_high[index]), row.get('max', np.inf)))
        actions.setdefault(jid, []).append(bounds)
    rows = []
    for jid in range(model.njnt):
        if jid in key_ids or not model.jnt_limited[jid]:
            continue
        if int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError('Hard joint limits require scalar robot joints')
        lower, upper = map(float, model.jnt_range[jid])
        for lo, hi in actions.get(jid, ()):
            lower, upper = max(lower, lo), min(upper, hi)
        lo, hi = inward_bounds(lower, upper)
        rows.append(dict(name=model.joint(jid).name, joint_id=jid,
            qpos_address=int(model.jnt_qposadr[jid]), dof_address=int(model.jnt_dofadr[jid]),
            min=lower, max=upper, float32_min=float(lo), float32_max=float(hi),
            unit='m' if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else 'rad'))
    return dict(schema=1, mode='hard_v1', joints=rows,
        scope='limited scalar robot joints; piano keys excluded',
        bounds='intersection of compiled model, physical action range and explicit position limits',
        application='after each physics substep and reset; before contacts, observations and recording',
        velocity='remove outward velocity at boundary; clear affected solver warmstart',
        dynamics='inelastic state projection, changes dynamics; no force/energy equivalence to a physical hard stop')


def bounded_positions(positions, lower, upper):
    """Return derived bounded positions and an auditable correction mask."""
    q = np.asarray(positions, dtype=float)
    lo, hi = inward_bounds(lower, upper, np.float64)
    if q.ndim != 2 or q.shape[1:] != lo.shape or not np.isfinite(q).all():
        raise ValueError('Positions must be finite rows matching every joint bound')
    bounded = np.clip(q, lo, hi)
    return bounded, bounded != q


def require_bounded(positions, lower, upper):
    """Fail export rather than disguise invalid measured states as valid motion."""
    _, changed = bounded_positions(positions, lower, upper)
    if changed.any():
        row, column = np.argwhere(changed)[0]
        raise ValueError(f'Joint position outside limits at sample {row}, column {column}; '
                         'use a separately labelled bounded command export for historical recordings')
