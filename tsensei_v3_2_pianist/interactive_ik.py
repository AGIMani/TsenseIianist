"""Small, dependency-light helpers for interactive damped IK.

The MuJoCo preview owns the kinematic model and joint-limit policy.  This
module only contains the numerical DLS step so the exact damping convention is
shared by the mouse-handle path and the wrist-follow path and can be tested
without opening a viewer.
"""

from __future__ import annotations

import numpy as np


def damped_least_squares_step(jacobian, error, damping):
    """Return ``J.T @ inv(J J.T + damping**2 I) @ error``.

    ``error`` is a task-space residual and ``jacobian`` maps joint increments
    to that residual.  The caller applies joint limits, velocity limits and
    backtracking around this step; this function deliberately does not clip or
    silently repair invalid inputs.
    """

    j = np.asarray(jacobian, dtype=float)
    e = np.asarray(error, dtype=float)
    if j.ndim != 2 or e.ndim != 1 or j.shape[0] != e.shape[0]:
        raise ValueError(f"Expected J=[task,joints], error=[task], got {j.shape}, {e.shape}")
    if not np.isfinite(j).all() or not np.isfinite(e).all():
        raise ValueError("DLS inputs must be finite")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("DLS damping must be finite and positive")
    regularized = j @ j.T + float(damping) ** 2 * np.eye(j.shape[0])
    return j.T @ np.linalg.solve(regularized, e)

