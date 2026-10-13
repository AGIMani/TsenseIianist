"""Static workspace scan using the same current rigid_wrist_command arm follower as training.

This is a kinematic diagnostic.  ``solved`` rows are numerical IK witnesses;
``proven_outside_bound`` rows are outside a conservative Link7 outer bound and
therefore cannot be reached for the requested pose at the fixed installation.
Rows marked ``unresolved`` remain inconclusive: they may be singular, rate
limited, or simply need a different IK seed.  No MuJoCo dynamics or collision
claim is made by this scanner.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .arm_follow import ArmFollower, rotation_error


def _limits(bundle, wrist):
    indices = np.asarray(wrist["indices"], dtype=int)
    if indices.shape != (3,):
        raise ValueError(f"Expected TX/TY/Pitch wrist request indices, got {indices.tolist()}")
    low = np.asarray(bundle.low)[indices]
    high = np.asarray(bundle.high)[indices]
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(low < high)):
        raise ValueError("Invalid wrist request bounds in the compiled scene")
    return low, high


def _outer_bounds(arm):
    """Return the loose and paired-link outer bounds used by the audit.

    The J3/J4 offsets rotate about one common axis.  Pairing those two offsets
    gives a tighter *still outer* bound than summing their lengths, while the
    remaining links are bounded by their norms.  The pose tolerance margin is
    the radial equivalent of the witness thresholds (0.1 mm, 0.001 rad).
    """

    joint_pos = np.asarray(arm["joint_pos"])
    joint_rot = np.asarray(arm["joint_rot"])
    axes = np.asarray(arm["axis"])
    simple = float(np.linalg.norm(joint_pos[1:], axis=1).sum())
    first = joint_rot[2].T @ joint_pos[2]
    second = joint_pos[3]
    axis = axes[2] / np.linalg.norm(axes[2])
    first_parallel = float(np.dot(first, axis))
    second_parallel = float(np.dot(second, axis))
    dot_max = (first_parallel * second_parallel
               + np.linalg.norm(first - first_parallel * axis)
               * np.linalg.norm(second - second_parallel * axis))
    paired = float(np.sqrt(np.dot(joint_pos[2], joint_pos[2])
                           + np.dot(joint_pos[3], joint_pos[3]) + 2 * dot_max))
    tight = float(np.linalg.norm(joint_pos[1]) + paired
                  + np.linalg.norm(joint_pos[4:], axis=1).sum())
    tolerance_margin = float(.0001 + 2 * np.linalg.norm(np.asarray(arm["tool_pos"]))
                              * np.sin(.001 / 2))
    return simple, tight, tolerance_margin


def _write_csv(path, rows):
    fields = [
        "tx_m", "ty_m", "pitch_rad", "lift_m", "status", "position_error_m",
        "orientation_error_rad", "attempts", "link7_radius_m", "reach_upper_m",
    ] + [f"q{i+1}_rad" for i in range(7)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, side_rows, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional export dependency
        return {"written": False, "reason": f"matplotlib unavailable: {exc}"}
    colors = {
        "solved": "#2ca02c",
        "proven_outside_bound": "#d62728",
        "unresolved": "#ffbf00",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for status, color in colors.items():
        points = [(float(row["tx_m"]), float(row["ty_m"]))
                  for row in side_rows if row["status"] == status]
        if points:
            xy = np.asarray(points)
            ax.scatter(xy[:, 0], xy[:, 1], s=13, c=color, label=status, alpha=.8)
    ax.set_xlabel("requested wrist TX (m)")
    ax.set_ylabel("requested wrist TY (m)")
    ax.set_title(title)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"written": True, "path": str(path)}


def scan_reachability(
    bundle,
    output: str | Path,
    *,
    tx_points: int = 41,
    ty_points: int = 9,
    pitch_points: int = 5,
    preparation_points: int | None = None,
    extra_attempts: int = 12,
    batch_size: int = 256,
    seed: int = 7241,
):
    """Scan wrist request coordinates from a training-augmented bundle.

    The default grid is the same 41 x 9 x 5 x 3 grid used by the existing
    workspace audit.  Smaller counts are useful for a quick smoke check.
    """

    for name, value in (("tx_points", tx_points), ("ty_points", ty_points),
                        ("pitch_points", pitch_points), ("extra_attempts", extra_attempts),
                        ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = getattr(bundle, "arm_follow_spec", None)
    if not spec:
        raise ValueError("Reachability scan requires a training scene with arm_follow")
    options = spec["options"]
    initial = {side: (np.asarray(value[0]), np.asarray(value[1]))
               for side, value in spec["initial_poses"].items()}
    dt = float(bundle.manifest["physics_dt"])
    rng = np.random.default_rng(seed)
    prep = bundle.preparation_options or {}
    lift = float(prep.get("lift", 0.0))
    if preparation_points is None:
        preparation_points = 3 if lift > 0 else 1
    if isinstance(preparation_points, bool) or not isinstance(preparation_points, int) or preparation_points < 1:
        raise ValueError("preparation_points must be a positive integer")
    z_values = np.linspace(0., lift, preparation_points) if preparation_points > 1 else np.asarray([0.])
    summary = {
        "schema": 1,
        "kind": "mjlab_arm_reachability_scan_v1",
        "source": "SceneBundle + ArmFollower.当前刚性臂控制",
        "physics_dt_s": dt,
        "grid": {
            "tx_points": tx_points, "ty_points": ty_points,
            "pitch_points": pitch_points, "preparation_points": preparation_points,
            "extra_attempts": extra_attempts,
        },
        "interpretation": [
            "solved is a numerical FK witness within 0.1 mm and 0.001 rad",
            "proven_outside_bound is outside a conservative Link7 outer bound",
            "unresolved is inconclusive and needs a path/rate/collision check",
        ],
        "sides": {},
        "files": {},
    }

    for side in ("left", "right"):
        wrist = spec["wrists"][side]
        low, high = _limits(bundle, wrist)
        xs = np.linspace(low[0], high[0], tx_points)
        ys = np.linspace(low[1], high[1], ty_points)
        pitches = np.linspace(low[2], high[2], pitch_points)
        mesh = np.stack(np.meshgrid(xs, ys, pitches, z_values, indexing="ij"), -1).reshape(-1, 4)
        count = len(mesh)
        solver = ArmFollower(options, initial, num_envs=count, dt=dt, device="cpu")
        arm = solver.sides[side]
        p0 = np.asarray(wrist["rest_pos"])
        r0 = np.asarray(wrist["rest_rot"])
        axes = np.asarray(wrist["axis"])
        world_axes = r0 @ axes.T
        target_p = p0 + mesh[:, :2] @ world_axes[:, :2].T + mesh[:, 3, None] * np.asarray([0., 0., 1.])
        target_r = r0 @ Rotation.from_rotvec(mesh[:, 2, None] * axes[2]).as_matrix()
        target_p_t = torch.as_tensor(target_p, dtype=torch.float64)
        target_r_t = torch.as_tensor(target_r, dtype=torch.float64)

        link7_rot = target_r @ np.asarray(arm["tool_rot"]).T
        link7_pos = target_p - np.einsum("nij,j->ni", link7_rot, np.asarray(arm["tool_pos"]))
        shoulder = np.asarray(arm["base_pos"]) + np.asarray(arm["base_rot"]) @ np.asarray(arm["joint_pos"][0])
        simple_upper, reach_upper, bound_margin = _outer_bounds(arm)
        radius = np.linalg.norm(link7_pos - shoulder, axis=1)
        # Use the loose bound to decide which unresolved targets deserve extra
        # IK seeds, then apply the tighter paired-link proof to the final
        # classification.  This preserves the audit's numerical witnesses
        # while gaining the stronger geometric proof afterward.
        outside_for_solver = radius > simple_upper + 1e-9
        outside_proven = radius > reach_upper + bound_margin + 1e-9
        best_q = np.asarray(arm["q0"])[None, :].repeat(count, axis=0)
        best_position = np.full(count, np.inf)
        best_orientation = np.full(count, np.inf)
        attempts = np.zeros(count, dtype=int)
        solved = np.zeros(count, dtype=bool)
        original_options = dict(solver.options)

        def solve_batch(ids, seeds, iterations, damping):
            if len(ids) == 0:
                return
            solver.num_envs = len(ids)
            solver.options = dict(original_options, iterations=iterations, damping=damping)
            low_t = arm["low"].expand(len(ids), -1)
            high_t = arm["high"].expand(len(ids), -1)
            q = solver.solve(arm, seeds, target_p_t[ids], target_r_t[ids], low_t, high_t)
            p, r, *_ = solver.fk(arm, q)
            position = (p - target_p_t[ids]).norm(dim=-1).numpy()
            orientation = rotation_error(r, target_r_t[ids]).norm(dim=-1).numpy()
            q_np = q.numpy()
            for row, index in enumerate(ids):
                index = int(index)
                attempts[index] += 1
                score = max(position[row] / 1e-4, orientation[row] / 1e-3)
                old = max(best_position[index] / 1e-4, best_orientation[index] / 1e-3)
                if score < old:
                    best_q[index] = q_np[row]
                    best_position[index] = position[row]
                    best_orientation[index] = orientation[row]
                if position[row] <= 1e-4 and orientation[row] <= 1e-3:
                    solved[index] = True

        started = time.monotonic()
        for start in range(0, count, batch_size):
            ids = np.arange(start, min(start + batch_size, count))
            solve_batch(ids, arm["q0"].expand(len(ids), -1).clone(), 12, .008)
        reset_success = int(np.count_nonzero((best_position <= 1e-4) & (best_orientation <= 1e-3)))
        for attempt in range(extra_attempts):
            pending = np.flatnonzero(~solved & ~outside_for_solver)
            if not len(pending):
                break
            for start in range(0, len(pending), batch_size):
                ids = pending[start:start + batch_size]
                if attempt == 0:
                    seeds = torch.as_tensor(best_q[ids], dtype=torch.float64)
                elif attempt == 1:
                    seeds = (arm["low"] + arm["high"]).expand(len(ids), -1) / 2
                else:
                    random = torch.as_tensor(rng.random((len(ids), 7)), dtype=torch.float64)
                    seeds = arm["low"] + random * (arm["high"] - arm["low"])
                solve_batch(ids, seeds, 96, .002 if attempt < 4 else .0005)

        rows = []
        for index, values in enumerate(mesh):
            status = "solved" if solved[index] else ("proven_outside_bound" if outside_proven[index] else "unresolved")
            row = dict(
                tx_m=float(values[0]), ty_m=float(values[1]), pitch_rad=float(values[2]), lift_m=float(values[3]),
                status=status,
                position_error_m=None if not np.isfinite(best_position[index]) else float(best_position[index]),
                orientation_error_rad=None if not np.isfinite(best_orientation[index]) else float(best_orientation[index]),
                attempts=int(attempts[index]), link7_radius_m=float(radius[index]), reach_upper_m=reach_upper,
            )
            row.update({f"q{i+1}_rad": float(best_q[index, i]) for i in range(7)})
            rows.append(row)
        csv_path = output / f"{side}.csv"
        _write_csv(csv_path, rows)
        plot_info = _plot(output / f"{side}.png", rows, f"{side} arm request reachability")
        summary["files"][side] = {"csv": str(csv_path), "plot": plot_info}
        summary["sides"][side] = {
            "total": count,
            "solved": int(np.count_nonzero(solved)),
            "proven_outside_bound": int(np.count_nonzero(outside_proven)),
            "unresolved": int(np.count_nonzero(~solved & ~outside_proven)),
            "reset_seed_current12_success": reset_success,
            "reach_upper_m": reach_upper,
            "simple_outer_bound_m": simple_upper,
            "bound_tolerance_margin_m": bound_margin,
            "shoulder_world_m": shoulder.tolist(),
            "bounds": {"low": low.tolist(), "high": high.tolist()},
            "elapsed_seconds": time.monotonic() - started,
        }
    summary_path = output / "summary.json"
    summary["files"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
