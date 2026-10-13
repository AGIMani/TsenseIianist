#!/usr/bin/env python3
"""Export the current MJLab arm/hand installation and joint-angle margins.

The export is built from the same saved preview/training configuration used by
``preview_mjlab.py --training-config``.  The figures show the actual compiled
Link7 -> flange -> four-hole connector -> hand geometry at the configured reset
pose (``initial_q_rad``).  The range panel uses the arm contract's independent
position limits; J6/J7's CCS coupling slack is reported separately.

This is a CPU-only geometry/FK export.  It does not step physics, run IK, or
send commands to a controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tsensei_v3_2_pianist.mjlab_env.arm_scene import FLANGE_IN_LINK7, SceneArmFollower
from tsensei_v3_2_pianist.mjlab_env.preview_contract import (
    build_bundle_from_config,
    load_training_config,
)


SIDES = ("left", "right")
SIDE_SUFFIX = {"left": "L", "right": "R"}
SIDE_TITLE = {"left": "Left", "right": "Right"}
ARM_COLOR = "#718096"
HAND_COLOR = "#4a83b8"
CONNECTOR_COLOR = "#d28b26"
LINK_COLOR = "#263b53"
JOINT_COLOR = "#c74b42"
FLANGE_COLOR = "#8759a8"
TEXT_COLOR = "#182e46"
MUTED_COLOR = "#53687b"
HALO = [path_effects.withStroke(linewidth=3, foreground="white")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mat_from_pose(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def body_pose(model, data, name: str) -> np.ndarray:
    body = model.body(name)
    return mat_from_pose(data.xpos[body.id], data.xmat[body.id].reshape(3, 3))


def display_meshes(model, data, side: str):
    """Return reduced triangle meshes for the actual compiled preview model."""
    result = []
    hand_prefix = f"{side}_/"
    arm_prefix = f"arm_follow_{side}_"
    connector_name = f"arm_connector_visual_{side}"
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 1:
            continue
        body_name = model.body(int(model.geom_bodyid[gid])).name or ""
        if not (body_name.startswith(arm_prefix) or body_name.startswith(hand_prefix)
                or body_name == connector_name):
            continue
        mesh_id = int(model.geom_dataid[gid])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count].astype(float)
        faces = model.mesh_face[face_start:face_start + face_count]
        # The displayed geometry is reduced only for plotting.  The compiled
        # model and all collision meshes remain untouched.
        step = 0.0014 if body_name.startswith(arm_prefix) else 0.0010
        _, first, inverse = np.unique(
            np.round(vertices / step).astype(np.int64), axis=0,
            return_index=True, return_inverse=True,
        )
        faces = inverse[faces]
        faces = faces[(faces[:, 0] != faces[:, 1])
                      & (faces[:, 1] != faces[:, 2])
                      & (faces[:, 0] != faces[:, 2])]
        if not len(faces):
            continue
        vertices = vertices[first]
        rotation = data.geom_xmat[gid].reshape(3, 3)
        vertices = vertices @ rotation.T + data.geom_xpos[gid]
        triangles = vertices[faces] * 1000.0
        if body_name.startswith(arm_prefix):
            color = ARM_COLOR
        elif body_name == connector_name:
            color = CONNECTOR_COLOR
        else:
            color = HAND_COLOR
        result.append((triangles, color, body_name))
    return result


def draw_meshes(ax, meshes, alpha=0.5):
    light = np.array([0.35, -0.40, 0.84])
    light /= np.linalg.norm(light)
    for triangles, color, _ in meshes:
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
        shade = 0.62 + 0.38 * np.abs(normals @ light)
        rgb = np.asarray(matplotlib.colors.to_rgb(color))
        facecolors = np.column_stack((shade[:, None] * rgb,
                                      np.full(len(triangles), alpha)))
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=facecolors, edgecolors="none",
            linewidths=0, rasterized=True, zorder=1,
        ))


def joint_and_mount_poses(model, data, side: str):
    suffix = SIDE_SUFFIX[side]
    names = [f"arm_follow_{side}_Base_{suffix}"]
    names += [f"arm_follow_{side}_Link{i}_{suffix}" for i in range(1, 8)]
    names += [
        f"arm_flange_{side}",
        f"arm_connector_{side}",
        f"{side}_/Right_hand_palm_link" if side == "right"
        else f"{side}_/Left_hand_palm_link",
    ]
    return {name: body_pose(model, data, name) for name in names}


def limits_rows(options: dict, side: str):
    q = np.asarray(options["initial_q_rad"][side], dtype=float)
    rows = []
    for index, (angle, limit) in enumerate(zip(q, options["joint_limits"][side]), 1):
        low, high = float(limit["min"]), float(limit["max"])
        if not (low - 1e-12 <= angle <= high + 1e-12):
            raise ValueError(f"{side} J{index} initial angle is outside its configured range")
        rows.append({
            "side": side,
            "joint": f"J{index}",
            "index": index,
            "current_rad": float(angle),
            "current_deg": math.degrees(float(angle)),
            "lower_rad": low,
            "lower_deg": math.degrees(low),
            "upper_rad": high,
            "upper_deg": math.degrees(high),
            # Every bundled arm URDF uses local +Z and the contract defines
            # +q as CCW when viewed from +axis toward the joint origin.
            "clockwise_remaining_rad": float(angle - low),
            "clockwise_remaining_deg": math.degrees(float(angle - low)),
            "counterclockwise_remaining_rad": float(high - angle),
            "counterclockwise_remaining_deg": math.degrees(float(high - angle)),
            "range_rad": high - low,
            "range_deg": math.degrees(high - low),
        })
    coupling = options["coupling"]
    value = float(coupling["coefficient"] * abs(q[int(coupling["j6_index"])])
                 + abs(q[int(coupling["j7_index"])]))
    slack = float(coupling["sum_limit_rad"] - value)
    return rows, {"value_rad": value, "slack_rad": slack,
                  "value_deg": math.degrees(value), "slack_deg": math.degrees(slack),
                  "coefficient": float(coupling["coefficient"]),
                  "sum_limit_rad": float(coupling["sum_limit_rad"]),
                  "sum_limit_deg": math.degrees(float(coupling["sum_limit_rad"])),
                  "j6_index": int(coupling["j6_index"]),
                  "j7_index": int(coupling["j7_index"])}


def set_3d_view(ax, meshes, poses, side: str):
    all_points = [triangles.reshape(-1, 3) for triangles, _, _ in meshes]
    all_points.append(np.stack([pose[:3, 3] for pose in poses.values()]) * 1000.0)
    points = np.concatenate(all_points)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(np.ptp(points, axis=0).max() / 2 + 45.0, 180.0)
    ax.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[1] - radius, center[1] + radius),
           zlim=(center[2] - radius, center[2] + radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.08)
    ax.set_proj_type("ortho")
    # The two views face the corresponding arm from the piano side while
    # retaining enough elevation to expose the flange and connector.
    ax.view_init(elev=18, azim=-62 if side == "right" else 62)
    ax.set_axis_off()


def marker(ax, point, label, color, size=34, label_offset=None):
    ax.scatter(*point, color=color, s=size, depthshade=False, zorder=25,
               edgecolors="white", linewidths=0.7)
    text_point = point if label_offset is None else point + np.asarray(label_offset)
    ax.text(*text_point, label, color=TEXT_COLOR, fontsize=10, fontweight="bold",
            path_effects=HALO, zorder=40)


def installation_page(side: str, model, data, options, follower, output: Path):
    poses = joint_and_mount_poses(model, data, side)
    meshes = display_meshes(model, data, side)
    rows, coupling = limits_rows(options, side)
    q = np.asarray(options["initial_q_rad"][side], dtype=float)

    fig = plt.figure(figsize=(20, 12), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 0.95),
                            left=0.025, right=0.985, top=0.91, bottom=0.06,
                            wspace=0.035)
    ax = fig.add_subplot(grid[0, 0], projection="3d", computed_zorder=False)
    ax.set_facecolor("#f8fafc")
    set_3d_view(ax, meshes, poses, side)
    draw_meshes(ax, meshes, alpha=0.58)

    chain_names = [
        f"arm_follow_{side}_Base_{SIDE_SUFFIX[side]}",
        *[f"arm_follow_{side}_Link{i}_{SIDE_SUFFIX[side]}" for i in range(1, 8)],
    ]
    chain_points = np.stack([poses[name][:3, 3] for name in chain_names]) * 1000.0
    ax.plot(*chain_points.T, color=LINK_COLOR, linewidth=2.2, zorder=20)
    for index, point in enumerate(chain_points[1:], 1):
        offset = {3: (0, 0, 14), 4: (0, 0, -14)}.get(index)
        marker(ax, point, f"J{index}", JOINT_COLOR, size=42, label_offset=offset)
    marker(ax, chain_points[0], "B", LINK_COLOR, size=52)

    mount_labels = (
        ("arm_flange_" + side, "F", FLANGE_COLOR),
        ("arm_connector_" + side, "C", CONNECTOR_COLOR),
        (next(name for name in poses if name.startswith(f"{side}_/")), "H", HAND_COLOR),
    )
    mount_points = []
    mount_offsets = {"F": (0, 0, 11), "C": (0, 0, -11), "H": (0, 0, 10)}
    for name, label, color in mount_labels:
        point = poses[name][:3, 3] * 1000.0
        mount_points.append(point)
        marker(ax, point, label, color, size=48, label_offset=mount_offsets[label])
    # Show the rigid attachment path separately from the moving arm chain.
    mount_points = np.asarray([chain_points[-1], *mount_points])
    ax.plot(*mount_points.T, color=CONNECTOR_COLOR, linestyle="--",
            linewidth=1.6, zorder=22)
    ax.text2D(0.03, 0.96, "B → J1 … J7 → F → C → H", transform=ax.transAxes,
              fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax.text2D(0.03, 0.92,
              "M6 visual chain + rigid flange/connector/hand; current reset q₀",
              transform=ax.transAxes, fontsize=10.5, color=MUTED_COLOR)
    ax.text2D(0.03, 0.035,
              "gray: arm meshes   blue: A3.2 hand   gold: four-hole connector   purple: flange",
              transform=ax.transAxes, fontsize=10, color=MUTED_COLOR)

    right = grid[0, 1].subgridspec(3, 1, height_ratios=(1.15, 2.3, 0.9), hspace=0.22)
    info = fig.add_subplot(right[0, 0])
    info.axis("off")
    base = follower.contract["base"][side]["position_m"]
    tool = np.asarray(options["link7_to_hand"][side]["xyz_m"]) * 1000.0
    info.text(0.0, 0.98, f"{SIDE_TITLE[side]} installation and joint margins",
              va="top", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    info.text(0.0, 0.76, f"Base world position: [{base[0]:+.6f}, {base[1]:+.6f}, {base[2]:+.6f}] m",
              fontsize=10.5, color=MUTED_COLOR)
    info.text(0.0, 0.60, f"F in Link7: [{FLANGE_IN_LINK7[0]*1000:+.1f}, {FLANGE_IN_LINK7[1]*1000:+.1f}, {FLANGE_IN_LINK7[2]*1000:+.1f}] mm",
              fontsize=10.5, color=MUTED_COLOR)
    info.text(0.0, 0.44, f"configured Link7→H xyz: [{tool[0]:+.1f}, {tool[1]:+.1f}, {tool[2]:+.1f}] mm",
              fontsize=10.5, color=MUTED_COLOR)
    info.text(0.0, 0.28,
              f"J6/J7 coupling: {coupling['value_deg']:.2f}° / {coupling['sum_limit_deg']:.2f}°; slack {coupling['slack_deg']:.2f}°",
              fontsize=10.5, color=MUTED_COLOR)
    info.text(0.0, 0.08,
              "+q is CCW when viewed from local +axis toward the joint origin; CW goes toward the lower limit.",
              fontsize=9.5, color=MUTED_COLOR)

    ranges = fig.add_subplot(right[1, 0])
    ranges.set_facecolor("#fbfcfe")
    ranges.set_xlim(-190, 190)
    ranges.set_ylim(-0.8, 6.8)
    ranges.set_yticks(range(6, -1, -1), [f"J{i}" for i in range(1, 8)])
    ranges.set_xlabel("joint angle (degrees; independent position limits)", color=MUTED_COLOR)
    ranges.set_xticks([-180, -90, 0, 90, 180])
    ranges.grid(axis="x", alpha=0.18)
    ranges.spines[["top", "right", "left"]].set_visible(False)
    ranges.spines["bottom"].set_color("#b5c0cb")
    ranges.tick_params(axis="y", length=0, colors=TEXT_COLOR)
    ranges.tick_params(axis="x", colors=MUTED_COLOR)
    for y, row in zip(range(6, -1, -1), rows):
        lo, hi, angle = row["lower_deg"], row["upper_deg"], row["current_deg"]
        ranges.plot([lo, hi], [y, y], color="#d5dce5", linewidth=10,
                    solid_capstyle="round", zorder=1)
        ranges.plot([lo, angle], [y, y], color="#d2873d", linewidth=8,
                    solid_capstyle="round", zorder=2)
        ranges.plot([angle, hi], [y, y], color="#3d9b77", linewidth=8,
                    solid_capstyle="round", zorder=2)
        ranges.scatter([angle], [y], color=TEXT_COLOR, s=32, zorder=3,
                       edgecolors="white", linewidths=0.8)
        ranges.text(-187, y + 0.18, f"q₀ {angle:+.1f}°", fontsize=8.4,
                    color=TEXT_COLOR, ha="left", va="center", clip_on=False)
        ranges.text(187, y - 0.20,
                    f"CW {row['clockwise_remaining_deg']:.1f}°  |  CCW {row['counterclockwise_remaining_deg']:.1f}°",
                    fontsize=8.4, color=MUTED_COLOR, ha="right", va="center", clip_on=False)
    ranges.set_title("Current angle inside each joint's hard position interval",
                     loc="left", fontsize=12, color=TEXT_COLOR, pad=10)

    note = fig.add_subplot(right[2, 0])
    note.axis("off")
    note.text(0.0, 0.96, "How to read the margins", fontsize=11.5,
              fontweight="bold", color=TEXT_COLOR, va="top")
    note.text(0.0, 0.68,
              "orange = clockwise room (q₀ − lower); green = counterclockwise room (upper − q₀).",
              fontsize=9.8, color=MUTED_COLOR)
    note.text(0.0, 0.43,
              "The CSV/JSON keep the exact radian values. J6/J7 coupling, braking, velocity and acceleration limits can reduce the feasible motion further.",
              fontsize=9.8, color=MUTED_COLOR, wrap=True)
    note.text(0.0, 0.13,
              "This export uses the configured training reset q₀; it is not a live viewer slider snapshot.",
              fontsize=9.8, color=MUTED_COLOR)

    fig.suptitle(f"Current MJLab training installation — {SIDE_TITLE[side]} arm and hand",
                 fontsize=21, fontweight="bold", color=TEXT_COLOR, x=0.03, ha="left", y=0.975)
    png = output / f"installation_{side}.png"
    pdf = output / f"installation_{side}.pdf"
    fig.savefig(png, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(pdf, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return rows, coupling, poses


def overview_page(model, data, output: Path):
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#f8fafc")
    all_meshes, all_poses = [], {}
    for side in SIDES:
        meshes = display_meshes(model, data, side)
        all_meshes.extend(meshes)
        poses = joint_and_mount_poses(model, data, side)
        all_poses.update({f"{side}:{key}": value for key, value in poses.items()})
    set_3d_view(ax, all_meshes, all_poses, "right")
    draw_meshes(ax, all_meshes, alpha=0.48)
    for side in SIDES:
        poses = joint_and_mount_poses(model, data, side)
        suffix = SIDE_SUFFIX[side]
        chain_names = [f"arm_follow_{side}_Base_{suffix}"] + [
            f"arm_follow_{side}_Link{i}_{suffix}" for i in range(1, 8)
        ]
        points = np.stack([poses[name][:3, 3] for name in chain_names]) * 1000.0
        ax.plot(*points.T, color=LINK_COLOR, linewidth=2.0, zorder=20)
        ax.scatter(*points[0], color=LINK_COLOR, s=50, zorder=25)
        hand_name = next(name for name in poses if name.startswith(f"{side}_/"))
        for label, name, color in (("F", f"arm_flange_{side}", FLANGE_COLOR),
                                   ("C", f"arm_connector_{side}", CONNECTOR_COLOR),
                                   ("H", hand_name, HAND_COLOR)):
            point = poses[name][:3, 3] * 1000.0
            ax.scatter(*point, color=color, s=36, zorder=25,
                       edgecolors="white", linewidths=0.6)
            ax.text(*point, f"{SIDE_TITLE[side][0]}-{label}", fontsize=9,
                    color=TEXT_COLOR, path_effects=HALO)
    ax.text2D(0.03, 0.95, "Both current rigid installations in the piano/world frame",
              transform=ax.transAxes, fontsize=16, fontweight="bold", color=TEXT_COLOR)
    ax.text2D(0.03, 0.91,
              "B → J1…J7 → fixed flange → fixed four-hole connector → hand root; q₀ reset pose",
              transform=ax.transAxes, fontsize=10.5, color=MUTED_COLOR)
    ax.text2D(0.03, 0.035,
              "gray: M6 visual links   blue: hands   gold: connectors   purple: flange",
              transform=ax.transAxes, fontsize=10, color=MUTED_COLOR)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    fig.savefig(output / "installation_overview.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(output / "installation_overview.pdf", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(path: Path, rows):
    fields = [
        "side", "joint", "index", "current_rad", "current_deg",
        "lower_rad", "lower_deg", "upper_rad", "upper_deg",
        "clockwise_remaining_rad", "clockwise_remaining_deg",
        "counterclockwise_remaining_rad", "counterclockwise_remaining_deg",
        "range_rad", "range_deg",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, report):
    lines = [
        "# Current MJLab arm/hand installation and angle margins",
        "",
        "The figures are rebuilt from `configs/preview/current_arm_training_20261011.json` "
        "through the same `build_bundle_from_config` path used by the synchronized MJLab preview.",
        "The pose is the saved training reset `arm_follow.initial_q_rad` (q₀), not a live viewer slider.",
        "",
        "## Installation",
        "",
        "Each side is rendered as the compiled chain `Base → Link1 … Link7 → fixed flange → "
        "fixed four-hole connector → fixed hand root`.  The M6 link bodies are kinematic visual "
        "meshes (`density=0`, `contype=conaffinity=0`); hand fingers and piano keys remain the "
        "dynamic physical bodies in the training model.",
        "",
        "The flange convention is local Link7 `[0, -95, 0] mm`.  The configured Link7-to-hand "
        "transform and exact matrices are in `installation_report.json`.",
        "",
        "## Angle margins",
        "",
        "+q follows the URDF local +Z right-hand convention.  Viewed from the +axis toward the "
        "joint origin, increasing q is counterclockwise.  Therefore `CW = q₀ − lower` and "
        "`CCW = upper − q₀`.  The per-joint values are in degrees on the figures and exact "
        "radians in `joint_margins.csv` and `installation_report.json`.",
        "",
        "The independent position intervals do not replace the CCS-696 J6/J7 coupling region "
        "or the runtime velocity, acceleration and braking limits; the coupling value/slack is "
        "reported for each side.",
        "",
        "## Files",
        "",
        "- `installation_left.png` / `.pdf`: left arm, connector and hand with margin panel.",
        "- `installation_right.png` / `.pdf`: right arm, connector and hand with margin panel.",
        "- `installation_overview.png` / `.pdf`: both installed assemblies.",
        "- `joint_margins.csv`: exact per-joint q₀, limits and clockwise/counterclockwise margins.",
        "- `installation_report.json`: transforms, coupling slack, source hashes and validation.",
        "- `arm_installation_limits_bundle.zip`: reproducible export bundle.",
        "",
        "This is a CPU-only FK/geometry export.  It does not step physics, run a policy, or control hardware.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config", type=Path,
        default=ROOT / "configs/preview/current_arm_training_20261011.json",
        help="saved full training/preview config",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "preview/outputs/arm_installation_limits_20261011",
        help="new output directory; refuses to overwrite a non-empty directory",
    )
    args = parser.parse_args()
    config_path = args.training_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite a non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = load_training_config(config_path)
    bundle, scene_directory, _ = build_bundle_from_config(config)
    data = bundle.reset_cpu()
    follower = SceneArmFollower(bundle, num_envs=1, device="cpu")
    options = follower.options
    model = bundle.model

    all_rows = []
    side_reports = {}
    for side in SIDES:
        rows, coupling, poses = installation_page(side, model, data, options, follower, output)
        all_rows.extend(rows)
        side_reports[side] = {
            "joint_margins": rows,
            "coupling": coupling,
            "base_contract": follower.contract["base"][side],
            "base_in_piano": follower.contract["base_in_piano"][side],
            "link7_to_hand": options["link7_to_hand"][side],
            "flange_in_link7_m": FLANGE_IN_LINK7.tolist(),
            "poses_world_m": {name: pose.tolist() for name, pose in poses.items()},
        }
    overview_page(model, data, output)
    write_csv(output / "joint_margins.csv", all_rows)

    source_paths = [
        config_path,
        ROOT / "preview/scripts/export_arm_installation_limits.py",
        ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_scene.py",
        ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_follow.py",
        ROOT / "tsensei_v3_2_pianist/mjlab_env/preview_contract.py",
        ROOT / "configs/preview/m6_sdk_joint_limits.json",
        ROOT / "configs/preview/hand_connector_20260930.json",
        ROOT / "preview/assets/tj_m6/left/arm.urdf",
        ROOT / "preview/assets/tj_m6/right/arm.urdf",
    ]
    report = {
        "schema": "arm_installation_limits.v1",
        "units": {"length": "m", "display_length": "mm", "angle": "rad", "display_angle": "deg"},
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "temporary_scene": str(scene_directory),
        "pose_definition": "saved arm_follow.initial_q_rad reset pose; no physics stepping or live viewer state",
        "positive_rotation_definition": "URDF local +Z right-hand rule; viewed from +axis toward origin, +q is counterclockwise",
        "clockwise_margin_formula": "q0 - lower",
        "counterclockwise_margin_formula": "upper - q0",
        "installation_contract": follower.contract,
        "sides": side_reports,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in source_paths
        },
        "validation": {
            "sides": list(SIDES),
            "joints_checked": len(all_rows),
            "all_initial_angles_inside_independent_limits": True,
            "mocap_visual_arm_only": True,
            "physics_steps": 0,
            "hardware_commands": 0,
        },
    }
    (output / "installation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_readme(output / "README.md", report)

    deliverables = sorted(p for p in output.iterdir() if p.is_file())
    manifest = {
        "schema": "arm_installation_limits_manifest.v1",
        "files": {p.name: sha256(p) for p in deliverables},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output / "arm_installation_limits_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in output.iterdir() if p.is_file() and p.name != "arm_installation_limits_bundle.zip"):
            archive.write(path, path.name)
    print(json.dumps({
        "output": str(output),
        "config_sha256": report["config_sha256"],
        "files": sorted(p.name for p in output.iterdir() if p.is_file()),
        "left": {"joints": len(side_reports["left"]["joint_margins"]),
                 "coupling_slack_deg": side_reports["left"]["coupling"]["slack_deg"]},
        "right": {"joints": len(side_reports["right"]["joint_margins"]),
                  "coupling_slack_deg": side_reports["right"]["coupling"]["slack_deg"]},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
