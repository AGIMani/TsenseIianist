#!/usr/bin/env python3
# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""Bounded analysis of the saved cross-hand CPU/GPU comparison.

Contact lists are CPU geometric queries at each saved trajectory's qpos.
They are not GPU solver forces or original GPU contact lists.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms


def intervals(mask, dt):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {"sampled_duration_s": 0., "intervals": []}
    chunks = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return {"sampled_duration_s": float(len(ids) * dt), "intervals": [
        {"first_sample_s": float((a[0] + 1) * dt), "last_sample_s": float((a[-1] + 1) * dt),
         "samples": len(a), "sampled_duration_s": float(len(a) * dt)} for a in chunks]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.comparison.read_text())
    trace = np.load(args.comparison.with_suffix(".npz"))
    recipe = make_recipe(initial_buffer_time=0, task_profile="tsensei_collision_v1", drive_profile="shadow_v1", collision_profile="convex_hull_v1")
    if recipe["source_sha256"] != reference["source_sha256"]:
        raise RuntimeError("Source changed after saved comparison")
    scene_path, task_path, _ = build(recipe)
    scene_path, _ = without_arms(scene_path, task_path, scene_path.parent / "hand_only")
    bundle = SceneBundle(scene_path, disable_wrist_tz=True, disable_wrist_yaw=True)
    model, dt = bundle.model, reference["physics_dt"]
    ids = model.actuator_trnid[bundle.action_ids, 0]
    names = {int(j): model.joint(int(j)).name for j in ids}
    groups = {
        "hand_slides_m_s": ids[model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_SLIDE],
        "finger_hinges_rad_s": np.asarray([j for j in ids if "palm_base_" not in names[int(j)]]),
        "wrist_pitch_rad_s": np.asarray([j for j in ids if names[int(j)].endswith("palm_base_pitch")]),
    }
    report = {"comparison": str(args.comparison.resolve()), "peak_absolute_velocities": {},
              "difference_duration": {}, "cross_hand_geometry_queries": [],
              "note": "Contact names and distances below are CPU geometric queries on saved CPU/GPU poses. No GPU original contact forces are inferred. Durations count sampled 5-ms intervals; no continuous-time claim."}
    for group, joints in groups.items():
        qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        report["peak_absolute_velocities"][group] = {}
        for backend in ("cpu", "gpu"):
            values = abs(trace[f"cross_hand_{backend}_qvel"][:, va])
            idx = np.unravel_index(np.argmax(values), values.shape)
            report["peak_absolute_velocities"][group][backend] = {
                "value": float(values[idx]), "joint": names[int(joints[idx[1]])], "time_s": float((idx[0] + 1) * dt)}
        qdelta = abs(trace["cross_hand_cpu_qpos"][:, qa] - trace["cross_hand_gpu_qpos"][:, qa]).max(1)
        vdelta = abs(trace["cross_hand_cpu_qvel"][:, va] - trace["cross_hand_gpu_qvel"][:, va]).max(1)
        thresholds = (.00001, .0001) if group.startswith("hand_slides") else (.001, .01)
        for threshold in thresholds:
            report["difference_duration"][group + "_qpos_gt_" + str(threshold)] = intervals(qdelta > threshold, dt)
        vthreshold = .01 if group.startswith("hand_slides") else 1.
        report["difference_duration"][group + "_qvel_gt_" + str(vthreshold)] = intervals(vdelta > vthreshold, dt)
        report["difference_duration"][group + "_final_qpos_error"] = float(qdelta[-1])
        report["difference_duration"][group + "_final_qvel_error"] = float(vdelta[-1])
    for time in (.3, .325, .33, .335, .34, .445, .5):
        tick = round(time / dt) - 1
        row = {"time_s": time, "queries": {}}
        for backend in ("cpu", "gpu"):
            data = bundle.reset_cpu()
            data.qpos[:] = trace[f"cross_hand_{backend}_qpos"][tick]
            data.qvel[:] = trace[f"cross_hand_{backend}_qvel"][tick]
            mujoco.mj_forward(model, data)
            contacts = []
            for c in data.contact:
                pair = [model.geom(int(g)).name for g in c.geom]
                if any(n.startswith("left_/") for n in pair) and any(n.startswith("right_/") for n in pair):
                    contacts.append({"pair": sorted(pair), "distance_m": float(c.dist)})
            row["queries"][backend] = contacts
        report["cross_hand_geometry_queries"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cross_hand_geometry_queries"}, indent=2))


if __name__ == "__main__":
    main()
