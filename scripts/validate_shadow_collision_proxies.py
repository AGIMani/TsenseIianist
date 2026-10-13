#!/usr/bin/env python3
"""Compare colliders with fixed controls and optional saved absolute targets.

No SAC checkpoint is loaded: old targets on new physics are a stress test,
not an evaluation of the trained policy. Scene exports are temporary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.mjlab_env.bundle import export_scene, SceneBundle
from scripts.validate_contact_stability import CASES, category, metrics, targets


def trajectory(bundle, controls):
    model, data = bundle.model, bundle.reset_cpu()
    _, qa, va = targets(model, bundle.reset_ctrl)
    positions, velocities = [], []
    forces = {name: 0. for name in CASES}
    pairs = {}
    for control in controls:
        data.ctrl[:] = control
        for _ in range(round(bundle.manifest["control_dt"] / model.opt.timestep)):
            mujoco.mj_step(model, data)
            positions.append(data.qpos[qa].copy())
            velocities.append(data.qvel[va].copy())
            for index, contact in enumerate(data.contact):
                kind = category(model, contact.geom)
                if kind is None or contact.efc_address < 0:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                forces[kind] = max(forces[kind], float(abs(force[0])))
                pair = " | ".join(sorted(model.geom(int(g)).name for g in contact.geom))
                row = pairs.setdefault(pair, {"peak_normal_force_N": 0., "min_distance_m": 0.})
                row["peak_normal_force_N"] = max(row["peak_normal_force_N"], float(abs(force[0])))
                row["min_distance_m"] = min(row["min_distance_m"], float(contact.dist))
    q, v = np.asarray(positions), np.asarray(velocities)
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite simulation state")
    tail = round(2 / model.opt.timestep)
    return {"tail_two_seconds": metrics(q[-tail:], v[-tail:], model.opt.timestep),
            "all_steps_peak_finger_speed_rad_s": float(abs(v).max()),
            "all_steps_peak_normal_force_N": forces,
            "top_contacts": sorted((dict(pair=k, **v) for k, v in pairs.items()),
                                   key=lambda r: r["peak_normal_force_N"], reverse=True)[:15]}


def local_old_states(bundle, recorded):
    """Revisit the three contact events from the 2026-09-25 motion audit."""
    model = bundle.model
    names = [model.joint(int(j)).name for j in model.actuator_trnid[bundle.action_ids, 0]]
    if names != recorded["joint_names"].tolist():
        raise ValueError("Saved contact-state joint contract differs from current model")
    rows = []
    for time in (.7, 1.645, 2.465):
        tick = round(time / model.opt.timestep)
        data = bundle.reset_cpu()
        data.time = time
        data.qpos[:] = recorded["pre_qpos"][tick]
        data.qvel[:] = recorded["pre_qvel"][tick]
        data.ctrl[:] = recorded["ctrl"][tick]
        data.qacc_warmstart[:] = recorded["qacc"][tick - 1]
        mujoco.mj_step(model, data)
        contacts = []
        for index, contact in enumerate(data.contact):
            if category(model, contact.geom) is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            contacts.append({"pair": [model.geom(int(g)).name for g in contact.geom],
                             "distance_m": float(contact.dist), "force_N": float(force[0])})
        rows.append({"old_state_time": time,
                     "contacts": sorted(contacts, key=lambda c: c["force_N"], reverse=True)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--old-contact-states", type=Path,
                        help="Optional baseline.npz from the 2026-09-25 contact audit")
    args = parser.parse_args()
    report = {"notes": "Only collision profile changes between branches; all other current source parameters identical. Recorded ctrl replay is a stress test, not a new policy evaluation.",
              "source_hashes": {}, "profiles": {}}
    for relative in ("models/hands/urdf_model.py", "models/hands/tsensei_hand.py",
                     "models/hands/tsensei_hand_constants.py", "assets/collision_proxies.json"):
        report["source_hashes"][relative] = hashlib.sha256((ROOT / "tsensei_v3_2_pianist" / relative).read_bytes()).hexdigest()
    recorded = np.load(args.rollout)["ctrl"] if args.rollout else None
    old_states = np.load(args.old_contact_states) if args.old_contact_states else None
    for profile in ("legacy_mesh_v1", "convex_hull_v1"):
        os.environ["TSENSEI_V32_COLLISION_PROFILE"] = profile
        with tempfile.TemporaryDirectory(prefix="v32-collision-check-") as temporary:
            export_scene(Path(temporary) / "scene", build_reference=False)
            bundle = SceneBundle(Path(temporary) / "scene", disable_wrist_tz=True, disable_wrist_yaw=True)
            data = bundle.reset_cpu()
            initial = [{"pair": [bundle.model.geom(int(g)).name for g in c.geom], "distance_m": float(c.dist)} for c in data.contact]
            entry = {"initial_contacts": initial, "hold_cases": {}}
            controls, _, _ = targets(bundle.model, bundle.reset_ctrl)
            for name, control in zip(CASES, controls):
                entry["hold_cases"][name] = trajectory(bundle, np.tile(control, (400, 1)))
            if recorded is not None:
                if recorded.shape[1] != bundle.model.nu:
                    raise ValueError("Recorded ctrl size does not match current 47-action scene")
                entry["recorded_targets"] = trajectory(bundle, recorded)
            if old_states is not None:
                entry["local_old_states"] = local_old_states(bundle, old_states)
            report["profiles"][profile] = entry
            print(profile, json.dumps({"initial_contact_count": len(initial), "tail_velocity_rms": {k: v["tail_two_seconds"]["velocity_rms_rad_s"] for k, v in entry["hold_cases"].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
