#!/usr/bin/env python3
"""Compare CPU and one GPU world at every 5-ms substep of the Shadow profiles.

This is a short physical trajectory comparison, not SAC convergence evidence.
The same float32 physical targets and model are used by both backends.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
from scripts.validate_contact_stability import CASES, category, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.)
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds) or not np.isclose(args.seconds * 20, round(args.seconds * 20)):
        parser.error("seconds must be positive and contain whole 50-ms control frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.perf_counter()
    report = {"status": "running", "num_envs": 1, "seconds_per_case": args.seconds,
              "notes": "Same model and actual float32 physical targets; fresh reset per case. CPU mj_forward matches GPU at each control boundary. No long-time equivalence or training convergence claim.", "cases": {}}
    traces = {}
    try:
        recipe = make_recipe(initial_buffer_time=0, task_profile="tsensei_collision_v1", drive_profile="shadow_v1", collision_profile="convex_hull_v1")
        scene_path, task_path, _ = build(recipe)
        scene_path, task_path = without_arms(scene_path, task_path, scene_path.parent / "hand_only")
        scene = PianoScene(scene_path, num_envs=1, disable_wrist_tz=True, disable_wrist_yaw=True,
                           action_profile="direct_joint_v1", physics_graph=True)
        bundle, model = scene.bundle, scene.model
        report.update(versions=recipe["versions"], source_sha256=recipe["source_sha256"],
                      model_sha256=hashlib.sha256((scene_path / "scene.xml").read_bytes()).hexdigest(),
                      physics_dt=model.opt.timestep, decimation=scene.decimation,
                      gpu=torch.cuda.get_device_name(), nq=model.nq, nu=model.nu)
        candidates, _, _ = targets(model, bundle.reset_ctrl)
        cases = {"neutral": bundle.reset_ctrl, **dict(zip(CASES, candidates))}
        slide_jids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)
        hand_jids = model.actuator_trnid[bundle.action_ids, 0]
        hinge_jids = hand_jids[model.jnt_type[hand_jids] == mujoco.mjtJoint.mjJNT_HINGE]
        groups = {"hand_slide": slide_jids, "hand_hinge": hinge_jids, "key_hinge": bundle.key_ids}
        steps = round(args.seconds / bundle.manifest["control_dt"])
        for name, control in cases.items():
            scene.reset()
            cpu = bundle.reset_cpu()
            physical = bundle.neutral_action.copy()
            physical[bundle.active_action_indices] = control[bundle.action_ids]
            action = scene.policy_action(physical[None])
            effective = scene.physical_action(action)[0].cpu().numpy()
            cpu.ctrl[bundle.action_ids] = effective[np.asarray(bundle.active_action_indices)]
            cqs, gqs, cvs, gvs, ckeys, gkeys = [], [], [], [], [], []
            contact_ticks = {kind: 0 for kind in CASES}
            initial_qerr = float(abs(cpu.qpos - scene.data.qpos[0].cpu().numpy()).max())
            ctrl_error = float(abs(effective[bundle.active_action_indices] - control[bundle.action_ids]).max())
            def sample(current):
                mujoco.mj_step(model, cpu)
                gpu_q = current.data.qpos[0].cpu().numpy().copy()
                gpu_v = current.data.qvel[0].cpu().numpy().copy()
                cqs.append(cpu.qpos.copy()); cvs.append(cpu.qvel.copy())
                gqs.append(gpu_q); gvs.append(gpu_v)
                ckeys.append(bundle.key_activation(cpu.qpos))
                gkeys.append(bundle.key_activation(gpu_q))
                present = {category(model, c.geom) for c in cpu.contact if c.efc_address >= 0}
                for kind in contact_ticks:
                    contact_ticks[kind] += int(kind in present)
            for _ in range(steps):
                scene.step(action, substep_callback=sample)
                mujoco.mj_forward(model, cpu)
            cq, gq, cv, gv = map(np.asarray, (cqs, gqs, cvs, gvs))
            ck, gk = np.asarray(ckeys), np.asarray(gkeys)
            finite = all(np.isfinite(a).all() for a in (cq, gq, cv, gv))
            delta_q, delta_v = abs(cq - gq), abs(cv - gv)
            mismatch = np.any(ck != gk, axis=1)
            row = {"finite": bool(finite), "initial_qpos_max_error": initial_qerr,
                   "float32_action_roundtrip_max_error": ctrl_error,
                   "substeps": len(cq), "key_mismatch_substeps": int(mismatch.sum()),
                   "first_key_mismatch_s": float((np.flatnonzero(mismatch)[0] + 1) * model.opt.timestep) if mismatch.any() else None,
                   "cpu_active_key_substeps": int(np.any(ck, axis=1).sum()),
                   "gpu_active_key_substeps": int(np.any(gk, axis=1).sum()), "windows": {}}
            row["cpu_contact_substeps_by_category"] = contact_ticks
            for horizon in (.05, .1, .25, .5, 1., 2.):
                if horizon > args.seconds + 1e-10:
                    continue
                count = round(horizon / model.opt.timestep)
                window = {}
                for group, ids in groups.items():
                    qa, va = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
                    worst = np.unravel_index(np.argmax(delta_q[:count, qa]), (count, len(ids)))
                    window[group] = {"qpos_max_error": float(delta_q[:count, qa].max()),
                                     "qvel_max_error": float(delta_v[:count, va].max()),
                                     "worst_qpos_joint": model.joint(int(ids[worst[1]])).name,
                                     "worst_qpos_time_s": float((worst[0] + 1) * model.opt.timestep)}
                window["key_mismatch_substeps"] = int(mismatch[:count].sum())
                row["windows"][str(horizon)] = window
            report["cases"][name] = row
            for suffix, value in (("cpu_qpos", cq), ("gpu_qpos", gq), ("cpu_qvel", cv), ("gpu_qvel", gv), ("cpu_keys", ck), ("gpu_keys", gk)):
                traces[name + "_" + suffix] = value
            print(name, json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
            print("last_window", json.dumps(next(reversed(row["windows"].values()))), flush=True)
        report["status"] = "finite_short_rollouts_recorded" if all(c["finite"] for c in report["cases"].values()) else "nonfinite_state"
        np.savez_compressed(args.output.with_suffix(".npz"), **traces)
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
