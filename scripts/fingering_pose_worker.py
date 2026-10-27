#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""CPU-only kinematic preview. JSON lines on stdin/stdout, diagnostics stderr.

Never advances dynamics, synthesizes a physical key press, or renders on a GPU.
The desktop only draws the returned display meshes and rigid transforms.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def emit(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


class Preview:
    def __init__(self, cache, *, local=False):
        import numpy as np
        import mujoco
        from scipy.spatial import ConvexHull
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
        from tsensei_v3_2_pianist.fingering_targets import target_position, key_reference
        self.np, self.mj = np, mujoco
        self.target_position = target_position
        self.key_reference = key_reference
        cache.mkdir(parents=True, exist_ok=True)
        if local:
            # A kinematic editor needs the source-built physical scene, not a
            # second full music-training environment or quantized task bundle.
            # Use exactly the same export_scene API in a Composer subprocess.
            recipe = dict(kind='editor_scene_only', pose=dict(world_x=.14, world_z=.06,
                          right_world_y=.15, left_world_y=-.15, palm_base_yaw=float(np.pi/6)),
                          collision_profile='convex_hull_v1', drive_profile='shadow_v1',
                          source_archive_sha256=(ROOT/'EDITOR_SOURCE_SHA256').read_text().strip())
            (cache/'recipe.json').write_text(json.dumps(recipe, indent=2))
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT), TSENSEI_V32_EXPORT_POSE=json.dumps(recipe['pose']),
                       TSENSEI_V32_COLLISION_PROFILE='convex_hull_v1', TSENSEI_V32_DRIVE_PROFILE='shadow_v1')
            command = [str(ROOT/'training/.venv/bin/python'), str(Path(__file__).resolve()),
                       '--local', '--compile-scene', '--cache', str(cache)]
            print(json.dumps(dict(stage='compile_scene', command=command)), file=sys.stderr, flush=True)
            compiled = subprocess.run(command, env=env, stdout=sys.stderr, stderr=sys.stderr)
            compiled.check_returncode()
            scene = cache/'scene'
        else:
            recipe = make_recipe(world_x=.14, world_z=.06, palm_base_yaw=np.pi/6, initial_buffer_time=0,
                                 preparation_time=0, task_profile="tsensei_collision_v1",
                                 drive_profile="shadow_v1", exact_note_timing=True)
            (cache / "recipe.json").write_text(json.dumps(recipe, indent=2))
            scene, _, _ = build(recipe)
        # Finish the Composer subprocess before importing Torch, to avoid
        # overlapping two large runtimes during local cold construction.
        import torch
        from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler, FOLLOWER_INDICES
        self.torch = torch
        torch.set_num_threads(1)
        self.bundle = b = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True)
        self.model, self.data = b.model, b.reset_cpu()
        m = self.model
        ranges = json.loads((ROOT / "configs/finger_coupling/angle_ranges_60_80.json").read_text())
        # The ranges file is the same input accepted by the unified launcher.
        ranges = ranges.get("angle_ranges_deg", ranges)
        self.coupler = FingerCoupler(b, device="cpu", options={
            "schema": 1, "mode": "distal_angle_range_v1",
            "angle_ranges_deg": ranges, "reduce_observation": False})
        self.full = b.active_action_indices
        self.qadr = m.jnt_qposadr[m.actuator_trnid[b.action_ids, 0]]
        self.base = b.neutral_action.copy()
        self.low, self.high = b.low.copy(), b.high.copy()
        for full, aid in zip(self.full, b.action_ids):
            joint = m.actuator_trnid[aid, 0]
            self.low[full] = max(self.low[full], m.jnt_range[joint, 0])
            self.high[full] = min(self.high[full], m.jnt_range[joint, 1])
        self.followers = set(FOLLOWER_INDICES)
        self.sites = [m.site(f"{side}_/{finger}_tip").id
                      for side in ("right", "left")
                      for finger in ("thumb", "index", "mid", "ring", "pinky")]
        self.keys = {}
        self.geoms = []
        self.meshes = []
        self.place(self.base)
        for gid in range(m.ngeom):
            name = m.geom(gid).name or ""
            side = 0 if name.startswith("right_/") else 1 if name.startswith("left_/") else None
            finger = -1
            if side is not None:
                if "_collision_" not in name:
                    continue
                for f, short in enumerate(("thumb", "index", "mid", "ring", "pinky")):
                    if f"_hand_{short}_" in name:
                        finger = side * 5 + f
            elif name.startswith("piano/") and ("white_key_geom_" in name or "black_key_geom_" in name):
                pass
            else:
                continue
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gid]
                verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
                # Display-only level of detail. The IK model and its collision
                # geometry remain untouched; never use this for contact tests.
                if len(verts) > 42:
                    chosen = [int(np.argmax(verts[:, 0]))]
                    dist = np.full(len(verts), np.inf)
                    for _ in range(41):
                        dist = np.minimum(dist, ((verts - verts[chosen[-1]])**2).sum(axis=1))
                        chosen.append(int(np.argmax(dist)))
                    verts = verts[chosen]
                faces = ConvexHull(verts).simplices
            elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                verts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[gid]
                faces = ConvexHull(verts).simplices
            else:
                continue
            pitch = None
            if side is None:
                # Model key names use the full piano index, not white/black ordinal.
                key_index = int(name.rsplit("_", 1)[1])
                pitch = key_index + 21
                self.keys[pitch] = gid
            self.geoms.append(gid)
            self.meshes.append(dict(name=name, vertices=verts.tolist(), faces=faces.tolist(),
                                    finger=finger, side=side, pitch=pitch,
                                    black="black_key_geom_" in name))
        if len(self.keys) != 88 or sum(x["side"] is not None for x in self.meshes) != 44:
            raise ValueError("Expected 88 piano keys and 44 current hand collision hulls")
        self.rest_key_pos = self.data.geom_xpos.copy()
        self.rest_site_pos = self.data.site_xpos.copy()
        self.last_signature = None
        self.last_result = None
        self.place(self.base)
        initial = self.frame([], [])
        for side, label in enumerate(('right', 'left')):
            for mesh in self.meshes:
                if mesh['side'] == side and not mesh['name'].startswith(label + '_/'):
                    raise ValueError('Hand geometry side/name mismatch')
        keyboard_y = [self.rest_key_pos[self.keys[p], 1] for p in sorted(self.keys)]
        if not np.all(np.diff(keyboard_y) > 0):
            raise ValueError('Keyboard pitches must increase along world +Y')
        self.ready = dict(type="ready", pid=os.getpid(), meshes=self.meshes, frame=initial,
                          version=mujoco.__version__, model="A3.2 / supplied_hand_v6",
                          source_sha256=(ROOT / "EDITOR_SOURCE_SHA256").read_text().strip(),
                          geometry="44 convex hulls; display-only simplified surfaces",
                          coupling=self.coupler.reference_diagnostics,
                          physics=False, gpu=False)
        (cache / "ready.json").write_text(json.dumps({k: v for k, v in self.ready.items() if k not in ("meshes", "frame")}, indent=2))

    def place(self, action):
        np, torch, d, m = self.np, self.torch, self.data, self.model
        d.qpos[:] = self.bundle.origin_qpos
        d.qpos[self.qadr] = action[self.full]
        self.mj.mj_fwdPosition(m, d)
        coupled, diagnostics = self.coupler.apply(
            torch.as_tensor(action, dtype=torch.float32)[None],
            actual_qpos=torch.as_tensor(d.qpos.copy(), dtype=torch.float32)[None],
            body_xmat=torch.as_tensor(d.xmat.copy(), dtype=torch.float32)[None])
        values = coupled[0].numpy().astype(float)
        d.qpos[self.qadr] = values[self.full]
        self.mj.mj_fwdPosition(m, d)
        return values, diagnostics

    def frame(self, active, targets):
        np, d = self.np, self.data
        return dict(transforms=np.concatenate(
            (d.geom_xpos[self.geoms], d.geom_xmat[self.geoms]), axis=1).tolist(),
            tips=d.site_xpos[self.sites].tolist(), active=active, targets=targets)

    def solve(self, request):
        np, torch = self.np, self.torch
        active = request.get("active", [])
        if len(active) > 128 or any(
            type(a.get("finger")) is not int or not 0 <= a["finger"] < 10
            or type(a.get("pitch")) is not int or a["pitch"] not in self.keys for a in active
        ):
            raise ValueError("Only piano pitches 21–108 and finger IDs 0–9 are supported in 3D")
        signature = tuple(sorted((a["finger"], a["pitch"]) for a in active))
        if signature == self.last_signature:
            return dict(self.last_result, request_id=request["request_id"])
        started = time.monotonic()
        groups = {}
        for f, pitch in signature:
            groups.setdefault(f, []).append(pitch)
        # Contradictory assignments stay visible in the score. Do not silently
        # choose one of two keys for the same finger.
        conflicted = [f for f, pitches in groups.items() if len(pitches) > 1]
        usable = [(f, pitches[0]) for f, pitches in groups.items() if len(pitches) == 1]
        targets = []
        for f, pitch in usable:
            gid = self.keys[pitch]
            sid, half_sizes = self.key_reference(self.model, gid)
            # Rest upper-face reference remains independent of mesh centroid.
            reference_pos = self.rest_site_pos[sid]
            pos = self.target_position(reference_pos, half_sizes,
                                       self.model.geom(gid).name, thumb=f in (0, 5),
                                       task_profile="tsensei_collision_v1")
            targets.append(dict(finger=f, pitch=pitch, position=pos.tolist()))
        x = self.base.copy()
        all_errors = []
        for side in (0, 1):
            subset = [t for t in targets if t["finger"] // 5 == side]
            if not subset:
                continue
            fs = [t["finger"] for t in subset]
            goal = np.array([t["position"] for t in subset])
            allowed = {side*25 + i for i in (20, 21, 23)}
            for f in fs:
                allowed.update(range(side*25 + (f % 5)*4, side*25 + (f % 5)*4 + 4))
            drivers = [i for i in self.full if i in allowed and i not in self.followers]

            def evaluate(a):
                values, diag = self.place(a)
                err = (goal-self.data.site_xpos[np.array(self.sites)[fs]]).reshape(-1)
                feasible = not bool(diag["coupling_infeasible"].any())
                return err, feasible

            err, feasible = evaluate(x)
            for iteration in range(18):
                if max(np.linalg.norm(err.reshape(-1, 3), axis=1)) < .001:
                    break
                jac = np.zeros((len(err), len(drivers)))
                for j, idx in enumerate(drivers):
                    trial = x.copy()
                    delta = min(2e-4, self.high[idx]-x[idx])
                    if delta < 1e-7:
                        delta = -min(2e-4, x[idx]-self.low[idx])
                    if abs(delta) < 1e-7:
                        continue
                    trial[idx] += delta
                    e, _ = evaluate(trial)
                    jac[:, j] = (err-e)/delta
                step = jac.T @ np.linalg.solve(jac @ jac.T + .0001*np.eye(len(err)), err)
                step = np.clip(step, -.12, .12)
                for j, idx in enumerate(drivers):
                    if idx % 25 in (20, 21):
                        step[j] = np.clip(step[j], -.04, .04)
                improved = False
                for scale in (1., .5, .2):
                    trial = x.copy()
                    trial[drivers] = np.clip(x[drivers] + scale*step, self.low[drivers], self.high[drivers])
                    e, ok = evaluate(trial)
                    if ok and np.linalg.norm(e) < np.linalg.norm(err)-1e-8:
                        x, err, feasible = trial, e, ok
                        improved = True
                        break
                if not improved:
                    break
            all_errors.extend((1000*np.linalg.norm(err.reshape(-1, 3), axis=1)).tolist())
        values, diagnostics = self.place(x)
        if bool(diagnostics["coupling_infeasible"].any()):
            raise ValueError("Pose violates the current 60–80° coupling; refusing preview")
        if np.any(values[self.full] < self.low[self.full]-1e-6) or np.any(values[self.full] > self.high[self.full]+1e-6):
            raise ValueError("Pose violates joint/action limits; refusing preview")
        result = dict(type="pose", request_id=request["request_id"], **self.frame(active, targets),
                      error_mm=max(all_errors, default=0), conflicts=conflicted,
                      coupling_angles=diagnostics["coupling_target_angle_deg"][0].tolist(),
                      elapsed_s=time.monotonic()-started, qpos=self.data.qpos.tolist(),
                      physical_key_positions_unchanged=bool(np.array_equal(
                          self.data.qpos[self.bundle.key_qadr],
                          self.bundle.origin_qpos[self.bundle.key_qadr])))
        self.last_signature, self.last_result = signature, result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--local", action="store_true", help="Explicit bounded local CPU preview only")
    parser.add_argument('--compile-scene', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.local:
        from fingering_editor_limits import verify_limits
        print(json.dumps(dict(local_limits=verify_limits(check_available=not args.compile_scene))), file=sys.stderr, flush=True)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MUJOCO_GL') != 'disable':
            parser.error('Local preview requires CUDA_VISIBLE_DEVICES empty and MUJOCO_GL=disable')
    elif not os.environ.get("SSH_CONNECTION"):
        parser.error("This worker must run through SSH on volc-a32; local model loading is disabled")
    if args.compile_scene:
        if not args.local:
            parser.error('--compile-scene requires the bounded local path')
        with contextlib.redirect_stdout(sys.stderr):
            from tsensei_v3_2_pianist.mjlab_env.bundle import export_scene
            export_scene(args.cache/'scene', gravity_compensation=True, build_reference=False)
        return
    with contextlib.redirect_stdout(sys.stderr):
        preview = Preview(args.cache, local=args.local)
    emit(preview.ready)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = preview.solve(request)
            emit(result)
        except Exception as exc:
            emit(dict(type="error", request_id=request.get("request_id"), error=str(exc)))


if __name__ == "__main__":
    main()
