#!/usr/bin/env python3
"""CPU-only full-capsule clearance and bounded finger posture feasibility audit.

Does not edit source parameters or step physics. Static IK endpoints are not a
proof of collision-free dynamic approach or compatible music performance.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import mujoco
    import numpy as np
    from scipy.optimize import minimize
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    metadata = json.loads(a.config.read_text())
    recipe = metadata['source_runtime'];prep = recipe['preparation'] or {}
    current = make_recipe(**recipe['pose'], **recipe['model'], **recipe['environment'],
        preparation_time=prep.get('duration', 0.), preparation_lift=prep.get('lift', .02),
        preparation_mode=prep.get('mode', 'fixed_v1'))
    scene, task, _ = build(current)
    if metadata.get('no_arms') or metadata['cli'].get('no_arms'):
        scene, task = without_arms(scene, task, scene.parent/'hand_only_bundle')
    assert sha256(scene/'manifest.json') == metadata['scene_sha256']
    assert sha256(task/'task.json') == metadata['task_sha256']
    bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=prep)
    model = bundle.model;data = bundle.reset_cpu()
    data.qpos[:] = bundle.origin_qpos
    data.mocap_pos[:] = bundle.origin_mocap_pos
    data.mocap_quat[:] = bundle.origin_mocap_quat
    mujoco.mj_forward(model, data)
    origin = data.qpos.copy()
    task_meta = json.loads((task/'task.json').read_text())
    keys = [model.geom(name).id for name in task_meta['key_geoms']]
    def extent(gid):
        center = data.geom_xpos[gid]
        rotation = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ends = center + np.asarray([-1., 1.])[:, None]*rotation[:, 2]*size[1]
            return ends.min(0)-size[0], ends.max(0)+size[0], ends
        assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        half = np.abs(rotation)@size
        return center-half, center+half, None
    key_rows = [dict(name=model.geom(g).name, min=extent(g)[0].tolist(), max=extent(g)[1].tolist()) for g in keys]
    white = max(row['max'][2] for row in key_rows if 'white' in row['name'])
    black = max(row['max'][2] for row in key_rows if 'black' in row['name'])
    groups = {}
    for side in ('right', 'left'):
        for finger in ('thumb', 'index', 'mid', 'ring', 'pinky'):
            geoms = [g for g in range(model.ngeom) if (name := model.geom(g).name)
                     and name.startswith(side+'_/') and '_'+finger+'_' in name and '_collision_' in name]
            joints = [j for j in range(model.njnt) if (name := model.joint(j).name)
                      and name.startswith(side+'_/') and '_'+finger+'_' in name]
            assert len(geoms) == len(joints) == 4, (side, finger, geoms, joints)
            groups[side+'_'+finger] = (geoms, joints)
    def finger_rows(geoms):
        return [dict(name=model.geom(g).name, body=model.body(model.geom_bodyid[g]).name,
            type=int(model.geom_type[g]), radius=float(model.geom_size[g,0]), half_length=float(model.geom_size[g,1]),
            min=extent(g)[0].tolist(), max=extent(g)[1].tolist(), capsule_endpoints=extent(g)[2].tolist(),
            white_plane_clearance=float(extent(g)[0][2]-white),
            black_plane_clearance=float(extent(g)[0][2]-black)) for g in geoms]
    origin_rows = {name: finger_rows(geoms) for name, (geoms, _) in groups.items()}
    solved = []
    # Four-joint independent fingers, fixed original wrist/mount. Both surface
    # choices are measured; neither is silently selected as a runtime contract.
    for name, (geoms, joints) in groups.items():
        qids = model.jnt_qposadr[joints];bounds = model.jnt_range[joints].copy()
        for i, joint in enumerate(joints):
            if 'thumb' not in name and '_MCP_' in model.joint(joint).name:
                bounds[i, 1] = 1.2
        start = origin[qids].copy()
        for label, plane in (('white_plus_2mm', white+.002), ('black_plus_2mm', black+.002)):
            data.qpos[:] = origin
            def clearance(q):
                data.qpos[qids] = q
                mujoco.mj_kinematics(model, data)
                return np.array([extent(g)[0][2]-plane for g in geoms])
            before = clearance(start)
            if before.min() >= -1e-8:
                candidate=start.copy();success=True;message='origin already feasible';iterations=0
            else:
                opt = minimize(lambda q: float(np.square(q-start).sum()), start,
                    method='SLSQP', bounds=bounds, constraints=[dict(type='ineq', fun=clearance)],
                    options=dict(maxiter=150, ftol=1e-10))
                candidate=opt.x;success=bool(opt.success);message=str(opt.message);iterations=int(opt.nit)
            after = clearance(candidate)
            solved.append(dict(finger=name, criterion=label, threshold_z=plane,
                solver_success=success, feasible=bool(after.min() >= -1e-7), solver_message=message,
                iterations=iterations, min_initial_clearance=float(before.min()),
                min_final_clearance=float(after.min()), joint_delta_norm=float(np.linalg.norm(candidate-start)),
                joints={model.joint(j).name:float(v) for j,v in zip(joints,candidate)},
                origin_joints={model.joint(j).name:float(v) for j,v in zip(joints,start)},
                geometry=finger_rows(geoms)))
    data.qpos[:] = origin;mujoco.mj_forward(model,data)
    report = dict(config=str(a.config.resolve()), scene_sha256=metadata['scene_sha256'],
        task_sha256=metadata['task_sha256'], source_recipe=current,
        white_rest_top_z=white, black_rest_top_z=black, key_geometries=key_rows,
        origin=origin_rows, static_feasibility=solved,
        limitation='Static independent-finger endpoints only, fixed wrists; no dynamics, no clearance path, no self-collision guarantee')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(white_rest_top_z=white, black_rest_top_z=black,
        origin_minima={name:dict(white_mm=1000*min(r['white_plane_clearance'] for r in rows),
            black_mm=1000*min(r['black_plane_clearance'] for r in rows)) for name,rows in origin_rows.items()},
        static_feasibility=[{k:v for k,v in row.items() if k not in ('geometry','joints','origin_joints')}
                            for row in solved]), indent=2))


if __name__ == '__main__':
    main()
