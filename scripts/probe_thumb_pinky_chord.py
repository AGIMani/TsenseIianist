#!/usr/bin/env python3
"""CPU static 11-joint IK audit for L1:C4 + L5:C3, not a playing controller.

Fits contact endpoints at fully depressed keys, records the separate legacy
reward-point errors and real MuJoCo static contacts. No stepping or training.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-config', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--starts', type=int, default=6)
    p.add_argument('--max-nfev', type=int, default=180)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import mujoco
    import numpy as np
    import torch
    from scipy.optimize import least_squares
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.finger_clearance import FingerClearanceGuard
    from tsensei_v3_2_pianist.fingering_targets import target_position
    torch.set_num_threads(1)
    saved = json.loads(args.run_config.read_text())
    old_recipe = saved['source_runtime']
    fields = {**old_recipe['pose'], **old_recipe['model'], **old_recipe['environment']}
    thumb = fields.pop('thumb_target', None)
    if thumb:
        fields.update(thumb_target_mode=thumb['mode'], thumb_white_target_distance=thumb['white_distance'],
                      thumb_black_target_distance=thumb['black_distance'])
    prep = old_recipe.get('preparation') or {}
    recipe = make_recipe(**fields, preparation_time=prep.get('duration', 0),
        preparation_lift=prep.get('lift', .02), preparation_mode=prep.get('mode', 'fixed_v1'))
    scene, task, _ = build(recipe)
    scene, task = without_arms(scene, task, scene.parent/'hand_only')
    b = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'])
    m = b.model
    g = FingerClearanceGuard(b, device='cpu', options=dict(schema=1, mode='full_finger_clearance_v1',
                            active_command_mode='active_contact_v1'))
    target_pairs = ((5, 39), (9, 27))
    groups = [g.groups[finger] for finger, _ in target_pairs]
    joints = list(groups[0]['chain']) + list(groups[1]['chain'][3:])
    if len(joints) != 11 or len(set(joints)) != 11:
        raise ValueError('Expected three shared wrist and four joints per finger')
    qadr = m.jnt_qposadr[joints]
    bounds = m.jnt_range[joints].copy()
    # Match Shadow software caps without shrinking mechanical joint ranges.
    for index, joint in enumerate(joints):
        name = m.joint(joint).name
        if 'pinky_MCP' in name:
            bounds[index, 1] = min(bounds[index, 1], 1.2)
    data = b.reset_cpu()
    data.qpos[:] = b.origin_qpos
    data.mocap_pos[:] = b.origin_mocap_pos
    data.mocap_quat[:] = b.origin_mocap_quat
    mujoco.mj_forward(m, data)
    origin = data.qpos.copy()
    recorded_origin = saved['preparation_contract']['origin']['joint_positions']
    origin_error = max(abs(float(origin[m.jnt_qposadr[j]])-recorded_origin[m.joint(j).name])
                       for j in range(m.njnt))
    if origin_error > 1e-10:
        raise ValueError('Current source origin differs from the recorded parent')
    recorded_model = json.loads((args.audit_dir/'model_indices.json').read_text())
    if recorded_model['geom_names'] != [m.geom(i).name for i in range(m.ngeom)]:
        raise ValueError('Current collision geometry layout differs from the recorded audit')
    geom_size_error = float(np.max(np.abs(m.geom_size-np.array(recorded_model['geom_size']))))
    if geom_size_error > 1e-10:
        raise ValueError('Current collision geometry sizes differ from the recorded audit')
    range_error = max(float(np.max(np.abs(m.jnt_range[m.joint(row['joint_name']).id]-row['joint_range'])))
                      for row in recorded_model['joints'])
    gain_error = max(float(np.max(np.abs(m.actuator_gainprm[row['actuator_id']]-row['gain'])))
                     for row in recorded_model['joints'])
    if range_error > 1e-10 or gain_error > 1e-10:
        raise ValueError('Current ranges or drive gains differ from the recorded audit')
    origin_x = origin[qadr].copy()
    task_manifest = json.loads((task/'task.json').read_text())
    key_geoms = np.array([m.geom(name).id for name in task_manifest['key_geoms']])
    tip_ids = [m.site(task_manifest['fingertip_sites'][finger]).id for finger, _ in target_pairs]
    rest_key_centers = data.geom_xpos[key_geoms].copy()
    rest_reward_targets = [target_position(rest_key_centers[key], m.geom_size[key_geoms[key]],
        m.geom(key_geoms[key]).name, thumb=finger in (0, 5), task_profile=fields['task_profile'],
        thumb_target=recipe['environment']['thumb_target'])
        for finger, key in target_pairs]
    left_geoms = [geom for group in g.groups[5:] for geom in group['geoms']]
    active_terminals = {group['geoms'][-1] for group in groups}
    radii = m.geom_size[left_geoms, 0].copy()
    spans = m.geom_size[left_geoms, 1].copy()
    penalty_indices = [index for index, geom in enumerate(left_geoms) if geom not in active_terminals]
    key_min = g.key_min_xy.numpy();key_max = g.key_max_xy.numpy()
    key_tops = g.key_top_z.numpy() if hasattr(g, 'key_top_z') else None
    if key_tops is None:
        key_tops = rest_key_centers[:, 2] + m.geom_size[key_geoms, 2]
    # This is a feasibility boundary, not a simulated press: explicitly set the
    # two desired key angles to their full-press limits, leave every other key at rest.
    for _, key in target_pairs:
        data.qpos[b.key_qadr[key]] = m.jnt_range[b.key_ids[key], 1]
    base_qpos = data.qpos.copy()
    mujoco.mj_forward(m, data)
    planes = []
    for finger, key in target_pairs:
        geom = key_geoms[key]
        normal = data.geom_xmat[geom].reshape(3, 3)[:, 2].copy()
        center = data.geom_xpos[geom].copy()
        planes.append((normal, float(center@normal + m.geom_size[geom, 2])))
    def capsule(geom):
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        radius, half = m.geom_size[geom, :2]
        ends = data.geom_xpos[geom] + np.array([-half, half])[:, None]*axis
        return ends, radius
    def residual(x):
        data.qpos[:] = base_qpos;data.qpos[qadr] = x
        mujoco.mj_kinematics(m, data)
        rows = []
        for index, ((finger, key), group, target, plane) in enumerate(zip(target_pairs, groups, rest_reward_targets, planes)):
            ends, radius = capsule(group['geoms'][-1]);normal, height = plane
            # Reward X/Y encourages the existing common target; the contact
            # objective is the capsule support surface, not the reward's Z.
            rows.extend((data.site_xpos[tip_ids[index], :2] - target[:2])*2)
            rows.append(((ends@normal).min()-radius-height)*3)
            low = ends.min(0)-radius;high = ends.max(0)+radius
            rows.extend(5*np.maximum(key_min[key]+.0002-low[:2], 0))
            rows.extend(5*np.maximum(high[:2]-key_max[key]+.0002, 0))
        axes = data.geom_xmat[left_geoms].reshape(-1, 3, 3)[:, :, 2]
        ends = data.geom_xpos[left_geoms, None, :] + np.array([-1., 1.])[None, :, None]*spans[:, None, None]*axes[:, None, :]
        low = ends.min(1)-radii[:, None];high = ends.max(1)+radii[:, None]
        overlap = ((high[:, None, :2] >= key_min[None]) & (low[:, None, :2] <= key_max[None])).all(-1)
        thresholds = np.where(overlap, key_tops[None]+.001, -.5).max(1)
        rows.extend(5*np.maximum(thresholds[penalty_indices]-low[penalty_indices, 2], 0))
        rows.extend(1e-5*(x-origin_x))
        return np.array(rows)
    warm_file = args.audit_dir/'best/detail-0/physics_200hz.npz'
    warm = np.load(warm_file)
    starts = [origin_x]
    for tick in (319, 329, 339):
        if warm['qpos'].shape[1] != m.nq:
            raise ValueError('Warm-start qpos layout differs from 47-action model')
        starts.append(warm['qpos'][tick, qadr])
    rng = np.random.default_rng(280927)
    while len(starts) < args.starts:
        x = starts[1].copy();x[3:] = rng.uniform(bounds[3:, 0], bounds[3:, 1])
        starts.append(x)
    assigned = torch.zeros(1, 10, 88, dtype=torch.bool)
    for finger, key in target_pairs:assigned[0, finger, key] = True
    results = []
    begin = time.monotonic()
    for trial, initial in enumerate(starts[:args.starts]):
        solved = least_squares(residual, np.clip(initial, bounds[:, 0]+1e-8, bounds[:, 1]-1e-8),
            bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=args.max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual(solved.x);mujoco.mj_forward(m, data)
        physical = g.physical_qpos(torch.tensor(data.qpos, dtype=torch.float32)[None])
        guard = g.evaluate(physical, torch.zeros(1, 2), assigned)
        fingers = []
        for index, ((finger, key), group, target, plane) in enumerate(zip(target_pairs, groups, rest_reward_targets, planes)):
            ends, radius = capsule(group['geoms'][-1]);normal, height = plane
            low=ends.min(0)-radius;high=ends.max(0)+radius
            current_target = target_position(data.geom_xpos[key_geoms[key]], m.geom_size[key_geoms[key]],
                m.geom(key_geoms[key]).name, thumb=finger in (0,5), task_profile=fields['task_profile'],
                thumb_target=recipe['environment']['thumb_target'])
            bottom_height = height - 2*m.geom_size[key_geoms[key], 2]
            bottom_clearance = min(float((capsule(geom)[0]@normal).min()-capsule(geom)[1]-bottom_height)
                                   for geom in group['geoms'])
            fingers.append(dict(finger=group['name'], midi=key+21,
                reward_target_rest=target.tolist(), fingertip=data.site_xpos[tip_ids[index]].tolist(),
                fingertip_to_reward_distance_m=float(np.linalg.norm(data.site_xpos[tip_ids[index]]-target)),
                reward_target_pressed=current_target.tolist(),
                fingertip_to_pressed_reward_distance_m=float(np.linalg.norm(data.site_xpos[tip_ids[index]]-current_target)),
                xy_error_m=float(np.linalg.norm(data.site_xpos[tip_ids[index], :2]-target[:2])),
                capsule_to_pressed_top_m=float((ends@normal).min()-radius-height),
                whole_finger_to_assigned_key_bottom_m=bottom_clearance,
                capsule_xy_within_assigned_key=bool(((low[:2]>=key_min[key])&(high[:2]<=key_max[key])).all()),
                min_capsule_z_m=float(low[2]), active_force_permitted=bool(guard['finger_active_force_permitted'][0,finger]),
                geometry_deficit_m=float(guard['finger_clearance_deficit_m'][0,finger])))
        contacts = []
        for contact in data.contact[:data.ncon]:
            if contact.dist > 1e-5:continue
            first, second = int(contact.geom1), int(contact.geom2)
            names = [m.geom(first).name, m.geom(second).name]
            if not any(name and name.startswith('left_/') for name in names):continue
            key_local_normal_z = None
            for is_first, geom in ((True,first),(False,second)):
                if geom in key_geoms:
                    normal=contact.frame[:3]*(1 if is_first else -1)
                    key_local_normal_z=float(normal@data.geom_xmat[geom].reshape(3,3)[:,2])
            contacts.append(dict(geoms=names,distance_m=float(contact.dist),key_local_normal_z=key_local_normal_z))
        row=dict(trial=trial, solver_success=bool(solved.success), nfev=int(solved.nfev),
            residual_norm=float(np.linalg.norm(solved.fun)),elapsed_seconds=time.monotonic()-begin,
            joints={m.joint(j).name:float(v)for j,v in zip(joints, solved.x)},fingers=fingers,
            static_left_contacts=contacts, idle_finger_deficits_m=guard['finger_clearance_deficit_m'][0,[6,7,8]].tolist(),
            qpos=data.qpos.tolist(), mocap_pos=data.mocap_pos.tolist(), mocap_quat=data.mocap_quat.tolist())
        results.append(row)
        print(json.dumps({k:v for k,v in row.items()if k not in ('qpos','mocap_pos','mocap_quat','joints')}),flush=True)
    best = min(results,key=lambda row: sum(f['xy_error_m']+abs(f['capsule_to_pressed_top_m']) for f in row['fingers'])+sum(max(-c['distance_m'],0)for c in row['static_left_contacts']))
    report=dict(source_config=str(args.run_config), source_sha256=recipe['source_sha256'],
        current_scene_sha256=__import__('hashlib').sha256((scene/'manifest.json').read_bytes()).hexdigest(),
        nq=m.nq,nu=m.nu,action_size=len(b.active_action_indices)+1,
        parent_numeric_checks=dict(max_origin_joint_error=origin_error,max_geom_size_error_m=geom_size_error,
            max_joint_range_error=range_error,max_actuator_gain_error=gain_error),
        optimized_joints=[m.joint(j).name for j in joints],bounds=bounds.tolist(),
        origin_qpos=origin.tolist(),targets=[dict(finger='L1',midi=60),dict(finger='L5',midi=48)],
        best_trial=best['trial'], results=results,
        limitations=['Static endpoint search only; desired key angles are imposed for feasibility, not generated by dynamics.',
          'No path, friction, force, actuator-speed, learned-control, timing or F1 guarantee.',
          'Failure to find an endpoint does not prove impossibility; only 11 joints vary, other fingers are at origin.',
          'Source recipe reuses parent pose/drive/timing; current source manifest differs in metadata.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),qpos=best['qpos'],mocap_pos=best['mocap_pos'],mocap_quat=best['mocap_quat'])
    print('Saved '+str(args.output),flush=True)


if __name__ == '__main__':main()
