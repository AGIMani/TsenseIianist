#!/usr/bin/env python3
"""CPU geometric contact audit of recorded states; never steps a policy/physics.

Run remotely. Contact depths are reconstructed at saved poses, not recorded
solver forces. Counts include palm/same-digit diagnostics separately.
"""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def identity(name):
    for side in ('left', 'right'):
        if name.startswith(side + '_/'):
            digit = next((d for d in ('thumb', 'index', 'mid', 'ring', 'pinky')
                          if '_hand_' + d + '_' in name), None)
            return side, digit
    return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1)
    a = p.parse_args()
    path = a.model or a.replay / 'kinematic_scene/model.mjb'
    model = (mujoco.MjModel.from_binary_path(str(path)) if path.suffix == '.mjb'
             else mujoco.MjModel.from_xml_path(str(path)))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
             for i in range(model.nbody)]
    z = np.load(a.replay / 'physics_trace.npz', allow_pickle=False)
    pairs = {}; counts = collections.Counter(); samples = 0
    start = time.perf_counter()
    for frame in range(0, len(z['qpos']), a.stride):
        data.qpos[:] = z['qpos'][frame]
        data.mocap_pos[:] = z['mocap_pos'][frame]
        data.mocap_quat[:] = z['mocap_quat'][frame]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        seen = {}; categories = set()
        for con in data.contact[:data.ncon]:
            if con.dist > 0:
                continue
            b1, b2 = (int(model.geom_bodyid[g]) for g in con.geom)
            if model.body_weldid[b1] == model.body_weldid[b2]:
                continue
            n1, n2 = sorted((names[b1], names[b2]))
            s1, d1 = identity(n1); s2, d2 = identity(n2)
            if s1 is None or s1 != s2:
                continue
            category = ('different_digits' if d1 and d2 and d1 != d2 else
                        'same_digit' if d1 and d1 == d2 else 'palm_or_mount')
            categories.add(category)
            key = (n1, n2)
            if key not in seen or con.dist < seen[key][0]:
                seen[key] = (float(con.dist), con.pos.copy(), category)
        for key, (dist, pos, cat) in seen.items():
            item = pairs.setdefault(key, dict(bodies=list(key), category=cat,
                frames=[], depths_mm=[], deepest=dict(depth_mm=-1)))
            depth = -dist * 1000
            item['frames'].append(frame); item['depths_mm'].append(depth)
            if depth > item['deepest']['depth_mm']:
                item['deepest'] = dict(depth_mm=depth, frame=frame,
                    time_s=(frame+1)*float(z['physics_dt']), position_m=pos.tolist())
        counts.update(categories)
        if seen:
            counts['any_self'] += 1
        samples += 1
    result = []
    for item in pairs.values():
        ds = item.pop('depths_mm')
        item.update(samples=len(ds), fraction=len(ds)/samples,
                    depth_mean_mm=float(np.mean(ds)), depth_p95_mm=float(np.percentile(ds,95)))
        result.append(item)
    result.sort(key=lambda x: (-x['samples'], -x['deepest']['depth_mm']))
    report = dict(scope='Native geometric reconstruction of recorded qpos; no dynamics or contact force claim',
        model=str(path), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mujoco_version=mujoco.__version__, samples=samples, stride=a.stride,
        physics_dt=float(z['physics_dt']), categories=dict(counts), pairs=result)
    report['collision_scan_seconds'] = time.perf_counter() - start
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in report.items() if k != 'pairs'},
        'top_pairs': [{k:v for k,v in item.items() if k != 'frames'} for item in result[:12]]}, indent=2))


if __name__ == '__main__':
    main()
