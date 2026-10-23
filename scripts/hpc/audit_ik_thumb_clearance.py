#!/usr/bin/env python3
"""Audit actual recorded thumb/key contacts, without integrating another replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from scripts.hpc.piano_ik_geometry import thumb_meshes, lower_support


def main():
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):
        p.error('Model loading must execute remotely')
    run = a.run.resolve()
    report = json.loads((run/'replay_report.json').read_text())
    trace = run/'physical_trace.npz'
    with np.load(trace, allow_pickle=False) as archive:
        z = {name:archive[name] for name in archive.files}
    bundle = SceneBundle(Path(report['source']['build'])/'scene',
                         disable_wrist_tz=True, disable_wrist_yaw=True)
    model = bundle.model
    data = bundle.reset_cpu()
    task = json.loads((Path(report['source']['build'])/'task/task.json').read_text())
    keys = {model.geom(name).id: i for i, name in enumerate(task['key_geoms'])}
    meshes = thumb_meshes(model)
    count = dict(top=0, side=0, underside=0)
    samples = {k: set() for k in count}
    contacts = []
    lowest = {model.geom(g).name: float('inf') for g in meshes}
    for i, t in enumerate(z['time']):
        data.qpos[:] = z['qpos'][i]
        data.qvel[:] = z['qvel'][i]
        data.ctrl[:] = z['ctrl'][i]
        mujoco.mj_forward(model, data)
        if not np.array_equal(bundle.key_activation(data.qpos), z['actual_keys'][i]):
            raise ValueError('Recorded activation mismatch')
        for g, vertices in meshes.items():
            name = model.geom(g).name
            lowest[name] = min(lowest[name], float(lower_support(data, g, vertices)[2]))
        for c in data.contact:
            if c.efc_address < 0 or c.dist > 1e-6:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in keys and g2 in meshes:
                key, thumb, normal = keys[g1], g2, c.frame[:3]
            elif g2 in keys and g1 in meshes:
                key, thumb, normal = keys[g2], g1, -c.frame[:3]
            else:
                continue
            # Contact inclusion uses active constraint and nonpositive gap;
            # forces reconstructed without warmstart need not match saved forces.
            kind = 'top' if normal[2] > .5 else ('underside' if normal[2] < -.5 else 'side')
            count[kind] += 1
            samples[kind].add(i)
            contacts.append(dict(time_s=float(t), pitch=key+21, geom=model.geom(thumb).name,
                                 kind=kind, normal_z=float(normal[2]), distance_m=float(c.dist)))
    result = dict(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                  source_sha256=runner_sha,
                  physics_hz=report['physics_hz'], frames=len(z['time']),
                  contact_points=count, contact_frames={k: len(v) for k, v in samples.items()},
                  geometry_minimum_world_z_m=lowest,
                  side_or_underside_free=not samples['side'] and not samples['underside'],
                  thumb_key_contact_free=not any(samples.values()),
                  method='Recorded states, original collision model; key-to-thumb normal z >.5 top, <-.5 underside, otherwise side',
                  contacts=contacts)
    (run/'thumb_contact_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'contacts'}, indent=2))


if __name__ == '__main__':
    main()
