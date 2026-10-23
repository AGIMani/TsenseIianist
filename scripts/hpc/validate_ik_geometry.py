#!/usr/bin/env python3
"""Check the analytic thumb separation Jacobian on recorded physical poses."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from scripts.hpc.piano_ik_geometry import hand_collision_meshes, thumb_cross_pairs, cross_clearance, landing_clearance
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recording',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):
        ap.error('Model loading and validation must run remotely')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    report=json.loads((a.recording/'replay_report.json').read_text())
    trace=a.recording/'physical_trace.npz';z=np.load(trace,allow_pickle=False)
    b=SceneBundle(Path(report['source']['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    m=b.model;d=b.reset_cpu()
    meshes=hand_collision_meshes(m);pairs=thumb_cross_pairs(m,meshes)
    ids=m.actuator_trnid[b.action_ids,0];qi=m.jnt_qposadr[ids];vi=m.jnt_dofadr[ids]
    task=json.loads((Path(report['source']['build'])/'task/task.json').read_text())
    terminal=int(m.site_bodyid[m.site(task['fingertip_sites'][5]).id])
    thumb=next(g for g in meshes if m.geom_bodyid[g]==terminal)
    key_positions=np.asarray(report['source']['key_centers'])
    key_sizes=np.asarray(report['source']['key_half_sizes'])
    errors=[];active=[];lateral_active=[];stamps=[-.5,0.,.5,16.4,5.1,23.9];eps=1e-7
    for stamp in stamps:
        negative=stamp<0
        times=z['preparation_time' if negative else 'time']
        d.qpos[:]=z['preparation_qpos' if negative else 'qpos'][np.argmin(abs(times-stamp))]
        mujoco.mj_forward(m,d)
        on=z['original_events'][(z['original_events'][:,0]<=stamp)&
                               (z['original_events'][:,1]>stamp)&(z['original_events'][:,3]==5)]
        key=int(on[0,2]) if len(on) else 57-21
        def evaluate(dofs=None):
            cross=cross_clearance(m,d,meshes,pairs,dofs=dofs)
            lateral=landing_clearance(m,d,thumb,meshes[thumb],key_positions[key],key_sizes[key],dofs=dofs)
            return np.r_[cross,lateral] if dofs is None else np.vstack((cross,lateral))
        jac=evaluate(vi)
        active.append(int((cross_clearance(m,d,meshes,pairs)<0).sum()))
        lateral_active.append(int((evaluate()[-2:]>0).sum()))
        numeric=[]
        for q in qi:
            original=d.qpos[q]
            d.qpos[q]=original+eps;mujoco.mj_forward(m,d)
            plus=evaluate()
            d.qpos[q]=original-eps;mujoco.mj_forward(m,d)
            minus=evaluate()
            numeric.append((plus-minus)/(2*eps));d.qpos[q]=original
        errors.append(float(np.max(abs(jac-np.array(numeric).T))))
    import scripts.hpc.piano_ik_geometry as geometry
    result=dict(pair_count=len(pairs),times_s=stamps,active_pair_counts=active,
                active_lateral_rows=lateral_active,
                max_jacobian_errors=errors,passed=max(errors)<1e-5 and any(active) and any(lateral_active),
                trace_sha256=sha(trace),geometry_sha256=sha(geometry.__file__),
                validation_source_sha256=sha(__file__),
                scope='Derivative check only; does not certify a collision-free trajectory')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise AssertionError('Geometry Jacobian validation failed')


if __name__=='__main__':main()
