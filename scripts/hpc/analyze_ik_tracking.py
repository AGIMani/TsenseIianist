#!/usr/bin/env python3
"""Separate recorded IK alignment, wrist following and finger following errors."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runner_sha=sha(__file__)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if not os.environ.get('SSH_CONNECTION'):ap.error('Run model analysis remotely')
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    r=json.loads((a.recording/'replay_report.json').read_text())
    with np.load(a.recording/'physical_trace.npz') as archive:
        z={name:archive[name] for name in archive.files}
    b=SceneBundle(Path(r['source']['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    m=b.model;d=b.reset_cpu();source=r['source']
    task=json.loads((Path(source['build'])/'task/task.json').read_text())
    sites=np.array([m.site(n).id for n in task['fingertip_sites']])
    qids=np.array(source['controlled_qpos_addresses'])
    wrist=np.array(['forearm_' in n for n in source['joints']]);finger=~wrist
    keys=np.array(source['key_centers'])
    rows=[[] for _ in range(10)];mismatch=0.
    # Sample every control frame; scoring still comes from the original 200Hz record.
    for i in range(0,len(z['time']),10):
        t=z['time'][i]
        ci=int(np.searchsorted(z['all_command_times'],t+1e-9,side='right')-1)
        music_ci=round(t*r['control_hz'])
        if music_ci>=len(z['ik_tips']):continue
        plan=z['all_plans'][ci]
        actual=z['qpos'][i]
        d.qpos[:]=actual;d.qpos[qids]=plan;mujoco.mj_kinematics(m,d)
        planned=d.site_xpos[sites].copy()
        mismatch=max(mismatch,float(np.max(abs(planned-z['ik_tips'][music_ci]))))
        d.qpos[:]=actual;d.qpos[qids[finger]]=plan[finger];mujoco.mj_kinematics(m,d)
        perfect_fingers=d.site_xpos[sites].copy()
        d.qpos[:]=actual;d.qpos[qids[wrist]]=plan[wrist];mujoco.mj_kinematics(m,d)
        perfect_wrists=d.site_xpos[sites].copy()
        for f in range(10):
            assigned=np.flatnonzero(z['assignment'][i,f])
            if not len(assigned):continue
            k=int(assigned[0])
            if z['actual_keys'][i,k]:continue
            if abs(z['wanted_tips'][music_ci,f,1]-keys[k,1])>.002:continue
            rows[f].append([
                abs(z['actual_tips'][i,f,1]-keys[k,1]),
                abs(planned[f,1]-keys[k,1]),
                abs(perfect_fingers[f,1]-keys[k,1]),
                abs(perfect_wrists[f,1]-keys[k,1]),
                abs(z['actual_tips'][i,f,1]-perfect_fingers[f,1]),
                abs(z['actual_tips'][i,f,1]-perfect_wrists[f,1]),
            ])
    fields=['actual_target','planned_target','ideal_finger_following_target','ideal_wrist_following_target',
            'finger_following_contribution','wrist_following_contribution']
    result=dict(trace_sha256=sha(a.recording/'physical_trace.npz'),source_sha256=runner_sha,
                sample_hz=r['control_hz'],recorded_ik_fk_max_error_m=mismatch,
                scope='Missed assigned notes with matching IK target; diagnostic FK swaps, no physical replay',
                per_finger=[])
    for f,values in enumerate(rows):
        result['per_finger'].append(dict(part=f,samples=len(values),error_mm={
            name:dict(zip(('p50','p90'),np.percentile(np.array(values)[:,j]*1000,[50,90]).tolist()))
            for j,name in enumerate(fields)} if values else {}))
    result['actuator_following']=[]
    aids=np.asarray(source['controlled_actuator_ids'])
    for column,(joint,aid,qid) in enumerate(zip(source['joints'],aids,qids)):
        finger_names=('thumb','index','mid','ring','pinky')
        matches=[f for f,name in enumerate(finger_names) if '_'+name+'_' in joint]
        if not matches:continue
        f=matches[0]+(5 if joint.startswith('left_') else 0)
        mask=np.any(z['assignment'][:,f]&~z['actual_keys'],axis=1)
        if not mask.any():continue
        jid=m.actuator_trnid[aid,0];vid=m.jnt_dofadr[jid]
        error=z['ctrl'][mask,column]-z['qpos'][mask,qid]
        force=(m.actuator_gainprm[aid,0]*z['ctrl'][mask,column]+m.actuator_biasprm[aid,0]
               +m.actuator_biasprm[aid,1]*z['qpos'][mask,qid]+m.actuator_biasprm[aid,2]*z['qvel'][mask,vid])
        limits=m.actuator_forcerange[aid]
        result['actuator_following'].append(dict(joint=joint,
            absolute_target_error_rad=dict(zip(('p50','p90'),np.percentile(abs(error),[50,90]).tolist())),
            kp=float(m.actuator_gainprm[aid,0]),damping=float(-m.actuator_biasprm[aid,2]),
            force_limited=bool(m.actuator_forcelimited[aid]),force_limits=limits.tolist(),
            force_near_limit_fraction=float(((force<.95*limits[0])|(force>.95*limits[1])).mean())))
    if mismatch>1e-7:raise AssertionError('Planned FK does not match recorded IK; reject diagnosis')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
