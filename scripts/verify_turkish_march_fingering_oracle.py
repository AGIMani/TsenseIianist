#!/usr/bin/env python3
"""Verify the MIDI-finger-to-key contract with an oracle key actuator.

The oracle sets MuJoCo piano key joints to the MIDI target state at each
physical sample. This isolates the requested fingering/F1 contract from the
learned policy and fingertip IK; it is an upper-bound verification, not a
policy result.
"""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',type=Path,required=True); p.add_argument('--task',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--duration',type=float,default=10.)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(out)
    m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml')); d=mujoco.MjData(m)
    manifest=json.loads((a.task/'task.json').read_text()); score=np.load(a.task/'score.npz',allow_pickle=False)
    goal=score['goal'][:,:88].astype(bool); fingers=score['fingers'].astype(bool)
    control_dt=float(manifest['control_dt']); physics_dt=float(manifest['physics_dt']); n=int(min(round(a.duration/physics_dt),round(goal.shape[0]*control_dt/physics_dt)))
    key_names=[n.replace('_key_geom_','_joint_') for n in manifest['key_geoms']]
    key_ids=np.array([m.joint(n).id for n in key_names],dtype=int); qadr=m.jnt_qposadr[key_ids]; limits=m.jnt_range[key_ids]; threshold=float(manifest['key_tolerance'])
    actual=[]; target=[]; finger_conflicts=0; finger_missing=0
    for step in range(n):
        ci=min(int(step*physics_dt/control_dt),goal.shape[0]-1); g=goal[ci]; f=fingers[ci]
        if np.any(f.sum(axis=0)>1): finger_conflicts+=int(np.sum(f.sum(axis=0)>1))
        if np.any(g != (f.sum(axis=0)>0)): finger_missing+=int(np.sum(g != (f.sum(axis=0)>0)))
        d.qpos[qadr]=np.where(g,limits[:,1],limits[:,0]); mujoco.mj_forward(m,d)
        act=np.abs(np.clip(d.qpos[qadr],limits[:,0],limits[:,1])-limits[:,1])<=threshold
        actual.append(act); target.append(g)
    actual=np.asarray(actual); target=np.asarray(target); tp=int(np.sum(actual&target)); fp=int(np.sum(actual&~target)); fn=int(np.sum(~actual&target)); f1=2*tp/max(2*tp+fp+fn,1)
    np.savez_compressed(out/'oracle_key_trace.npz',actual=actual,target=target)
    report=dict(kind='midi_fingering_key_oracle_v1',duration_s=n*physics_dt,control_hz=1/control_dt,physics_hz=1/physics_dt,physical_frames=n,tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=f1,finger_conflicts=finger_conflicts,finger_missing=finger_missing,limitations='Oracle key actuation upper bound; not a learned policy, fingertip-contact measurement, or hardware result.')
    (out/'oracle_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
