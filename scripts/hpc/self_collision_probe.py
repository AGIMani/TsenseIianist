#!/usr/bin/env python3
"""Remote CPU static collision probe. JSONL input/output; never integrates physics.

Only a verified existing MJB is loaded. Display clients must render returned
qpos with kinematics only and label scores as hypothetical held-pose estimates.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.self_collision_metrics import (classify_contact, finger_identity, score_contacts,
                                            validate_request, validate_coupling_contract)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def verify_sources(recording, expected_model, expected_archive):
    source=json.loads((recording/'source_config.json').read_text())
    diagnostic=json.loads((recording/'model_diagnostic.json').read_text())
    validate_coupling_contract(diagnostic.get('coupling_options'))
    if diagnostic.get('archive_sha256')!=expected_archive:
        raise ValueError('Pinned archive identity differs from recording')
    if diagnostic.get('model_sha256')!=expected_model or sha(recording/'model.mjb')!=expected_model:
        raise ValueError('Pinned MJB SHA256 mismatch')
    if sha(recording/'source_config.json')!='17695e6ed5c39811049161b74d95fd60f8ceeafb1bc08edf1b604fda8596e1b8':
        raise ValueError('This probe requires the audited pinned source_config')
    changed=[]
    for relative,digest in source['source_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path)!=digest:changed.append(relative)
    if changed:raise ValueError('Source mismatch: '+', '.join(changed[:12]))
    return source,diagnostic


class Probe:
    def __init__(self, recording, expected_model, expected_archive):
        self.recording=recording
        self.source,self.diagnostic=verify_sources(recording,expected_model,expected_archive)
        import numpy as np
        import mujoco
        import torch
        from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
        from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
        self.np,self.mj,self.torch=np,mujoco,torch
        torch.set_num_threads(1)
        if mujoco.__version__!=self.source['versions']['learner']['mujoco']:
            raise ValueError('MuJoCo version differs from pinned build')
        self.model=m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'))
        manifest_path=recording/'build/scene/manifest.json'
        if sha(manifest_path)!='a7e8d0ca58c2ee2a47a9db457eaf1ad817b381567a454dd9e1d0613ead26f95a':
            raise ValueError('Audited scene manifest mismatch')
        manifest=json.loads(manifest_path.read_text())
        if m.nq!=134 or m.nu!=46 or m.nmocap!=0 or not np.isclose(m.opt.timestep,.001):
            raise ValueError('Expected audited no-arm 1000 Hz model')
        self.names=manifest['action_names']
        self.full=np.array([i for i in range(50) if i not in (22,24,47,49)])
        action_ids=np.array([m.actuator(self.names[i]).id for i in self.full])
        joints=m.actuator_trnid[action_ids,0]
        self.qadr=m.jnt_qposadr[joints]
        self.low,self.high=map(np.asarray,physical_bounds('direct_joint_v1',self.names,manifest['action_min'],manifest['action_max']))
        origin=np.zeros(m.nq)
        for name,value in zip(manifest['joint_names'],manifest['qpos']):
            jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
            if jid>=0:origin[m.jnt_qposadr[jid]]=value
        self.origin=origin
        self.neutral=np.array(manifest['neutral_action']);self.neutral[[22,47]]=0.
        def reset_cpu():
            d=mujoco.MjData(m);d.qpos[:]=origin;mujoco.mj_forward(m,d);return d
        bundle=SimpleNamespace(model=m,manifest=manifest,action_ids=action_ids,active_action_indices=self.full,
            low=np.asarray(manifest['action_min']),high=np.asarray(manifest['action_max']),
            neutral_action=self.neutral,origin_qpos=origin,
            origin_mocap_pos=np.zeros((0,3)),origin_mocap_quat=np.zeros((0,4)),reset_cpu=reset_cpu,
            key_qadr=np.array([m.jnt_qposadr[m.joint(n).id] for n in manifest['piano_joints']]))
        self.coupler=FingerCoupler(bundle,device='cpu',options=self.diagnostic['coupling_options'],
                                  physical_low=self.low,physical_high=self.high)
        followers=set(self.coupler.indices[:,1:].flatten().numpy().tolist())
        self.controls=[dict(name=self.names[i],index=int(i),min=float(self.low[i]),max=float(self.high[i]),
                            value=float(self.neutral[i]),unit='m' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else 'rad')
                       for i,j in zip(self.full,joints) if i not in followers]
        self.data=reset_cpu();self.actions=self.neutral.copy()
        self.body_names=[m.body(i).name or '' for i in range(m.nbody)]
        # Check the lightweight adapter against the exact frozen naming function.
        runtime_path=ROOT/'tsensei_v3_2_pianist/mjlab_env/self_collision.py'
        tree=ast.parse(runtime_path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finger_identity')
        scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(runtime_path),'exec'),scope)
        if any(finger_identity(n)!=scope['finger_identity'](n) for n in self.body_names):
            raise ValueError('UI identity adapter differs from pinned runtime')
        reward=json.loads((ROOT/'configs/rewards/default_20261006.json').read_text())['terms']['self_collision_reward']
        if reward!=dict(enabled=True,weight=.1,params=dict(aggregation='substep_sum',maximum_penalty_per_step=5.)):
            raise ValueError('Self-collision reward differs from audited enabled .1 / capped substep_sum contract')
        self.reward=reward
        self.provenance=dict(model_sha256=expected_model,archive_sha256=expected_archive,
            source_config_sha256=sha(recording/'source_config.json'),scene_manifest_sha256=sha(manifest_path),
            model_diagnostic_sha256=sha(recording/'model_diagnostic.json'),
            coupling_options=self.diagnostic['coupling_options'],coupling_contract_verified=True,
            source_file_count=len(self.source['source_sha256']),source_files_verified=True,
            worker_sha256=sha(__file__),metrics_helper_sha256=sha(ROOT/'scripts/self_collision_metrics.py'),
            reward_runtime_sha256=sha(runtime_path),reward_config_sha256=sha(ROOT/'configs/rewards/default_20261006.json'),
            remote_project=str(ROOT),model_path=str(recording/'model.mjb'),hostname=os.uname().nodename,pid=os.getpid(),
            mujoco_version=mujoco.__version__,physics_hz=1000.,control_hz=20.,physics_steps=0,
            reference_pose='Frozen manifest initial posture for every unedited independent control; eight finger PIP/DIP recomputed by current 10–80 degree coupling',
            limitations='Static native contact geometry, no integration or measured contact force; convex hulls can fill concavities; not hardware calibration')

    def pose(self, request):
        update=validate_request(request,self.controls)
        np,mj,torch=self.np,self.mj,self.torch;m,d=self.model,self.data
        proposed=self.neutral.copy() if request.get('reset',False) else self.actions.copy()
        for name,value in update.items():proposed[self.names.index(name)]=value
        d.qpos[:]=self.origin;d.qvel[:]=0.;d.qpos[self.qadr]=proposed[self.full]
        delta=None
        for iteration in range(8):
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            mapped,info=self.coupler.apply(torch.tensor(proposed[None],dtype=torch.float32),
                actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
            current=mapped[0].numpy().astype(float)
            delta=float(np.max(np.abs(current[self.full]-d.qpos[self.qadr])))
            d.qpos[self.qadr]=current[self.full]
            if delta<1e-7:break
        mj.mj_forward(m,d)
        self.actions=proposed
        rows=[]
        for ci in range(d.ncon):
            c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
            b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            n1,n2=self.body_names[b1],self.body_names[b2]
            if not (n1.startswith(('left_/','right_/')) or n2.startswith(('left_/','right_/'))):continue
            row=classify_contact(n1,n2,float(c.dist),same_weld=bool(m.body_weldid[b1]==m.body_weldid[b2]))
            row.update(geom1=m.geom(g1).name,geom2=m.geom(g2).name,pos=c.pos.tolist(),native_contact_index=ci)
            rows.append(row)
        return dict(type='pose',id=request.get('id'),qpos=d.qpos.tolist(),
            values={c['name']:float(proposed[c['index']]) for c in self.controls},
            applied_actions=current.tolist(),coupling={**{k:v[0].numpy().tolist() for k,v in info.items()},
                'fixed_point_iterations':iteration+1,'max_pose_change_rad':delta,'converged':bool(delta<1e-7)},
            contacts=rows,score=score_contacts(rows),physics_steps=0,static_geometry=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--expected-model-sha256',required=True)
    p.add_argument('--expected-archive-sha256',required=True);p.add_argument('--log',type=Path,required=True)
    args=p.parse_args()
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):
        p.error('Static model/FK/collision work must run remotely')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    args.log.parent.mkdir(parents=True,exist_ok=True)
    with args.log.open('x') as log:
        def record(kind,value):
            log.write(json.dumps(dict(time_unix_s=time.time(),kind=kind,payload=value),allow_nan=False)+'\n');log.flush()
        def emit(value):
            record('response',value);print(json.dumps(value,allow_nan=False),flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            probe=Probe(args.recording.resolve(),args.expected_model_sha256,args.expected_archive_sha256)
            initial=probe.pose({'id':'initial','reset':True})
        emit({**initial,**probe.provenance,'type':'ready','protocol':'static_self_collision_v1',
              'action_names':probe.names,'controls':probe.controls,'log_path':str(args.log.resolve())})
        for line in sys.stdin:
            request=None
            try:
                request=json.loads(line);record('request',request)
                with contextlib.redirect_stdout(sys.stderr):response=probe.pose(request)
                emit(response)
            except Exception as exc:
                emit(dict(type='error',id=request.get('id') if isinstance(request,dict) else None,error=str(exc)))


if __name__=='__main__':main()
