#!/usr/bin/env python3
"""Sample actuator-command corrections through unchanged MuJoCo dynamics.

Cross-entropy trajectory search is a numerical optimizer, not policy learning.
The resulting command proposal must be independently replayed and audited.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limiter():
    return TargetMotionLimiter(dict(control_dt=CDT,channels=CHANNELS),num_envs=1,device='cpu',
        reset_action=B.neutral_action,physical_low=LOW,physical_high=HIGH)


def stop(data):
    old=data.qpos[SQ].copy();new=np.clip(old,SL,SH);v=data.qvel[SV]
    outward=((new<=SL)&(v<0))|((new>=SH)&(v>0))
    data.qpos[SQ]=new;data.qvel[SV[outward]]=0
    data.qacc_warmstart[SV[(old!=new)|outward]]=0


def apply(command,limit):
    return limit.apply(torch.tensor(np.clip(command,LOW,HIGH)[None],dtype=torch.float32))[0][0].numpy()


def activation(data):
    return np.clip(data.qpos[B.key_qadr],KEY_LOW,KEY_HIGH)>=KEY_HIGH*B.manifest['key_activation_fraction']


def init_worker():
    global PREDICTION, LIMIT
    torch.set_num_threads(1)
    PREDICTION=mujoco.MjData(M);LIMIT=limiter()


def evaluate(payload):
    x,context=payload
    state,previous,velocity,base,cols,start=context
    d=PREDICTION
    mujoco.mj_setState(M,d,state,SPEC)
    LIMIT.previous.copy_(torch.from_numpy(previous))
    LIMIT.velocity.copy_(torch.from_numpy(velocity))
    cost=0.;samples=len(base)*STRIDE
    for h,seed_command in enumerate(base):
        command=seed_command.copy();command[FULL[cols]]+=SCALE[cols]*x
        d.ctrl[AIDS]=apply(command,LIMIT)[FULL]
        for sub in range(STRIDE):
            stamp=start+h*CDT+sub*DT
            mujoco.mj_forward(M,d)
            on=EVENTS[(EVENTS[:,0]<=stamp)&(EVENTS[:,1]>stamp)]
            desired=np.zeros(88,bool);desired[on[:,2].astype(int)]=True
            active=activation(d)
            tp=np.count_nonzero(active&desired);fp=np.count_nonzero(active&~desired);fn=np.count_nonzero(~active&desired)
            denominator=2*tp+fp+fn
            cost+=0. if denominator==0 else 1-2*tp/denominator
            travel=np.clip((d.qpos[B.key_qadr]-KEY_LOW)/(KEY_HIGH-KEY_LOW),0,1)
            cost+=.05*(np.square(1-travel[desired]).sum()+np.square(np.maximum(travel[~desired]-.4,0)).sum())/max(len(on),1)
            forbidden=set()
            assigned_contact=set()
            for contact_index,c in enumerate(d.contact):
                g1,g2=int(c.geom1),int(c.geom2)
                if c.efc_address<0 or c.dist>1e-6:continue
                terminal,key=(TERMINALS.get(g1),GEOM_KEYS.get(g2))
                if terminal is None or key is None:terminal,key=TERMINALS.get(g2),GEOM_KEYS.get(g1)
                if terminal is not None and key is not None:
                    force=np.zeros(6);mujoco.mj_contactForce(M,d,contact_index,force)
                    if force[0]>1e-8:assigned_contact.add((terminal,key))
                if not ({g1,g2}&THUMBS):continue
                if g1 in GEOM_KEYS and g2 in THUMBS:
                    thumb,normal=g2,c.frame[:3]
                elif g2 in GEOM_KEYS and g1 in THUMBS:
                    thumb,normal=g1,-c.frame[:3]
                else:
                    thumb=g1 if g1 in THUMBS else g2
                    other=g2 if thumb==g1 else g1
                    if other not in HANDS:continue
                    different=any('_'+f+'_' in M.geom(other).name for f in ('index','mid','ring','pinky'))
                    if HANDS[thumb]==HANDS[other] and not different:continue
                    normal=None
                if normal is not None and thumb in TERMINALS and normal[2]>.5:continue
                forbidden.add(thumb)
            cost+=5.*len(forbidden)
            for _,_,key,finger in on:
                key,finger=int(key),int(finger)
                if not (active[key] and (finger,key) in assigned_contact):
                    cost+=(ASSIGNED_WEIGHT+(THUMB_ASSIGNED_WEIGHT if finger in (0,5) else 0.))/max(len(on),1)
            if sub==STRIDE-1:
                for _,_,key,finger in on:
                    key,finger=int(key),int(finger)
                    dy=(d.site_xpos[SITES[finger],1]-KEY_POS[key,1])/.01
                    cost+=STRIDE*.02*min(dy*dy,25)/max(len(on),1)
                    geom,vertices=TIP_MESHES[finger]
                    rotation=d.geom_xmat[geom].reshape(3,3)
                    bottom=float(np.min(vertices@rotation[2]+d.geom_xpos[geom,2]))
                    above=max(0.,bottom-(KEY_POS[key,2]+KEY_SIZE[key,2]-.008))/.01
                    cost+=STRIDE*HEIGHT_WEIGHT*min(above*above,25)/max(len(on),1)
                for f,reference in IDLE.items():
                    soon=np.any((EVENTS[:,3]==f)&(EVENTS[:,0]<=stamp+.5)&(EVENTS[:,1]>stamp))
                    if not soon:
                        cost+=STRIDE*.02*np.square(d.qpos[QIDS[FINGER_COLS[f]]]-reference).mean()
            mujoco.mj_step(M,d);stop(d)
    return cost/samples+.002*float(x@x)/max(len(x),1)


def main():
    global B,M,CDT,DT,STRIDE,CHANNELS,LOW,HIGH,SQ,SV,SL,SH,AIDS,FULL,QIDS
    global KEY_LOW,KEY_HIGH,KEY_POS,KEY_SIZE,EVENTS,SPEC,SCALE,GEOM_KEYS,HANDS,THUMBS,TERMINALS,SITES,IDLE,FINGER_COLS
    global ASSIGNED_WEIGHT,THUMB_ASSIGNED_WEIGHT,HEIGHT_WEIGHT,TIP_MESHES
    runner_sha=sha(__file__)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--horizon',type=int,default=8)
    ap.add_argument('--execute-controls',type=int,default=2)
    ap.add_argument('--population',type=int,default=64)
    ap.add_argument('--iterations',type=int,default=6)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--optimize-start',type=float,default=0.)
    ap.add_argument('--optimize-end',type=float,default=1e10)
    ap.add_argument('--optimize-wrist',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--random-seed',type=int,default=0)
    ap.add_argument('--assigned-weight',type=float,default=.5)
    ap.add_argument('--thumb-assigned-weight',type=float,default=.5)
    ap.add_argument('--tip-height-weight',type=float,default=.1)
    ap.add_argument('--reference',type=Path,action='append',default=[])
    a=ap.parse_args()
    if not os.environ.get('SSH_CONNECTION'):ap.error('Remote CPU execution required')
    if not 1<=a.workers<=16 or a.population<8 or not 1<=a.execute_controls<=a.horizon:
        ap.error('Invalid resource/search settings')
    if not 1<=a.iterations<=32 or not np.isfinite([a.assigned_weight,a.thumb_assigned_weight,a.tip_height_weight]).all() or min(a.assigned_weight,a.thumb_assigned_weight,a.tip_height_weight)<0:
        ap.error('Invalid optimization weights or iterations')
    torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'arguments.json').write_text(json.dumps(vars(a),default=str,indent=2))
    report=json.loads((a.seed/'replay_report.json').read_text())
    with np.load(a.seed/'physical_trace.npz') as archive:
        z={name:archive[name] for name in archive.files}
    source=report['source'];B=SceneBundle(Path(source['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    M=B.model;actual=B.reset_cpu();AIDS=B.action_ids;FULL=B.active_action_indices
    joints=M.actuator_trnid[AIDS,0];QIDS=M.jnt_qposadr[joints]
    LOW,HIGH=map(np.asarray,physical_bounds('direct_joint_v1',B.manifest['action_names'],B.low,B.high))
    stops=source['joint_state_contract']['joints']
    SQ=np.array([r['qpos_address'] for r in stops]);SV=np.array([r['dof_address'] for r in stops])
    SL=np.array([r['float32_min'] for r in stops]);SH=np.array([r['float32_max'] for r in stops])
    stop(actual)
    for column,j in enumerate(joints):
        name=M.joint(j).name
        if name in source.get('initial_thumb_pose',{}):actual.qpos[QIDS[column]]=source['initial_thumb_pose'][name]
    stop(actual);mujoco.mj_forward(M,actual)
    CDT=1/report['control_hz'];DT=float(M.opt.timestep);STRIDE=round(CDT/DT)
    if not np.isclose(STRIDE*DT,CDT) or report.get('wrist_control_hz',1/CDT)!=1/CDT:
        raise ValueError('Search requires the same wrist/finger control rate')
    CHANNELS={}
    for i,name in enumerate(B.manifest['action_names']):
        axis=name.rsplit('_',1)[-1]
        if axis in ('Tx','Ty','Pitch'):
            CHANNELS[name]=dict(index=i,min=LOW[i],max=HIGH[i],max_velocity=.99 if axis!='Pitch' else 3.1416,
                max_acceleration=1.99 if axis!='Pitch' else 15.708,position_unit='',velocity_unit='',acceleration_unit='')
    current_limit=limiter()
    KEY_LOW=M.jnt_range[B.key_ids,0];KEY_HIGH=M.jnt_range[B.key_ids,1]
    task=json.loads((Path(source['build'])/'task/task.json').read_text())
    SITES=np.array([M.site(n).id for n in task['fingertip_sites']])
    GEOM_KEYS={M.geom(n).id:k for k,n in enumerate(task['key_geoms'])}
    KEY_POS=np.array(source['key_centers']);KEY_SIZE=np.array(source['key_half_sizes']);EVENTS=z['original_events']
    ASSIGNED_WEIGHT=a.assigned_weight;THUMB_ASSIGNED_WEIGHT=a.thumb_assigned_weight;HEIGHT_WEIGHT=a.tip_height_weight
    HANDS={g:int(M.geom(g).name.startswith('left_')) for g in range(M.ngeom)
           if M.geom(g).name.startswith(('right_','left_')) and '_collision_' in M.geom(g).name}
    THUMBS={g for g in HANDS if '_thumb_' in M.geom(g).name}
    TERMINALS={g:f for g in HANDS for f,s in enumerate(SITES) if M.geom_bodyid[g]==M.site_bodyid[s]}
    TIP_MESHES={}
    for g,f in TERMINALS.items():
        if M.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Expected actual terminal collision mesh')
        mesh=int(M.geom_dataid[g]);start=M.mesh_vertadr[mesh]
        TIP_MESHES[f]=(g,M.mesh_vert[start:start+M.mesh_vertnum[mesh]].copy())
    FINGER_COLS={f:[] for f in range(10)};wrists=[]
    for i,j in enumerate(joints):
        name=M.joint(j).name
        if 'forearm_' in name:wrists.append(i)
        for f in range(10):
            side='right_' if f<5 else 'left_'
            if name.startswith(side) and '_'+('thumb','index','mid','ring','pinky')[f%5]+'_' in name:FINGER_COLS[f].append(i)
    IDLE={f:np.array([source['thumb_idle_reference'][M.joint(joints[c]).name] for c in FINGER_COLS[f]])
          for f in (0,5)} if source.get('thumb_idle_reference') else {}
    SCALE=np.array([.035 if M.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else .3 if 'forearm_' in M.joint(j).name else 1. for j in joints])
    SPEC=mujoco.mjtState.mjSTATE_INTEGRATION
    rng=np.random.default_rng(a.random_seed);commands=z['all_commands'];times=z['all_command_times']
    references=[]
    for path in a.reference:
        ref_report=json.loads((path/'replay_report.json').read_text())
        if ref_report['source']['joints']!=source['joints']:raise ValueError('Reference joint mapping differs')
        with np.load(path/'physical_trace.npz') as ref:
            if ref['all_commands'].shape!=commands.shape or not np.array_equal(ref['all_command_times'],times) or not np.array_equal(ref['original_events'][:,:3],EVENTS[:,:3]):
                raise ValueError('Reference command times or original MIDI do not match')
            references.append(ref['all_commands'].copy())
    if len(references)>a.population-4:raise ValueError('Too many reference candidates')
    generated=[];record_times=[];qs=[];vs=[];ctrl=[];keys=[];history=[]
    offset=np.zeros(len(AIDS));last_search=-a.execute_controls
    started=time.monotonic();end=float(z['time'][-1])
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=init_worker) as pool:
        for ci,t in enumerate(times):
            target=commands[ci].copy()
            if a.optimize_start<=t<a.optimize_end:
                if ci-last_search>=a.execute_controls:
                    notes=EVENTS[(EVENTS[:,0]<t+a.horizon*CDT)&(EVENTS[:,1]>t)]
                    fingers=set(notes[:,3].astype(int))|{0,5}
                    cols=np.array(sorted(set(sum([FINGER_COLS[f] for f in fingers],[])+(wrists if a.optimize_wrist else []))))
                    state=np.empty(mujoco.mj_stateSize(M,SPEC));mujoco.mj_getState(M,actual,state,SPEC)
                    block=commands[np.minimum(np.arange(ci,ci+a.horizon),len(commands)-1)]
                    context=(state,current_limit.previous.numpy().copy(),current_limit.velocity.numpy().copy(),block,cols,float(t))
                    mean=np.clip(offset[cols],-.9,.9);std=np.full(len(cols),.35)
                    prior_end=min(ci+a.execute_controls,len(commands))
                    priors=[np.clip(((ref[ci:prior_end,FULL[cols]]-commands[ci:prior_end,FULL[cols]])/SCALE[cols]).mean(0),-1,1)
                            for ref in references]
                    best=np.zeros(len(cols));best_cost=float('inf');baseline=None
                    for iteration in range(a.iterations):
                        candidates=np.clip(mean+rng.normal(size=(a.population,len(cols)))*std,-1,1)
                        candidates[0]=best;candidates[1]=mean;candidates[2]=0
                        for i,prior in enumerate(priors):candidates[i+3]=prior
                        costs=np.array(list(pool.map(evaluate,[(x,context) for x in candidates],chunksize=1)))
                        if iteration==0:baseline=float(costs[2])
                        order=np.argsort(costs);elite=candidates[order[:max(4,a.population//8)]]
                        if costs[order[0]]<best_cost:best_cost=float(costs[order[0]]);best=candidates[order[0]].copy()
                        mean=.7*elite.mean(axis=0)+.3*mean
                        std=np.maximum(.7*elite.std(axis=0)+.3*std,.03)
                    offset[:]=0;offset[cols]=best;last_search=ci
                    history.append(dict(time=float(t),baseline_cost=baseline,cost=best_cost,dimensions=len(cols)))
                    print(json.dumps(dict(time=float(t),elapsed=time.monotonic()-started,baseline_cost=baseline,cost=best_cost)),flush=True)
                target[FULL]+=SCALE*offset
            applied=apply(target,current_limit);actual.ctrl[AIDS]=applied[FULL];generated.append(applied.copy())
            for sub in range(STRIDE):
                stamp=float(t+sub*DT)
                if stamp>end+1e-8:break
                mujoco.mj_forward(M,actual)
                record_times.append(stamp);qs.append(actual.qpos.copy());vs.append(actual.qvel.copy())
                ctrl.append(actual.ctrl.copy());keys.append(activation(actual))
                if stamp<end-1e-8:mujoco.mj_step(M,actual);stop(actual)
    np.savez_compressed(a.output/'commands.npz',time=times,commands=generated)
    np.savez_compressed(a.output/'search_trace.npz',time=record_times,qpos=qs,qvel=vs,ctrl=ctrl,actual_keys=keys)
    (a.output/'optimization.json').write_text(json.dumps(history,indent=2))
    (a.output/'source.json').write_text(json.dumps(dict(source_sha256=runner_sha,seed_trace_sha256=sha(a.seed/'physical_trace.npz'),
        reference_trace_sha256={str(path):sha(path/'physical_trace.npz') for path in a.reference},
        control_hz=1/CDT,physics_hz=1/DT,physical_key_states_written=False,wall_seconds=time.monotonic()-started,
        validation='Independent full replay required; no learning or model/actuator modification'),indent=2))


if __name__=='__main__':main()
