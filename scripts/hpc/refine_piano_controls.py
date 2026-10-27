#!/usr/bin/env python3
# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""Refine an IK seed with deterministic, contact-dynamics predictive control.

Runs on the remote CPU, without a policy, reward learning, key actuators, or
physical key state assignment. The training hard_v1 robot stop is preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runner_sha256=sha(__file__)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--command-override',type=Path)
    ap.add_argument('--initial-thumb-pose-from-command',action='store_true',
                    help='Select the first commanded thumb pose at episode reset, before preparation')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=30)
    ap.add_argument('--control-hz',type=float)
    ap.add_argument('--wrist-control-hz',type=float)
    ap.add_argument('--horizon',type=int,default=3)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--control-blocks',type=int,default=1)
    ap.add_argument('--iterations',type=int,default=5)
    ap.add_argument('--no-optimize',action='store_true')
    ap.add_argument('--tip-weight',type=float,default=.15)
    ap.add_argument('--white-front-min',type=float)
    ap.add_argument('--regularization',type=float,default=.04)
    ap.add_argument('--optimize-wrist',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--active-key-weight',type=float,default=2.)
    ap.add_argument('--active-travel',type=float,default=.96)
    ap.add_argument('--frame-normalization',type=float,default=0.)
    ap.add_argument('--assigned-contact-weight',type=float,default=0.)
    ap.add_argument('--inactive-key-weight',type=float,default=1.)
    ap.add_argument('--inactive-travel',type=float,default=.4)
    ap.add_argument('--activation-objective-weight',type=float,default=0.)
    ap.add_argument('--travel-objective-weight',type=float,default=1.)
    ap.add_argument('--activation-temperature',type=float,default=.025)
    ap.add_argument('--activation-travel-margin',type=float)
    ap.add_argument('--release-window',type=float,default=0.)
    ap.add_argument('--optimize-before',type=float,default=0.)
    ap.add_argument('--optimize-start',type=float,default=0.)
    ap.add_argument('--optimize-end',type=float,default=1e10)
    ap.add_argument('--tip-z-weight',type=float,default=0.)
    ap.add_argument('--tip-depth',type=float,default=.008)
    ap.add_argument('--finger-scale',type=float,default=.3)
    ap.add_argument('--thumb-clearance-weight',type=float,default=0.)
    ap.add_argument('--thumb-cross-clearance-weight',type=float,default=0.)
    ap.add_argument('--thumb-forbidden-contact-weight',type=float,default=0.,
                    help='Penalize predicted thumb key-side/underside, proximal-key and inter-finger contacts')
    ap.add_argument('--thumb-idle-pose-weight',type=float,default=.2,
                    help='Initial-pose reference weight when a thumb has no note in the next 0.5 seconds')
    ap.add_argument('--hold-offset',action='store_true')
    a=ap.parse_args()
    if not 1<=a.control_blocks<=a.horizon:ap.error('Control blocks must be between 1 and horizon')
    if not 1<=a.workers<=8:ap.error('Use 1 to 8 CPU workers')
    if a.activation_temperature<=0 or min(a.activation_objective_weight,a.travel_objective_weight)<0:
        ap.error('Invalid optimization objective weights or activation temperature')
    if a.activation_travel_margin is not None and not 0<a.activation_travel_margin<.08:
        ap.error('Activation travel margin must lie between 0 and .08')
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):ap.error('Remote model execution required')
    torch.set_num_threads(1)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'arguments.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    seed=a.seed.resolve();z=np.load(seed/'physical_trace.npz',allow_pickle=False)
    report=json.loads((seed/'replay_report.json').read_text())
    white_front_min=(report['source'].get('white_front_min',.022)
                     if a.white_front_min is None else a.white_front_min)
    b=SceneBundle(Path(report['source']['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    m=b.model;actual=b.reset_cpu();prediction=b.reset_cpu();dt=float(m.opt.timestep)
    times=z['all_command_times'];commands=z['all_commands'];seed_cdt=1/float(report['control_hz'])
    if a.command_override:
        proposal=np.load(a.command_override,allow_pickle=False)
        if not np.array_equal(proposal['time'],times):raise ValueError('Proposal time mismatch')
        commands=proposal['commands']
        if commands.shape!=z['all_commands'].shape or not np.isfinite(commands).all():raise ValueError('Invalid command proposal')
    cdt=seed_cdt if a.control_hz is None else 1/a.control_hz
    stride=round(cdt/dt)
    if stride<1 or not np.isclose(stride*dt,cdt):raise ValueError('Control period must be an integer physics period')
    wrist_hz=a.wrist_control_hz or (report.get('wrist_control_hz',1/cdt) if a.control_hz is None else 1/cdt)
    wdt=1/wrist_hz;wstride=round(wdt/dt)
    if wstride<stride or wstride%stride or not np.isclose(wstride*dt,wdt):raise ValueError('Wrist period must be an integer multiple of control period')
    events=z['original_events'];duration=min(a.duration,float(events[:,1].max())+.25)
    if duration > float(z['time'][-1])+dt:
        raise ValueError('Seed trajectory does not cover the requested duration')
    task=json.loads((Path(report['source']['build'])/'task/task.json').read_text())
    sites=np.array([m.site(n).id for n in task['fingertip_sites']]);keygeom=np.array([m.geom(n).id for n in task['key_geoms']])
    keypos=actual.geom_xpos[keygeom].copy();keysize=m.geom_size[keygeom].copy()
    terminal={g:f for f,s in enumerate(sites) for g in range(m.ngeom) if m.geom_bodyid[g]==m.site_bodyid[s] and '_collision_' in m.geom(g).name}
    from scripts.hpc.piano_ik_geometry import (thumb_meshes,keyboard_clearance,
        hand_collision_meshes,thumb_cross_pairs,cross_clearance)
    import scripts.hpc.piano_ik_geometry as geometry_module
    geometry_source=Path(geometry_module.__file__).read_bytes()
    (out/'geometry_helper.py').write_bytes(geometry_source)
    thumb_collision_meshes=thumb_meshes(m)
    thumb_contact_rows={g:i for i,g in enumerate(thumb_collision_meshes)}
    full_hand_meshes=hand_collision_meshes(m)
    thumb_opposite_pairs=thumb_cross_pairs(m,full_hand_meshes)
    geomkeys={int(g):k for k,g in enumerate(keygeom)}
    handgeom={g:int(m.geom(g).name.startswith('left_')) for g in range(m.ngeom)
              if m.geom(g).name.startswith(('right_','left_')) and '_collision_' in m.geom(g).name}
    aids=b.action_ids;full=b.active_action_indices;jids=m.actuator_trnid[aids,0];qids=m.jnt_qposadr[jids]
    lo,hi=map(np.asarray,physical_bounds('direct_joint_v1',b.manifest['action_names'],b.low,b.high))
    rows=report['source']['joint_state_contract']['joints']
    sq=np.array([r['qpos_address'] for r in rows]);sv=np.array([r['dof_address'] for r in rows])
    sl=np.array([r['float32_min'] for r in rows]);sh=np.array([r['float32_max'] for r in rows])
    def stop(d):
        old=d.qpos[sq].copy();new=np.clip(old,sl,sh);v=d.qvel[sv]
        outward=((new<=sl)&(v<0))|((new>=sh)&(v>0));changed=(old!=new)|outward
        d.qpos[sq]=new;d.qvel[sv[outward]]=0;d.qacc_warmstart[sv[changed]]=0
    stop(actual)
    initial_thumb_pose=report['source'].get('initial_thumb_pose',{})
    if a.initial_thumb_pose_from_command:
        initial_thumb_pose={m.joint(j).name:float(commands[0,full[col]])
                            for col,j in enumerate(jids) if '_thumb_' in m.joint(j).name}
    if initial_thumb_pose:
        expected={m.joint(j).name for j in jids if '_thumb_' in m.joint(j).name}
        if set(initial_thumb_pose)!=expected:raise ValueError('Initial thumb pose must name all eight joints')
        for col,j in enumerate(jids):
            name=m.joint(j).name
            if name not in initial_thumb_pose:continue
            value=initial_thumb_pose[name]
            if not lo[full[col]]<=value<=hi[full[col]]:raise ValueError('Initial thumb pose outside physical limits')
            actual.qpos[qids[col]]=value
        stop(actual)
        mujoco.mj_forward(m,actual)
    initial_controlled_qpos=actual.qpos[qids].copy()
    channels={}
    for i,n in enumerate(b.manifest['action_names']):
        axis=n.rsplit('_',1)[-1]
        if axis in ('Tx','Ty','Pitch'):
            channels[n]=dict(index=i,min=lo[i],max=hi[i],max_velocity=.99 if axis!='Pitch' else 3.1416,
                max_acceleration=1.99 if axis!='Pitch' else 15.708,position_unit='',velocity_unit='',acceleration_unit='')
    limiter=TargetMotionLimiter(dict(control_dt=wdt,channels=channels),num_envs=1,device='cpu',reset_action=b.neutral_action,physical_low=lo,physical_high=hi)
    test_limiter=TargetMotionLimiter(dict(control_dt=wdt,channels=channels),num_envs=1,device='cpu',reset_action=b.neutral_action,physical_low=lo,physical_high=hi)
    def apply_command(command,limiter,relative_step):
        request=torch.tensor(np.clip(command,lo,hi)[None],dtype=torch.float32)
        if relative_step%wstride==0:return limiter.apply(request)[0]
        request[:,limiter.indices]=limiter.previous.float()
        return request
    worker_state=threading.local()
    worker_state.prediction=prediction;worker_state.limiter=test_limiter
    pool=ThreadPoolExecutor(max_workers=a.workers) if a.workers>1 else None
    finger_columns={f:[] for f in range(10)};wrist=[]
    for ai,j in enumerate(jids):
        name=m.joint(j).name
        if 'palm_base_' in name:wrist.append(ai)
        for f in range(10):
            side='right_' if f<5 else 'left_';finger=('thumb','index','mid','ring','pinky')[f%5]
            if name.startswith(side) and '_'+finger+'_' in name:finger_columns[f].append(ai)
    scale=np.array([.03 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else a.finger_scale for j in jids])
    idle_reference=report['source'].get('thumb_idle_reference',{})
    idle_thumb_targets={f:np.array([idle_reference[m.joint(jids[c]).name]
                                   for c in finger_columns[f]]) for f in (0,5)} if idle_reference else {}
    qlo=m.jnt_range[b.key_ids,0];qhi=m.jnt_range[b.key_ids,1]
    activation_travel=np.full_like(qhi, float(b.manifest['key_activation_fraction']))
    active_travel=(a.active_travel if a.activation_travel_margin is None
                   else activation_travel+a.activation_travel_margin)
    inactive_travel=(a.inactive_travel if a.activation_travel_margin is None
                     else activation_travel-a.activation_travel_margin)
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(m,state_spec))
    start_step=round(times[0]/dt);end_step=round(duration/dt)
    output_time=[];qs=[];vs=[];cs=[];aks=[];contacts=[];forces=[];handcontacts=[];tips=[]
    preparation_time=[];preparation_qpos=[];preparation_qvel=[];preparation_ctrl=[]
    thumb_audit_times=[];thumb_audit_contacts=[];thumb_audit_forces=[]
    thumb_key_details=[]
    thumb_geoms={g for g in handgeom if '_thumb_' in m.geom(g).name}
    command_times=[];issued_commands=[];optimization=[]
    online_tp=online_fp=online_fn=0;online_frame_sum=0.
    started=time.monotonic();offset=np.zeros(len(aids))
    for step in range(start_step,end_step+1):
        t=step*dt
        if (step-start_step)%stride==0:
            ci=min(int(np.floor((t-times[0])/seed_cdt+1e-7)),len(commands)-1)
            target=commands[ci].copy()
            if a.optimize_start-a.optimize_before<=t<a.optimize_end and not a.no_optimize:
                notes=events[(events[:,0]<t+a.horizon*cdt)&(events[:,1]>t-a.release_window)]
                fingers=set(notes[:,3].astype(int))
                if (a.thumb_clearance_weight or a.thumb_cross_clearance_weight or a.thumb_forbidden_contact_weight
                    or (a.thumb_idle_pose_weight and idle_thumb_targets)):
                    fingers.update((0,5))
                cols=sorted(set(sum([finger_columns[f] for f in fingers],[])+(wrist if a.optimize_wrist else [])))
                cols=np.array(cols,dtype=int)
                mujoco.mj_getState(m,actual,state,state_spec)
                prev=limiter.previous.clone();velocity=limiter.velocity.clone()
                schedule=[]
                for h in range(a.horizon):
                    for sub in range(stride):
                        stamp=t+h*cdt+(sub+1)*dt
                        on=events[(events[:,0]<=stamp)&(events[:,1]>stamp)]
                        desired=np.zeros(88);desired[on[:,2].astype(int)]=1
                        schedule.append((desired,on))
                def residual(x):
                    if not hasattr(worker_state,'prediction'):
                        worker_state.prediction=mujoco.MjData(m)
                        worker_state.limiter=TargetMotionLimiter(dict(control_dt=wdt,channels=channels),num_envs=1,device='cpu',reset_action=b.neutral_action,physical_low=lo,physical_high=hi)
                    prediction=worker_state.prediction;test_limiter=worker_state.limiter
                    mujoco.mj_setState(m,prediction,state,state_spec)
                    test_limiter.previous.copy_(prev);test_limiter.velocity.copy_(velocity)
                    errors=[];sample=0
                    for h in range(a.horizon):
                        future_ci=min(int(np.floor((t+h*cdt-times[0])/seed_cdt+1e-7)),len(commands)-1)
                        command=commands[future_ci].copy()
                        if h<a.control_blocks or a.hold_offset:
                            block=min(h,a.control_blocks-1)
                            command[full[cols]]+=scale[cols]*x.reshape(a.control_blocks,len(cols))[block]
                        applied=apply_command(command,test_limiter,step-start_step+h*stride)
                        prediction.ctrl[aids]=applied[0].numpy()[full]
                        for sub in range(stride):
                            mujoco.mj_step(m,prediction);stop(prediction)
                            desired,on=schedule[sample];sample+=1
                            # Keep the raw physical travel in the optimization.
                            # Clipping at the stop erases the release gradient of
                            # an accidentally held-down key. Activation remains
                            # the environment's unchanged threshold below.
                            travel=(prediction.qpos[b.key_qadr]-qlo)/(qhi-qlo)
                            # A smooth travel error supplies gradients before activation.
                            # True F1 is measured separately using the unchanged threshold.
                            key_error=np.where(desired>0,
                                a.active_key_weight*np.minimum(travel-active_travel,0),
                                a.inactive_key_weight*np.maximum(travel-inactive_travel,0))
                            normalization=np.sqrt(stride*a.horizon)*max(float(desired.sum()),1.)**a.frame_normalization
                            errors.extend(a.travel_objective_weight*key_error/normalization)
                            if a.activation_objective_weight:
                                activation=1/(1+np.exp(np.clip((activation_travel-travel)/a.activation_temperature,-40,40)))
                                errors.extend(a.activation_objective_weight*(activation-desired)/normalization)
                            if a.thumb_forbidden_contact_weight:
                                forbidden=np.zeros(len(thumb_contact_rows))
                                for contact in prediction.contact:
                                    g1,g2=int(contact.geom1),int(contact.geom2)
                                    if contact.efc_address<0 or contact.dist>1e-6:continue
                                    if g1 in geomkeys and g2 in thumb_contact_rows:
                                        thumb,normal=g2,contact.frame[:3]
                                    elif g2 in geomkeys and g1 in thumb_contact_rows:
                                        thumb,normal=g1,-contact.frame[:3]
                                    else:
                                        thumb=g1 if g1 in thumb_contact_rows else g2
                                        other=g2 if thumb==g1 else g1
                                        if thumb not in thumb_contact_rows or other not in handgeom:continue
                                        other_name=m.geom(other).name
                                        different_finger=any('_'+finger+'_' in other_name
                                                             for finger in ('index','mid','ring','pinky'))
                                        if handgeom[thumb]==handgeom[other] and not different_finger:continue
                                        normal=None
                                    if normal is not None and thumb in terminal and normal[2]>.5:continue
                                    row=thumb_contact_rows[thumb]
                                    forbidden[row]=max(forbidden[row],(.001-float(contact.dist))/.001)
                                errors.extend(a.thumb_forbidden_contact_weight*forbidden/np.sqrt(stride*a.horizon))
                            if sub==stride-1 and idle_thumb_targets and a.thumb_idle_pose_weight:
                                stamp=t+(h+1)*cdt
                                for finger,reference in idle_thumb_targets.items():
                                    upcoming=np.any((events[:,3]==finger)&(events[:,0]<=stamp+.5)&(events[:,1]>stamp))
                                    error=np.zeros(len(reference)) if upcoming else prediction.qpos[qids[finger_columns[finger]]]-reference
                                    errors.extend(a.thumb_idle_pose_weight*error/np.sqrt(a.horizon))
                            if sub==stride-1 and a.tip_weight:
                                mujoco.mj_kinematics(m,prediction)
                                tip_error=np.zeros((10,2))
                                for _,_,k,f in on:
                                    k=int(k);f=int(f);pos=prediction.site_xpos[sites[f]]
                                    tip_error[f,1]=(pos[1]-keypos[k,1])/.01
                                    xmin=keypos[k,0]-.8*keysize[k,0];xmax=keypos[k,0]+.8*keysize[k,0]
                                    if 'white' in m.geom(int(keygeom[k])).name:xmin=max(xmin,white_front_min)
                                    tip_error[f,0]=(pos[0]-np.clip(pos[0],xmin,xmax))/.01
                                errors.extend((a.tip_weight*tip_error).ravel())
                            if sub==stride-1 and a.thumb_clearance_weight:
                                mujoco.mj_kinematics(m,prediction)
                                permitted={int(m.site_bodyid[sites[int(f)]])
                                           for _,_,_,f in on if int(f) in (0,5)}
                                errors.extend(a.thumb_clearance_weight*keyboard_clearance(
                                    m,prediction,thumb_collision_meshes,keypos,keysize,
                                    permitted_terminal_bodies=permitted)/.01)
                            if sub==stride-1 and a.thumb_cross_clearance_weight:
                                mujoco.mj_kinematics(m,prediction)
                                errors.extend(a.thumb_cross_clearance_weight*cross_clearance(
                                    m,prediction,full_hand_meshes,thumb_opposite_pairs)/.01)
                            if sub==stride-1 and a.assigned_contact_weight:
                                mujoco.mj_kinematics(m,prediction)
                                distances=np.zeros(10)
                                for _,_,k,f in on:
                                    f=int(f);k=int(k)
                                    distance=.05
                                    for g,gf in terminal.items():
                                        if gf==f:distance=min(distance,mujoco.mj_geomDistance(m,prediction,g,int(keygeom[k]),.05,None))
                                    distances[f]=max(distance+.0003,0)/.01
                                errors.extend(a.assigned_contact_weight*distances)
                            if sub==stride-1 and a.tip_z_weight:
                                mujoco.mj_kinematics(m,prediction)
                                vertical=np.zeros(10)
                                for _,_,k,f in on:
                                    f=int(f);k=int(k)
                                    vertical[f]=(prediction.site_xpos[sites[f],2]-(keypos[k,2]+keysize[k,2]-a.tip_depth))/.01
                                errors.extend(a.tip_z_weight*vertical)
                    errors.extend(a.regularization*x)
                    return np.asarray(errors)
                if len(cols):
                    initial=np.tile(np.clip(offset[cols],-.999,.999),a.control_blocks)
                    # Explicit central steps avoid finite differences vanishing in
                    # the float32 command limiter near x=0.
                    def jacobian(x):
                        eps=.015
                        def column(j):
                            xp=x.copy();xm=x.copy();xp[j]+=eps;xm[j]-=eps
                            return (residual(xp)-residual(xm))/(2*eps)
                        columns=list(pool.map(column,range(len(x)))) if pool else [column(j) for j in range(len(x))]
                        return np.stack(columns,axis=1)
                    initial_error=residual(initial);initial_cost=float(initial_error@initial_error)
                    zero=np.zeros_like(initial);zero_error=residual(zero);zero_cost=float(zero_error@zero_error)
                    if zero_cost<initial_cost:initial=zero;initial_cost=zero_cost
                    solution=least_squares(residual,initial,jac=jacobian,bounds=(-np.ones(len(initial)),np.ones(len(initial))),
                        max_nfev=a.iterations,ftol=.01,xtol=.01,gtol=.01)
                    selected=solution.x if 2*solution.cost<initial_cost else initial
                    selected_blocks=selected.reshape(a.control_blocks,len(cols))
                    offset[:]=0;offset[cols]=selected_blocks[min(1,a.control_blocks-1)]
                    target[full[cols]]+=scale[cols]*selected_blocks[0]
                    optimization.append(dict(time=t,initial_cost=initial_cost,final_cost=float(2*solution.cost),nfev=solution.nfev))
            applied=apply_command(target,limiter,step-start_step)
            actual.ctrl[aids]=applied[0].numpy()[full]
            command_times.append(t);issued_commands.append(applied[0].numpy().copy())
            if abs(t-round(t))<dt/2:
                print(json.dumps(dict(time=t,elapsed=time.monotonic()-started,optimized=not a.no_optimize,
                    prefix_micro_f1=2*online_tp/max(2*online_tp+online_fp+online_fn,1),
                    prefix_mean_frame_f1=online_frame_sum/max(len(aks),1))),flush=True)
        # Include the preparation interval in the collision evidence. Musical
        # scoring below still uses only t >= 0 against the unmodified MIDI.
        mujoco.mj_forward(m,actual)
        thumb_counts=np.zeros(4,int);thumb_forces=np.zeros(4)
        key_details=np.zeros(5,int)  # top, side, underside, nonterminal, unassigned
        on_now=events[(events[:,0]<=t)&(events[:,1]>t)]
        assigned_now={(int(f),int(k)) for s,e,k,f in on_now}
        for i in range(actual.ncon):
            c=actual.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            if c.efc_address<0 or c.dist>1e-6 or not ({g1,g2}&thumb_geoms):continue
            category=None
            if g1 in geomkeys or g2 in geomkeys:
                category=3
                if g1 in geomkeys:key,thumb,normal=geomkeys[g1],g2,c.frame[:3]
                else:key,thumb,normal=geomkeys[g2],g1,-c.frame[:3]
                key_details[0 if normal[2]>.5 else 2 if normal[2]<-.5 else 1]+=1
                part=0 if m.geom(thumb).name.startswith('right_') else 5
                if thumb not in terminal:key_details[3]+=1
                if (part,key) not in assigned_now:key_details[4]+=1
            elif g1 in handgeom and g2 in handgeom:
                if handgeom[g1]!=handgeom[g2]:
                    category=0 if g1 in thumb_geoms and g2 in thumb_geoms else 1
                elif (g1 in thumb_geoms)!=(g2 in thumb_geoms):category=2
            if category is not None:
                cf=np.zeros(6);mujoco.mj_contactForce(m,actual,i,cf)
                thumb_counts[category]+=1;thumb_forces[category]+=max(float(cf[0]),0.)
        thumb_audit_times.append(t);thumb_audit_contacts.append(thumb_counts);thumb_audit_forces.append(thumb_forces)
        thumb_key_details.append(key_details)
        if step<0:
            preparation_time.append(t);preparation_qpos.append(actual.qpos.copy())
            preparation_qvel.append(actual.qvel.copy());preparation_ctrl.append(actual.ctrl.copy())
        if step>=0:
            con=np.zeros((10,88),bool);force=np.zeros((10,88));hc=np.zeros(2,int)
            for i in range(actual.ncon):
                c=actual.contact[i];g1,g2=int(c.geom1),int(c.geom2)
                if c.efc_address<0 or c.dist>1e-6:continue
                cf=np.zeros(6);mujoco.mj_contactForce(m,actual,i,cf)
                if cf[0]<=1e-8:continue
                f,k=terminal.get(g1),geomkeys.get(g2)
                if f is None or k is None:f,k=terminal.get(g2),geomkeys.get(g1)
                if f is not None and k is not None:con[f,k]=True;force[f,k]+=cf[0]
                if g1 in handgeom and g2 in handgeom:hc[int(handgeom[g1]!=handgeom[g2])]+=1
            output_time.append(t);qs.append(actual.qpos.copy());vs.append(actual.qvel.copy());cs.append(actual.ctrl.copy())
            active_now=b.key_activation(actual.qpos);desired_now=np.zeros(88,bool)
            desired_now[events[(events[:,0]<=t)&(events[:,1]>t),2].astype(int)]=True
            tp_now=int((active_now&desired_now).sum());fp_now=int((active_now&~desired_now).sum());fn_now=int((~active_now&desired_now).sum())
            online_tp+=tp_now;online_fp+=fp_now;online_fn+=fn_now
            online_frame_sum+=2*tp_now/(2*tp_now+fp_now+fn_now) if tp_now+fp_now+fn_now else 1.
            aks.append(active_now);contacts.append(con);forces.append(force);handcontacts.append(hc);tips.append(actual.site_xpos[sites].copy())
        if step<end_step:mujoco.mj_step(m,actual);stop(actual)
    if pool:pool.shutdown()
    stamps=np.array(output_time);active=np.array(aks);target=np.zeros_like(active);assigned=np.zeros((len(stamps),10,88),bool)
    for s,e,k,f in events:
        mask=(stamps>=s)&(stamps<e);target[mask,int(k)]=True;assigned[mask,int(f),int(k)]=True
    tp=int((active&target).sum());fp=int((active&~target).sum());fn=int((~active&target).sum())
    frame_den=active.sum(axis=1)+target.sum(axis=1)
    frame_f1=np.divide(2*(active&target).sum(axis=1),frame_den,
        out=np.ones(len(stamps)),where=frame_den>0)
    source=report['source'].copy();source.update(refinement_runner_sha256=runner_sha256,seed_trace_sha256=sha(seed/'physical_trace.npz'))
    source['geometry_helper_sha256']=hashlib.sha256(geometry_source).hexdigest()
    source['white_front_min']=white_front_min
    source['initial_thumb_pose']=initial_thumb_pose
    source['initial_physical_controlled_qpos']=initial_controlled_qpos.tolist()
    if a.command_override:source['command_override_sha256']=sha(a.command_override)
    final=dict(kind='deterministic_mujoco_predictive_ik_refinement',duration_s=duration,physics_hz=1/dt,control_hz=1/cdt,
        wrist_control_hz=1/wdt,
        original_midi_micro_f1=2*tp/max(2*tp+fp+fn,1),tp=tp,fp=fp,fn=fn,
        original_midi_mean_frame_f1=float(frame_f1.mean()),
        assigned_contact_recall=float((np.array(contacts)&assigned).sum()/max(assigned.sum(),1)),
        full_piece=bool(duration>=events[:,1].max()),physical_key_qpos_written_after_reset=False,
        robot_state_projection='training hard_v1 robot-only stop',source=source,pid=os.getpid(),
        no_optimization=a.no_optimize,wall_seconds=time.monotonic()-started)
    final['acceptance_passed']=(final['full_piece'] and final['original_midi_micro_f1']>=.95
        and final['original_midi_mean_frame_f1']>=.95)
    final['acceptance_criterion']='Full original piece: micro-F1 and mean physical-frame F1 both >= 0.95'
    thumb_array=np.asarray(thumb_audit_contacts)
    final['thumb_contact_audit_including_preparation']={
        'start_s':float(thumb_audit_times[0]),'end_s':float(thumb_audit_times[-1]),
        'physics_hz':1/dt,'frames':len(thumb_audit_times),
        'criterion':'Active contact constraint and distance <= 1e-6 m; no force filter',
        'frames_by_category':dict(zip(
            ['cross_hand_thumb_thumb','cross_hand_thumb_other','same_hand_thumb_other','thumb_keyboard'],
            (thumb_array>0).sum(axis=0).astype(int).tolist()))}
    final['thumb_key_contact_frames_including_preparation']=dict(zip(
        ['top','side','underside','nonterminal','unassigned_to_thumb_at_original_midi_time'],
        (np.asarray(thumb_key_details)>0).sum(axis=0).astype(int).tolist()))
    window=(stamps>=a.optimize_start)&(stamps<a.optimize_end)
    wt=int((active[window]&target[window]).sum());wf=int((active[window]&~target[window]).sum());wn=int((~active[window]&target[window]).sum())
    final['optimized_window']=dict(start=a.optimize_start,end=min(a.optimize_end,duration),
        micro_f1=2*wt/max(2*wt+wf+wn,1),mean_frame_f1=float(frame_f1[window].mean()) if window.any() else None)
    np.savez_compressed(out/'physical_trace.npz',time=stamps,qpos=qs,qvel=vs,ctrl=cs,
        actual_keys=active,target_keys=target,assignment=assigned,finger_key_contact=contacts,contact_force=forces,
        hand_contacts=handcontacts,actual_tips=tips,original_events=events,all_command_times=command_times,
        all_commands=issued_commands,key_qpos=np.array(qs)[:,b.key_qadr],key_limits=m.jnt_range[b.key_ids],
        key_activation_fraction=b.manifest['key_activation_fraction'],
        preparation_time=preparation_time,preparation_qpos=preparation_qpos,
        preparation_qvel=preparation_qvel,preparation_ctrl=preparation_ctrl,
        thumb_audit_time=thumb_audit_times,thumb_audit_contacts=thumb_audit_contacts,
        thumb_audit_force=thumb_audit_forces,thumb_key_details=thumb_key_details)
    (out/'replay_report.json').write_text(json.dumps(final,indent=2));(out/'optimization.json').write_text(json.dumps(optimization,indent=2))
    print(json.dumps({k:v for k,v in final.items() if k!='source'},indent=2),flush=True)


if __name__=='__main__':main()
