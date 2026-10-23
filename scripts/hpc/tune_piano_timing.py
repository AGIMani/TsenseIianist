#!/usr/bin/env python3
"""Seeded, derivative-free actuator timing refinement using real physics F1."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
import torch
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--population',type=int,default=4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--shift',type=float,default=.04)
    p.add_argument('--thumb-bias',action='store_true')
    p.add_argument('--method',choices=['de','coordinate'],default='de')
    a=p.parse_args()
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):p.error('Remote model execution required')
    if not 1<=a.workers<=8:p.error('Use 1 to 8 workers')
    torch.set_num_threads(1)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'arguments.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    z=np.load(a.seed/'physical_trace.npz',allow_pickle=False)
    report=json.loads((a.seed/'replay_report.json').read_text())
    b=SceneBundle(Path(report['source']['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    m=b.model;reset=b.reset_cpu();dt=float(m.opt.timestep)
    cdt=1/float(report['control_hz']);stride=round(cdt/dt)
    wdt=1/float(report.get('wrist_control_hz',report['control_hz']));wstride=round(wdt/dt)
    duration=min(a.duration,float(z['time'][-1]));last_step=round(duration/dt)
    times=z['all_command_times'];commands=z['all_commands'];first_step=round(times[0]/dt)
    events=z['original_events'];stamps=np.arange(last_step+1)*dt
    desired=np.zeros((len(stamps),88),bool)
    for s,e,k,_ in events:desired[(stamps>=s)&(stamps<e),int(k)]=True
    aids=b.action_ids;full=b.active_action_indices;jids=m.actuator_trnid[aids,0]
    lo,hi=map(np.asarray,physical_bounds('direct_joint_v1',b.manifest['action_names'],b.low,b.high))
    rows=report['source']['joint_state_contract']['joints']
    sq=np.array([r['qpos_address'] for r in rows]);sv=np.array([r['dof_address'] for r in rows])
    sl=np.array([r['float32_min'] for r in rows]);sh=np.array([r['float32_max'] for r in rows])
    def stop(d):
        old=d.qpos[sq].copy();new=np.clip(old,sl,sh);v=d.qvel[sv]
        outward=((new<=sl)&(v<0))|((new>=sh)&(v>0));changed=(old!=new)|outward
        d.qpos[sq]=new;d.qvel[sv[outward]]=0;d.qacc_warmstart[sv[changed]]=0
    stop(reset)
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(m,state_spec));mujoco.mj_getState(m,reset,state,state_spec)
    channels={}
    for i,n in enumerate(b.manifest['action_names']):
        axis=n.rsplit('_',1)[-1]
        if axis in ('Tx','Ty','Pitch'):
            channels[n]=dict(index=i,min=lo[i],max=hi[i],max_velocity=.99 if axis!='Pitch' else 3.1416,
                max_acceleration=1.99 if axis!='Pitch' else 15.708,position_unit='',velocity_unit='',acceleration_unit='')
    finger_columns={f:[] for f in range(10)}
    for ai,j in enumerate(jids):
        n=m.joint(j).name
        for f in range(10):
            side='right_' if f<5 else 'left_';finger=('thumb','index','mid','ring','pinky')[f%5]
            if n.startswith(side) and '_'+finger+'_' in n:finger_columns[f].append(full[ai])
    bias_cols=finger_columns[0] if a.thumb_bias else []
    local=threading.local();lock=threading.Lock();best={'loss':float('inf')};count=0;start=time.monotonic()
    def proposal(x):
        values=commands.copy()
        for f,cols in finger_columns.items():
            for j in cols:values[:,j]=np.interp(times+x[f],times,commands[:,j])
        if bias_cols:values[:,bias_cols]+=x[10:]
        return np.clip(values,lo,hi)
    def replay(values):
        if not hasattr(local,'data'):
            local.data=mujoco.MjData(m)
            local.limiter=TargetMotionLimiter(dict(control_dt=wdt,channels=channels),num_envs=1,device='cpu',reset_action=b.neutral_action,physical_low=lo,physical_high=hi)
        d=local.data;lim=local.limiter
        mujoco.mj_setState(m,d,state,state_spec);lim.previous.copy_(lim.reset_target[None]);lim.velocity.zero_()
        active=np.zeros_like(desired)
        for step in range(first_step,last_step+1):
            if (step-first_step)%stride==0:
                ci=min((step-first_step)//stride,len(values)-1)
                request=torch.tensor(values[ci][None],dtype=torch.float32)
                if (step-first_step)%wstride==0:applied=lim.apply(request)[0]
                else:
                    applied=request;applied[:,lim.indices]=lim.previous.float()
                d.ctrl[aids]=applied[0].numpy()[full]
            if step>=0:
                mujoco.mj_forward(m,d);active[step]=b.key_activation(d.qpos)
            if step<last_step:mujoco.mj_step(m,d);stop(d)
        return active
    zero=np.zeros(10+len(bias_cols));baseline=replay(proposal(zero))
    if not np.array_equal(baseline,z['actual_keys'][:len(baseline)]):raise ValueError('Baseline physical replay does not exactly reproduce the seed')
    def evaluate(x):
        nonlocal count
        active=replay(proposal(x));tp=int((active&desired).sum());fp=int((active&~desired).sum());fn=int((~active&desired).sum())
        micro=2*tp/max(2*tp+fp+fn,1);den=active.sum(1)+desired.sum(1)
        mean=float(np.divide(2*(active&desired).sum(1),den,out=np.ones(len(den)),where=den>0).mean())
        loss=1-.5*(micro+mean)
        with lock:
            count+=1
            if loss<best['loss']:
                best.update(loss=loss,micro_f1=micro,mean_frame_f1=mean,x=x.tolist(),evaluations=count,elapsed=time.monotonic()-start)
                (out/'best.json').write_text(json.dumps(best,indent=2));np.savez_compressed(out/'command_proposal.npz',time=times,commands=proposal(x))
                print(json.dumps(best),flush=True)
        return loss
    evaluate(zero)
    bounds=[(-a.shift,a.shift)]*10+[(-.35,.35)]*len(bias_cols)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        if a.method=='de':
            result=differential_evolution(evaluate,bounds,maxiter=a.iterations,popsize=a.population,seed=123,
                x0=zero,workers=pool.map,updating='deferred',polish=False,tol=0,atol=0)
            success=bool(result.success);message=str(result.message)
        else:
            steps=np.array([.005]*10+[.03]*len(bias_cols))
            for iteration in range(a.iterations):
                current=np.array(best['x']);previous_loss=best['loss'];candidates=[]
                for j in range(len(current)):
                    for sign in (-1,1):
                        x=current.copy();x[j]=np.clip(x[j]+sign*steps[j],*bounds[j]);candidates.append(x)
                list(pool.map(evaluate,candidates))
                if best['loss']>=previous_loss-1e-12:steps*=.5
                if steps[0]<.0001:break
            success=steps[0]<.0001;message='Coordinate search finished; no claim of global optimality'
    final=dict(best=best,evaluations=count,duration_s=duration,full_piece=bool(duration>=events[:,1].max()),
        baseline_exact_reproduction=True,runner_sha256=runner_sha,seed_trace_sha256=hashlib.sha256((a.seed/'physical_trace.npz').read_bytes()).hexdigest(),
        optimizer_success=bool(success),optimizer_message=message,wall_seconds=time.monotonic()-start)
    (out/'tuning_report.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)


if __name__=='__main__':main()
