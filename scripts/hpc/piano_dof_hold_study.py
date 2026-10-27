#!/usr/bin/env python3
"""Remote 3 s actuator hold diagnostics for same-score paired static candidates.

This is a prepared pose experiment, not a timed MIDI performance or training.
The actual piano state is never written after the common neutral model reset.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def hold_request(neutral,goal,t):
    alpha=np.clip((t-1.)/.5,0.,1.);alpha=alpha*alpha*(3-2*alpha)
    result=np.asarray(neutral).copy()
    fingers=list(range(20))+list(range(25,45))
    result[fingers]=neutral[fingers]+alpha*(goal[fingers]-neutral[fingers])
    result[[20,21,22,23,45,46,47,48]]=goal[[20,21,22,23,45,46,47,48]]
    return result


def hold_window(times):
    return (np.asarray(times)>2.)&(np.asarray(times)<=3.)


def run_hold(job):
    row,settings,output=job
    from scripts.hpc.piano_dof_contact_study import variant_model,sha,write
    from scripts.hpc.piano_fullscore_contact_bank import finite_top_contact
    from scripts.self_collision_metrics import classify_contact
    from tsensei_v3_2_pianist.finger_resistance import Resistance
    from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter
    from tsensei_v3_2_pianist.mjlab_env.joint_limits import resolve_robot_limits
    sm,variant,bundle=variant_model(row['variant'],settings)
    p,m,mj,torch=sm.probe,sm.m,sm.mj,sm.torch;d=bundle.reset_cpu()
    goal=np.asarray(row['action']);neutral=sm.neutral.copy()
    wanted={n['finger']:n['pitch'] for n in row['representative']['notes']}
    target=np.zeros(88,bool)
    for pitch in wanted.values():target[pitch-21]=True
    channels={}
    for i in p.full:
        name=p.names[i];axis=name.rsplit('_',1)[-1]
        if axis not in ('Tx','Ty','Tz','Pitch'):continue
        rot=axis=='Pitch';tz=axis=='Tz'
        channels[name]=dict(index=int(i),min=float(p.low[i]),max=float(p.high[i]),
            max_velocity=3.1416 if rot else .05 if tz else 1.99,
            max_acceleration=15.708 if rot else .5 if tz else 1.99,
            position_unit='rad' if rot else 'm',velocity_unit='rad/s' if rot else 'm/s',acceleration_unit='rad/s^2' if rot else 'm/s^2')
    limiter=TargetMotionLimiter(dict(control_dt=.05,channels=channels),num_envs=1,device='cpu',reset_action=neutral,physical_low=p.low,physical_high=p.high)
    keyids=np.array([m.joint(n).id for n in bundle.manifest['piano_joints']]);keyq=m.jnt_qposadr[keyids];keymax=m.jnt_range[keyids,1]
    assert len(keyids)==88 and [int(n.rsplit('_',1)[1]) for n in bundle.manifest['piano_joints']]==list(range(88))
    stops=resolve_robot_limits(m,key_ids=keyids,action_ids=np.arange(m.nu),active_indices=p.full,physical_low=p.low,physical_high=p.high)
    stopq=np.array([r['qpos_address'] for r in stops['joints']]);stopv=np.array([r['dof_address'] for r in stops['joints']])
    stoplo=np.array([r['float32_min'] for r in stops['joints']]);stophi=np.array([r['float32_max'] for r in stops['joints']]);stopcounts=np.zeros(len(stopq),int)
    assert not set(stopq)&set(keyq)
    def project():
        old=d.qpos[stopq].copy();new=np.clip(old,stoplo,stophi);vel=d.qvel[stopv]
        outward=((new<=stoplo)&(vel<0))|((new>=stophi)&(vel>0));changed=(old!=new)|outward
        d.qpos[stopq]=new;d.qvel[stopv[outward]]=0;d.qacc_warmstart[stopv[changed]]=0;stopcounts[:]+=changed
    project();resistance=Resistance(m);assert len(resistance.rows)==40
    rows=[];qtrace=[];vtrace=[];command=[];actual=[];keyangles=[];nativecounts=[];forcecounts=[];maxforce=[];maxdepth=[];toptrace=[];couplingerrors=[]
    self_rewards=[];control_pair_sum=0;command_times=[];command_coupling=[];started=time.time()
    for step in range(3000):
        t=step*.001;mj.mj_forward(m,d)
        if step%50==0:
            requested=hold_request(neutral,goal,t)
            mapped,info=p.coupler.apply(torch.tensor(np.clip(requested,p.low,p.high)[None],dtype=torch.float32),
                actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
            issued=limiter.apply(mapped)[0][0].numpy();command.append(issued.copy());command_times.append(t)
            command_coupling.append(float(info['coupling_target_error_deg'].abs().max()))
        applied,info=p.coupler.apply(torch.tensor(issued[None],dtype=torch.float32),actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
        # The coupler only changes followers; final wrist target limiter remains authoritative.
        np.testing.assert_array_equal(applied[0].numpy()[list(limiter.indices.numpy())],issued[list(limiter.indices.numpy())])
        d.ctrl[:]=applied[0].numpy()[p.full]
        resistance.update(d);mj.mj_step(m,d);project();mj.mj_forward(m,d)
        force_categories=collections.Counter();native_categories=collections.Counter();pairs=set();top=np.zeros(10,bool);peaks=collections.defaultdict(float);depths=collections.defaultdict(float)
        for ci,c in enumerate(d.contact[:d.ncon]):
            g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in sm.identities and g2 not in sm.identities:continue
            body1=m.body(int(m.geom_bodyid[g1])).name;body2=m.body(int(m.geom_bodyid[g2])).name
            classified=classify_contact(body1,body2,float(c.dist),same_weld=bool(m.body_weldid[m.geom_bodyid[g1]]==m.body_weldid[m.geom_bodyid[g2]]))
            if g1 in sm.identities and g2 in sm.identities:
                s1,f1=sm.identities[g1];s2,f2=sm.identities[g2]
                cat='cross_hand' if s1!=s2 else 'finger_palm' if f1 is None or f2 is None else 'same_finger' if f1==f2 else 'same_hand_different_fingers'
            elif g1 in sm.key_pitch or g2 in sm.key_pitch:cat='hand_key'
            else:cat='hand_piano_or_other'
            if c.dist<=0:
                native_categories[cat]+=1
                if classified['reward_counted']:pairs.add(tuple(sorted((body1,body2))))
            if c.efc_address<0 or c.dist>1e-6:continue
            force=np.zeros(6);mj.mj_contactForce(m,d,ci,force)
            if force[0]<=1e-8:continue
            force_categories[cat]+=1;peaks[cat]=max(peaks[cat],float(force[0]));depths[cat]=max(depths[cat],max(0.,-float(c.dist)))
            keyg,handg=(g1,g2) if g1 in sm.key_pitch else (g2,g1)
            if keyg in sm.key_pitch and handg in sm.identities:
                _,finger=sm.identities[handg];pitch=sm.key_pitch[keyg]
                if wanted.get(finger)==pitch:
                    site,half=sm.key_refs[pitch];rot=d.site_xmat[site].reshape(3,3);local=(c.pos-d.site_xpos[site])@rot
                    normal=c.frame[:3] if keyg==g1 else -c.frame[:3]
                    if finite_top_contact(local,half,float(normal@rot[:,2]),handg in sm.terminal[finger],float(c.dist)):top[finger]=True
        control_pair_sum+=len(pairs)
        if step%50==49:self_rewards.append(-min(5.,.1*control_pair_sum));control_pair_sum=0
        qtrace.append(d.qpos.copy());vtrace.append(d.qvel.copy());actual.append(d.qpos[keyq]>=keymax);keyangles.append(d.qpos[keyq]/keymax)
        nativecounts.append(dict(native_categories));forcecounts.append(dict(force_categories));maxforce.append(dict(peaks));maxdepth.append(dict(depths));toptrace.append(top)
        couplingerrors.append(float(info['coupling_target_error_deg'].abs().max()))
    times=(np.arange(3000)+1)*.001;actual=np.asarray(actual);toptrace=np.asarray(toptrace);hold=hold_window(times);angles=np.asarray(keyangles);cmd=np.asarray(command)
    motion={}
    for name,r in channels.items():
        velocity=np.diff(np.r_[neutral[r['index']],cmd[:,r['index']]])/.05;acceleration=np.diff(np.r_[0.,velocity])/.05
        motion[name]=dict(peak_target_velocity=float(abs(velocity).max()),peak_target_acceleration=float(abs(acceleration).max()),
            velocity_limit=r['max_velocity'],acceleration_limit=r['max_acceleration'],passed=bool((abs(velocity)<=r['max_velocity']+1e-4).all() and (abs(acceleration)<=r['max_acceleration']+1e-4).all()))
    categories=['same_hand_different_fingers','same_finger','finger_palm','cross_hand','hand_key','hand_piano_or_other']
    peaks={cat:dict(peak_force_N=max(x.get(cat,0.) for x in maxforce),peak_penetration_m=max(x.get(cat,0.) for x in maxdepth),
        force_bearing_substeps=sum(x.get(cat,0)>0 for x in forcecounts),native_substeps=sum(x.get(cat,0)>0 for x in nativecounts),
        hold_force_bearing_substeps=sum(forcecounts[i].get(cat,0)>0 for i in np.flatnonzero(hold))) for cat in categories}
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'trace.npz',time=times,qpos=qtrace,qvel=vtrace,actual_keys=actual,key_angle_fraction=angles,
        assigned_finite_top=toptrace,command_times=command_times,commands=cmd,coupling_error_deg=couplingerrors,
        self_collision_per_control_step=self_rewards,target_pitches=np.array([wanted[f] for f in sorted(wanted)]),
        target_fingers=np.array(sorted(wanted)),finger_pitch_pairs=np.array(sorted(wanted.items())))
    report=dict(kind='paired_pose_hold_physics_diagnostic_v1',source_case_signature=row['case_signature'],original_case_time=row['representative']['time_seconds'],notes=row['representative']['notes'],
        variant=variant,variant_path=row['variant'],candidate_action=goal.tolist(),candidate_static_passed=row['check']['static_passed'],
        source_fingering_sha256=settings['source_fingering_sha256'],source_scan_sha256=settings['source_scan_sha256'],
        worker_source_sha256=sha(__file__),variant_loader_sha256=sha(ROOT/'scripts/hpc/piano_dof_contact_study.py'),
        source_model_sha256=variant['model_sha256'],trace_sha256=sha(folder/'trace.npz'),pid=os.getpid(),command=sys.argv,
        physics_hz=1000,control_hz=20,physics_steps=3000,duration_s=3.,physical_key_qpos_written_after_reset=False,
        reset='Common manifest neutral state; after reset only existing hard_v1 robot joint-stop projection changes robot state; piano keys are never projected',
        recorded_state='Post hard_v1 robot-stop projection at every physics substep; not an unprojected native trajectory',
        preprojection_peak_overshoot_recorded=False,
        sequence='0..1s wrist target motion with reference fingers and MCP/SW pinky parking targets;1..1.5s smooth finger target entry;1.5..3s hold;metrics hold=(2,3]s post-step samples',
        hold_samples=int(hold.sum()),hold_correct_activation_fraction=float(actual[hold][:,target].mean()),
        hold_all_assigned_keys_activation_fraction=float(actual[hold][:,target].all(axis=1).mean()),
        hold_false_active_key_samples=int(actual[hold][:,~target].sum()),hold_finite_top_fraction=float(toptrace[hold][:,list(wanted)].mean()),
        hold_mean_key_angle_fraction={str(k):float(angles[hold,k-21].mean()) for k in sorted(wanted.values())},
        contact_categories=peaks,target_motion=motion,target_motion_passed=all(v['passed'] for v in motion.values()),
        max_command_coupling_error_deg=max(command_coupling),max_substep_coupling_error_deg=max(couplingerrors),
        total_self_collision_penalty=float(sum(self_rewards)),max_negative_self_collision_per_control_step=float(min(self_rewards)),
        self_collision_contract='Actual native contact body-pair dedup per1000Hz substep; .1 perpair; cap5 per50substep control step; separate from force-bearing diagnostic',
        robot_stop_counts=stopcounts.tolist(),robot_stop_count_by_joint={r['name']:int(n) for r,n in zip(stops['joints'],stopcounts)},
        robot_stop_contract=stops,wall_seconds=time.time()-started,
        limitations=['Hold experiment is not original MIDI timing or a whole-song F1.','TZ speed .05m/s and acceleration .5m/s² are explicit research limits, not calibrated hardware bounds.',
                    'Native full gravity compensation and fitted resistance retained; convex contact geometry is not an exact shell.','Better hold results do not certify fast entry/release or original-tempo transitions.'])
    write(folder/'report.json',report)
    return dict(path=str(folder),case=row['case_signature'],variant=row['variant'],activation=report['hold_correct_activation_fraction'],all_keys=report['hold_all_assigned_keys_activation_fraction'],motion_passed=report['target_motion_passed'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not os.environ.get('SSH_CONNECTION'):p.error('Physics is remote-only')
    if not 1<=a.workers<=8:p.error('At most eight single-thread workers')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    from scripts.hpc.piano_dof_contact_study import sha,write,CSV_SHA,MODEL_SHA,ARCHIVE_SHA
    from scripts.hpc.self_collision_probe import verify_sources
    verify_sources(a.recording.resolve(),MODEL_SHA,ARCHIVE_SHA)
    selection=json.loads(a.selection.read_text());assert selection['source_fingering_sha256']==CSV_SHA
    a.output.mkdir(parents=True,exist_ok=False)
    settings=dict(recording=str(a.recording.resolve()),geometry=str(a.geometry.resolve()),source_fingering_sha256=CSV_SHA,
                  source_scan_sha256=selection['source_scan_sha256'])
    write(a.output/'run.json',dict(command=sys.argv,pid=os.getpid(),started_unix_s=time.time(),selection_sha256=sha(a.selection),worker_source_sha256=sha(__file__)))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_hold,(r,settings,str(a.output/f'case_{i:02d}'))) for i,r in enumerate(selection['rows'])]
        for f in as_completed(futures):results.append(f.result());print(json.dumps(results[-1]),flush=True)
    write(a.output/'summary.json',dict(rows=results,finished_unix_s=time.time(),exit_status=0,training_performed=False))


if __name__=='__main__':main()
