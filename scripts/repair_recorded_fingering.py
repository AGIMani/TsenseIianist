#!/usr/bin/env python3
"""Offline recorded-pose fingering analysis. Run model/IK stages remotely on CPU.

Changes only NoteSequence.note.part. Static IK is an optimistic screening proxy,
not dynamics, a replacement for the follower controller, or a success prediction.
"""
import argparse
import csv
import hashlib
import importlib.util
from importlib.metadata import distribution
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def sequence_type():
    path = Path(__file__).with_name('music_pb2.py')
    if not path.exists():
        path = distribution('note-seq').locate_file('note_seq/protobuf/music_pb2.py')
    spec = importlib.util.spec_from_file_location('repair_music_pb2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NoteSequence


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def edges(notes, spans, guard=.05):
    pairs = []
    for i, j in itertools.combinations(range(len(notes)), 2):
        a, b = notes[i], notes[j]
        if a.part//5 != b.part//5 or a.pitch == b.pitch:
            continue
        grid = max(spans[i][0], spans[j][0]) < min(spans[i][1], spans[j][1])
        gap = max(a.start_time, b.start_time)-min(a.end_time, b.end_time)
        if grid or gap < guard:
            pairs.append((i, j, grid, gap))
    return pairs


def audit(notes, parts, spans, pairs, retained=None):
    conflicts, crossings, rapid = [], [], []
    for i, j, grid, gap in pairs:
        a, b = int(parts[i]), int(parts[j])
        if a == b:
            if grid:
                frames = (set(retained[i]) & set(retained[j]) if retained is not None else
                          range(max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])))
                for frame in frames:
                    conflicts.append(dict(frame=frame, finger=a, notes=[i,j]))
            rapid.append(dict(notes=[i,j], gap_s=gap, finger=a))
        direction = 1 if a < 5 else -1
        if a%5 and b%5 and (a-b)*(notes[i].pitch-notes[j].pitch)*direction < 0:
            crossings.append(dict(notes=[i,j], gap_s=gap, grid_overlap=grid))
    return dict(conflicting_finger_frames=len({(r['frame'],r['finger']) for r in conflicts}),
                conflicts=conflicts, rapid_same_finger_different_keys=rapid,
                nonthumb_order_reversals=crossings)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-ik', action='store_true')
    p.add_argument('--reach-weight', type=float, default=1.)
    a=p.parse_args(); src=a.input; out=a.output;out.mkdir(parents=True,exist_ok=True)
    Seq=sequence_type(); raw=(src/'score.proto').read_bytes(); seq=Seq.FromString(raw)
    original=Seq.FromString(raw);notes=seq.notes; parts=np.array([n.part for n in notes])
    config=json.loads((src/'config.json').read_text())
    comparison=json.loads((src/'comparison.json').read_text())
    assert sha(src/'score.proto')==comparison['original_midi_sha256']
    assert sha(src/'config.json')==comparison['config_sha256']
    assert sha(src/'score.proto')==config['source_runtime']['external_score']['sha256']
    coverage=json.loads((src/'score_coverage.json').read_text())['control_frame_coverage']
    dt=config['score_timing_contract']['control_dt']; origin=min(n.start_time for n in notes)
    spans=[(int((n.start_time-origin)*(1/dt)),max(int((n.start_time-origin)*(1/dt))+1,
             math.ceil((n.end_time-origin)*(1/dt)))) for n in notes]
    assert spans==[(n['quantized_start_frame'],n['quantized_end_frame']) for n in coverage['notes']]
    pairs=edges(notes,spans)
    z=np.load(src/'key_comparison.npz'); times=z['time']; actual=z['actual'];reference=z['original']
    offset=comparison['music_offset_seconds']; recalls=[]; rows=[]; masks=[]
    for i,n in enumerate(notes):
        start=n.start_time-origin+offset;end=n.end_time-origin+offset
        assert np.allclose(z['original_events'][i],[start,end,n.pitch],atol=1e-10,rtol=0)
        mask=(times>=start-1e-10)&(times<end-1e-10);masks.append(mask)
        hit=actual[mask,n.pitch-21];recall=float(hit.mean());recalls.append(recall)
        assert np.all(reference[mask,n.pitch-21])
        rows.append(dict(note_index=i,pitch=n.pitch,start_s=n.start_time,end_s=n.end_time,
                         old_part=n.part,active_samples=int(hit.sum()),target_samples=len(hit),
                         activation_coverage=recall,fully_missed=not bool(hit.any())))
    recalls=np.array(recalls)
    retained=[r['retained_frames'] for r in coverage['notes']]
    before=audit(notes,parts,spans,pairs,retained)
    anno=np.load(src/'score_annotations.npz')['fingers']
    assert before['conflicting_finger_frames']==int((anno.sum(-1)>1).sum())
    if not a.reuse_ik:
        import mujoco
        from scipy.optimize import least_squares
        meta=json.loads((src/'replay.json').read_text())
        assert mujoco.__version__==meta['dependency_versions']['mujoco']
        assert sha(src/'physics_trace.npz')==meta['physics_trace_sha256']
        assert meta['checkpoint_sha256']==comparison['checkpoint_sha256']
        m=mujoco.MjModel.from_binary_path(str(src/'model.mjb'));d=mujoco.MjData(m)
        trace=np.load(src/'physics_trace.npz'); q=trace['qpos'];initial=trace['initial_qpos']
        assert q.shape==(len(times),m.nq)
        sites=[m.site(f'{side}_/{f}_tip').id for side in ('right','left')
               for f in ('thumb','index','mid','ring','pinky')]
        groups=[];channels={r['full_index']:r for r in config['action_contract']['channels']}
        coupling={r['name']:r for r in config['finger_coupling_contract']['coupled_fingers']}
        for f in range(10):
            side=('right','left')[f//5];name=('thumb','index','mid','ring','pinky')[f%5]
            kinds=('rota_block_joint','rotaback_joint2','MCP_joint','PIP_joint') if f%5==0 else ('MCP_joint','swing_joint','PIP_joint','DIP_joint')
            joints=[m.joint(f'{side}_/{side}_hand_{name}_{kind}').id for kind in kinds]
            bounds=m.jnt_range[joints].copy(); cp=coupling.get(side+'_'+name)
            for k in range(4):
                full=(f//5)*25+(f%5)*4+k
                if full in channels:
                    bounds[k,0]=max(bounds[k,0],channels[full]['physical_min'])
                    bounds[k,1]=min(bounds[k,1],channels[full]['physical_max'])
                else:
                    bounds[k,0]=max(bounds[k,0],cp['follower_low'][k-2])
                    bounds[k,1]=min(bounds[k,1],cp['follower_high'][k-2])
            groups.append((m.jnt_qposadr[joints],bounds,cp))
        d.qpos[:]=initial;mujoco.mj_kinematics(m,d)
        key_geoms={int(m.geom(i).name.rsplit('_',1)[1])+21:i for i in range(m.ngeom)
                   if (m.geom(i).name or '').startswith('piano/') and '_key_geom_' in m.geom(i).name}
        targets={}
        for pitch,g in key_geoms.items():
            target=d.geom_xpos[g].copy();sz=m.geom_size[g]
            target[0]+=sz[0]-(.02 if 'black' in m.geom(g).name else .04);target[2]+=.5*sz[2]
            targets[pitch]=target
        # Recorded transforms only; no physics integration or contact solver.
        tips=np.zeros((len(q),10,3))
        for t,pose in enumerate(q):
            d.qpos[:]=pose;mujoco.mj_kinematics(m,d);tips[t]=d.site_xpos[sites]
        errors=np.zeros((len(notes),5)); angles=np.zeros_like(errors); solutions=np.zeros((len(notes),5,4))
        started=time.monotonic()
        for i,n in enumerate(notes):
            ids=np.flatnonzero(masks[i]); mid=int(ids[len(ids)//2]);base=q[mid].copy();target=targets[n.pitch]
            measured=tips[masks[i],n.part]-target
            rows[i].update(recorded_tip_target_min_mm=float(np.linalg.norm(measured,axis=1).min()*1000),
                           recorded_tip_target_median_mm=float(np.median(np.linalg.norm(measured,axis=1))*1000),
                           recorded_tip_xy_min_mm=float(np.linalg.norm(measured[:,:2],axis=1).min()*1000),
                           recorded_tip_z_min_mm=float(measured[:,2].min()*1000),ik_reference_time_s=float(times[mid]))
            for f in range(n.part//5*5,n.part//5*5+5):
                adr,bounds,cp=groups[f];lo,hi=bounds.T
                def evaluate(x, residual=True):
                    d.qpos[:]=base;d.qpos[adr]=x;mujoco.mj_kinematics(m,d)
                    err=(d.site_xpos[sites[f]]-target)*1000
                    angle=0.;violation=0.
                    if cp:
                        direction=d.xmat[cp['body_ids'][-1]].reshape(3,3)@np.array(cp['distal_direction_local'])
                        normal=np.array(config['finger_coupling_contract']['plane_normal'])
                        angle=float(np.rad2deg(np.arcsin(np.clip(-direction@normal,-1,1))))
                        low,high=config['finger_coupling_contract']['angle_ranges_deg'][cp['name']]
                        violation=max(low-angle,angle-high,0.)
                    if residual:return np.r_[err,10*violation,.02*(x-np.clip(base[adr],lo,hi))]
                    return float(np.linalg.norm(err)),angle,violation
                candidates=[]
                for seed in (base[adr],initial[adr]):
                    sol=least_squares(evaluate,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=60,
                                      ftol=1e-6,xtol=1e-6,gtol=1e-6)
                    e,ang,v=evaluate(sol.x,False);candidates.append((e*e+(10*v)**2,e,ang,sol.x))
                _,e,ang,x=min(candidates,key=lambda r:r[0]);errors[i,f%5]=e;angles[i,f%5]=ang;solutions[i,f%5]=x
            if i%20==0:print(json.dumps(dict(stage='static_ik',notes_done=i+1,total=len(notes),elapsed_s=round(time.monotonic()-started,1))),flush=True)
        np.savez_compressed(out/'ik_screening.npz',errors_mm=errors,angles_deg=angles,solutions=solutions,
                            recorded_tips=tips,time=times,targets=np.array([targets[n.pitch] for n in notes]))
        dump(out/'note_analysis.json',rows)
    else:
        rows=json.loads((out/'note_analysis.json').read_text())
    ik=np.load(out/'ik_screening.npz');errors=ik['errors_mm'];angles=ik['angles_deg']
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    rr=[];cc=[];vv=[];low=[];high=[]
    def row(cols,lo,hi):
        r=len(low);rr.extend([r]*len(cols));cc.extend(cols);vv.extend([1.]*len(cols));low.append(lo);high.append(hi)
    for i in range(len(notes)):row([i*5+f for f in range(5)],1.,1.)
    soft_reversals=[]
    for i,j,grid,gap in pairs:
        # A 50 ms release guard also avoids immediately reusing a finger on
        # another key. It does not move any source event or training target.
        for f in range(5):row([i*5+f,j*5+f],-np.inf,1.)
        direction=1 if notes[i].part<5 else -1
        for f in range(1,5):
            for g in range(1,5):
                if (f-g)*(notes[i].pitch-notes[j].pitch)*direction<0:
                    if gap < 0:
                        row([i*5+f,j*5+g],-np.inf,1.)
                    else:
                        # Released notes can cross during a transition. Penalize
                        # this risk, but do not force a distant thumb assignment.
                        aux=len(notes)*5+len(soft_reversals)
                        row([i*5+f,j*5+g,aux],-np.inf,1.)
                        vv[-1]=-1.
                        soft_reversals.append((i,j,f,g))
    nvars=len(notes)*5+len(soft_reversals)
    constraints=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(low),nvars)).tocsc(),low,high)
    edits=np.ones_like(errors);edits[np.arange(len(notes)),parts%5]=0
    violation=np.maximum(60-angles,0)+np.maximum(angles-80,0);violation[:,0]=0
    # Penalize only excess above 5 mm, since a reward landmark is not contact.
    # Successful original notes have higher edit cost; failed notes are cheaper
    # to reassign. This is an explicit planner heuristic, not measured new recall.
    reach=np.maximum(errors-5,0)**2/100+violation**2
    if not math.isfinite(a.reach_weight) or a.reach_weight<=0:raise ValueError('Invalid reach weight')
    costs=a.reach_weight*reach+edits*(.75+2.25*recalls[:,None])
    result=milp(np.r_[costs.ravel(),np.full(len(soft_reversals),.5)],integrality=np.ones(nvars),bounds=Bounds(0.,1.),
                constraints=constraints,options=dict(time_limit=180.,mip_rel_gap=0.))
    if not result.success:raise RuntimeError(result.message)
    new=parts//5*5+result.x[:len(notes)*5].reshape(-1,5).argmax(1)
    after=audit(notes,new,spans,pairs,retained)
    assert after['conflicting_finger_frames']==0 and not after['rapid_same_finger_different_keys']
    assert not any(r['gap_s']<0 for r in after['nonthumb_order_reversals'])
    for i,n in enumerate(notes):
        n.part=int(new[i]);rows[i].update(new_part=int(new[i]),changed=bool(new[i]!=parts[i]),
            old_static_error_mm=float(errors[i,parts[i]%5]),new_static_error_mm=float(errors[i,new[i]%5]),
            candidate_errors_mm=errors[i].tolist(),candidate_angles_deg=angles[i].tolist())
    check=Seq.FromString(seq.SerializeToString())
    for n,o in zip(check.notes,original.notes):n.part=o.part
    assert check==original and check.SerializeToString()==original.SerializeToString()
    assert all(n.part//5==o.part//5 for n,o in zip(seq.notes,original.notes))
    path=out/'015-5_fingered_repaired.proto';path.write_bytes(seq.SerializeToString())
    dump(out/'note_analysis.json',rows)
    with (out/'fingering.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['note_index','pitch','start_s','end_s','hand','old_finger_1_to_5','new_finger_1_to_5','part','old_replay_coverage','old_static_error_mm','new_static_error_mm'])
        writer.writerows([i,n.pitch,n.start_time,n.end_time,'right' if n.part<5 else 'left',int(parts[i]%5+1),n.part%5+1,n.part,recalls[i],errors[i,parts[i]%5],errors[i,n.part%5]] for i,n in enumerate(notes))
    report=dict(kind='recorded_pose_constrained_fingering_candidate',note_count=len(notes),changed_notes=int((new!=parts).sum()),
                changed_right=int(((new!=parts)&(parts<5)).sum()),changed_left=int(((new!=parts)&(parts>=5)).sum()),
                continuous_time_same_finger_overlaps=sum(a.part==b.part and max(a.start_time,b.start_time)<min(a.end_time,b.end_time) for a,b in itertools.combinations(original.notes,2)),
                fully_missed=int((recalls==0).sum()),below_half_coverage=int((recalls<.5).sum()),
                physical_sample_micro_f1=comparison['physical_sample_micro_f1'],before=before,after=after,
                preserved='Every protobuf field except note.part; original hands, note order, pitches, velocities and exact times preserved',
                source_score_sha256=sha(src/'score.proto'),output_score_sha256=sha(path),checkpoint_sha256=comparison['checkpoint_sha256'],
                config_sha256=sha(src/'config.json'),model_sha256=sha(src/'model.mjb'),source_script_sha256=sha(__file__),
                control_dt=dt,physics_dt=float(z['physics_dt']),trim_origin_s=origin,comparison_music_offset_s=offset,
                release_guard_s=.05,solver_status=result.message,solver_gap=float(result.mip_gap),
                reach_weight=a.reach_weight,
                selection='min sum(reach_weight*(max(error_mm-5,0)^2/100 + coupling_violation_deg^2) + changed*(0.75+2.25*old_coverage)) + 0.5*release_transition_nonthumb_reversals; unique fingers on control grid or <50ms release gap; nonthumb ordering hard only for actual simultaneous notes',
                old_mean_static_error_mm=float(errors[np.arange(len(notes)),parts%5].mean()),
                new_mean_static_error_mm=float(errors[np.arange(len(notes)),new%5].mean()),
                limitations=['Recorded 200Hz key activation identifies key success, not which finger caused it.',
                  'IK holds the old replay wrist at the middle of each note, uses existing model/action limits and 60-80deg distal-angle bounds; bounded multistart search is not proof of unreachability.',
                  'PIP/DIP are allowed to vary inside the angle band; the deployed deterministic follower solution may differ. Static screening is optimistic.',
                  'No contact forces, inter-finger collision, dynamic transition or new policy rollout was evaluated. A new score requires training/evaluation; old F1 is not a new-score result.',
                  'IK distance is to the training reward landmark, not a measured physical contact distance.'],
                new_training_or_physical_validation='not_run')
    dump(out/'revision_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','limitations')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
