#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""Model-derived neutral fingertip XY fingering candidates; not dynamic reach certification."""
import argparse, itertools, json, math, hashlib, csv, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def extract(score,out):
    import numpy as np, torch, mujoco
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
    from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
    from tsensei_v3_2_pianist.fingering_targets import target_position, target_for_key
    recipe=make_recipe(world_x=.14,world_z=.06,control_hz=20,physics_hz=1000,midi=str(score.resolve()),initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',task_profile='tsensei_collision_v1',drive_profile='shadow_v1',exact_note_timing=True)
    scene,task,_=build(recipe); b=SceneBundle(scene,disable_wrist_tz=True,disable_wrist_yaw=True); d=b.reset_cpu()
    lo,hi=physical_bounds('direct_joint_v1',b.manifest['action_names'],b.low,b.high)
    opts=dict(schema=1,mode='distal_angle_range_v1',angle_ranges_deg=json.loads((ROOT/'configs/finger_coupling/angle_ranges_60_80.json').read_text()),reduce_observation=False)
    c=FingerCoupler(b,device='cpu',options=opts,physical_low=lo,physical_high=hi)
    tensor=lambda a:torch.tensor(np.asarray(a),dtype=torch.float32)[None]
    a,info=c.apply(tensor(b.neutral_action),actual_qpos=tensor(d.qpos),body_xmat=tensor(d.xmat))
    values=a[0].numpy(); qadr=b.model.jnt_qposadr[b.model.actuator_trnid[b.action_ids,0]]
    d.qpos[qadr]=values[b.active_action_indices]; mujoco.mj_forward(b.model,d)
    manifest=json.loads((task/'task.json').read_text())
    tips=np.array([d.site_xpos[b.model.site(n).id] for n in manifest['fingertip_sites']])
    targets=[]
    for f in range(10):
        targets.append([target_for_key(b.model, d, b.model.geom(n).id,thumb=f in (0,5),task_profile='tsensei_collision_v1').tolist() for n in manifest['key_geoms']])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(recipe=recipe,tip_xyz=tips.tolist(),key_target_xyz=targets,neutral_action=values.tolist(),coupling_reference=c.reference_diagnostics,limitations='Fixed compiled coupled neutral pose XY alignment only; no physical key actuation, no dynamic reach or collision guarantee.'),indent=2)+'\n')
    print('MODEL_GEOMETRY',out,flush=True)

def optimize(score,geometry,out):
    import numpy as np
    from tsensei_v3_2_pianist.music.midi_file import MidiFile,NoteTrajectory
    from generate_candidate_fingering import signature
    m=MidiFile.from_file(score); original=sorted(signature(n) for n in m.seq.notes)
    g=json.loads(geometry.read_text()); tip=np.array(g['tip_xyz']); keys=np.array(g['key_target_xyz']); notes=m.seq.notes; origin=min(n.start_time for n in notes); dt=.05
    # Normalize metres to 23.5 mm white-key widths. XY is directly model-derived.
    anchors=(keys[:,:,:2]-tip[:,None,:2])/.0235
    def states(left):
        ids=[i for i,n in enumerate(notes) if n.instrument==int(left)]
        spans={i:(int((notes[i].start_time-origin)/dt),max(int((notes[i].start_time-origin)/dt)+1,math.ceil((notes[i].end_time-origin)/dt))) for i in ids}
        frames=sorted({s for s,e in spans.values()}); layers=[]
        for frame in frames:
            active=[i for i in ids if spans[i][0]<=frame<spans[i][1]]; pitches=sorted({notes[i].pitch for i in active})
            if len(pitches)>5: raise ValueError('More than 5 simultaneous pitches')
            maps=[]; centers=[]; costs=[]
            for fs in itertools.permutations(range(5*int(left),5*int(left)+5),len(pitches)):
                by=dict(zip(pitches,fs)); aa=np.array([anchors[f,p-21] for p,f in by.items()]); center=aa.mean(0)
                maps.append({i:by[notes[i].pitch] for i in active});centers.append(center);costs.append(np.mean(np.sum((aa-center)**2,axis=1)))
            layers.append(dict(frame=frame,maps=maps,centers=np.array(centers),posture=np.array(costs)))
        return layers
    layers_by={str(h):states(h) for h in (False,True)}
    def solve(layers,w):
        total=layers[0]['posture'].copy(); back=[]
        for prev,cur in zip(layers,layers[1:]):
            elapsed=max(.25,(cur['frame']-prev['frame'])*dt)
            move=np.sum((cur['centers'][:,None]-prev['centers'][None])**2,axis=2)/elapsed
            transition=move*w+total[None,:]
            for k,cm in enumerate(cur['maps']):
                for j,pm in enumerate(prev['maps']):
                    if any(i in pm and pm[i]!=f for i,f in cm.items()):transition[k,j]=np.inf
            parent=transition.argmin(1); total=cur['posture']+transition[np.arange(len(parent)),parent]; back.append(parent)
        j=int(total.argmin()); assignment={}
        for k in range(len(layers)-1,-1,-1):
            assignment.update(layers[k]['maps'][j]); j=int(back[k-1][j]) if k else 0
        return assignment
    def metrics(assign):
        posture=movement=0.; n=0; max_res=0.;max_shift=0.
        for layers in layers_by.values():
            last=None
            for layer in layers:
                aa=np.array([anchors[assign[i],notes[i].pitch-21] for i in layer['maps'][0]])
                center=aa.mean(0);res=np.sum((aa-center)**2,axis=1); posture+=float(res.mean());max_res=max(max_res,float(res.max()));n+=1
                if last is not None:
                    dist=float(np.sum((center-last[0])**2)); movement+=dist/max(.25,(layer['frame']-last[1])*dt);max_shift=max(max_shift,math.sqrt(dist)*.0235)
                last=(center,layer['frame'])
        return dict(mean_xy_alignment_residual_mm=.0235*1000*math.sqrt(posture/n),max_xy_alignment_residual_mm=.0235*1000*math.sqrt(max_res),max_neutral_anchor_shift_m=max_shift,posture_cost=posture,movement_cost=movement,selection_cost=posture+.35*movement)
    candidates=[('baseline_heuristic',{i:n.part for i,n in enumerate(notes)})]
    for w in (.1,.35,1.):
        assignment={}
        for layers in layers_by.values():assignment.update(solve(layers,w))
        candidates.append((f'model_xy_movement_{w}',assignment))
    reports={name:metrics(a) for name,a in candidates}; chosen,assignment=min(candidates,key=lambda c:reports[c[0]]['selection_cost'])
    for i,n in enumerate(notes):n.part=assignment[i]
    assert original==sorted(signature(n) for n in notes)
    trajectory=NoteTrajectory.from_midi(m.trim_silence(),dt)
    assert all(len({n.fingering for n in chord})==len(chord) for chord in trajectory.notes)
    out.mkdir(parents=True,exist_ok=False); proto=out/'turkish-march-advanced_fingered.proto';m.save(proto)
    with (out/'fingering.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['pitch','start_seconds','end_seconds','hand','finger_number'])
        w.writerows((n.pitch,n.start_time,n.end_time,'left' if n.part>=5 else 'right',n.part%5+1) for n in notes)
    report=dict(selected=chosen,candidates=reports,note_count=len(notes),all_note_attributes_except_fingering_preserved=True,control_hz=20,physics_hz=1000,simultaneous_finger_conflicts=0,proto_sha256=hashlib.sha256(proto.read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),source_proto_sha256=hashlib.sha256(score.read_bytes()).hexdigest(),selection='Minimum posture + 0.35 movement cost among baseline and three DP candidates. Geometry-derived neutral tip XY residual, not a physical success score.',limitations='Only nominal coupled pose geometry, not joint-space reachability, force or collision validation. No guarantee of globally optimal fingering or trained performance.')
    (out/'selection_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','optimize']);p.add_argument('--score',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    extract(a.score,a.geometry) if a.mode=='extract' else optimize(a.score,a.geometry,a.output)
