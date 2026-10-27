#!/usr/bin/env python3
"""Remote static full-score contact candidates; cache never certifies dynamics."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def onset_nodes(notes):
    if len({n['index'] for n in notes})!=len(notes):raise ValueError('Duplicate note index')
    for n in notes:
        if (not all(math.isfinite(n[k]) for k in ('start','end')) or not 0<=n['start']<n['end']
                or not 0<=n['finger']<10 or not 21<=n['pitch']<=108):
            raise ValueError('Invalid piano note')
    result=[]
    for at in sorted({n['start'] for n in notes}):
        active=sorted((n for n in notes if n['start']<=at<n['end']),key=lambda n:(n['finger'],n['pitch']))
        if len({n['finger'] for n in active})!=len(active):raise ValueError('Finger overlap at onset')
        result.append(dict(time_seconds=at,notes=active,
            held_note_indices=sorted(n['index'] for n in active if n['start']<at),
            onset_note_indices=sorted(n['index'] for n in active if n['start']==at)))
    return result


def static_signature(node,contract):
    return digest(dict(contract=contract,finger_pitch=sorted((n['finger'],n['pitch']) for n in node['notes'])))


def cached_action(row,signature,contract):
    if (row.get('kind')!='static_candidate_v1' or row.get('signature')!=signature
            or row.get('contract_sha256')!=digest(contract)):
        raise ValueError('Static candidate source/signature mismatch')
    action=row.get('action',[])
    if len(action)!=51 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in action):
        raise ValueError('Invalid finite static action seed')
    return action.copy()


def finite_top_contact(local,half,alignment,is_terminal,distance):
    return bool(is_terminal and distance<=0 and alignment>=.7 and abs(local[2]-half[2])<=.002
                and abs(local[0])<=half[0]-.005 and abs(local[1])<=half[1]-.0005)


def occurrence_record(node,candidate,check):
    if check is None:raise ValueError('Each occurrence requires independent static revalidation')
    return dict(**node,signature=candidate['signature'],action=candidate['action'],
                **check,revalidated=True,physics_steps=0,dynamic_acceptance=None)


class StaticModel:
    """One remote worker's scratch data, sharing the verified binary contract."""
    def __init__(self,settings):
        from scripts.hpc.self_collision_probe import Probe,sha
        from scripts.plan_model_fingering import Geometry
        from tsensei_v3_2_pianist.fingering_targets import key_reference
        self.settings=settings
        self.probe=p=Probe(Path(settings['recording']),settings['model_sha256'],settings['archive_sha256'])
        self.np,self.mj,self.torch=p.np,p.mj,p.torch
        self.m,self.d=p.model,p.data
        self.source=json.loads(Path(settings['geometry']).read_text())
        if self.source['source_archive_sha256']!=settings['archive_sha256']:
            raise ValueError('Geometry seed archive differs from the pinned source')
        self.geometry=Geometry(self.source)
        self.neutral=p.neutral.copy()
        for off in (16,41):self.neutral[off:off+4]=[.490541428,.01000039,.389615625,.213344008]
        self.independent=[c['index'] for c in p.controls]
        self.keys={};self.terminal={};self.vertices={};self.hand=[];self.identities={}
        from scripts.self_collision_metrics import finger_identity
        digits=['thumb','index','mid','ring','pinky']
        for g in range(self.m.ngeom):
            name=self.m.geom(g).name or '';body=self.m.body(int(self.m.geom_bodyid[g])).name or ''
            if name.startswith(('piano/white_key_geom_','piano/black_key_geom_')):
                self.keys[int(name.rsplit('_',1)[1])+21]=g
            if name.startswith(('left_/','right_/')) and '_collision_' in name:
                self.hand.append(g);identity=finger_identity(body)
                f=None if identity is None else (5 if identity[0]=='left' else 0)+digits.index(identity[1])
                self.identities[g]=(0 if name.startswith('right_/') else 1,f)
                if f is not None and (('_thumb_PIP_link_' in name) if f%5==0 else '_DIP_link_' in name):
                    self.terminal.setdefault(f,[]).append(g)
                    mesh=int(self.m.geom_dataid[g]);a=int(self.m.mesh_vertadr[mesh]);n=int(self.m.mesh_vertnum[mesh])
                    self.vertices[g]=self.m.mesh_vert[a:a+n].copy()
        if len(self.keys)!=88 or len(self.terminal)!=10:raise ValueError('Incomplete native geometry mapping')
        self.key_pitch={g:pitch for pitch,g in self.keys.items()}
        self.key_refs={pitch:key_reference(self.m,g) for pitch,g in self.keys.items()}
        self.hand_rows={g:i for i,g in enumerate(self.hand)}
        self.extra=[]
        for path in settings['seed_banks']:
            report=json.loads(Path(path).read_text())
            self.extra.extend(report.get('poses',[]))
        self.provenance={**p.provenance,'geometry_seed_sha256':sha(settings['geometry']),
            'geometry_seed_model_sha256':self.source.get('model_sha256'),
            'geometry_seed_role':'Initialization only; every candidate rebuilt and checked in pinned MJB',
            'bank_worker_sha256':sha(__file__),'physics_steps':0}

    def seed(self,node):
        result=self.neutral.copy()
        for side,off in ((0,0),(1,25)):
            group=sorted((n for n in node['notes'] if n['finger']//5==side),key=lambda n:n['pitch'])
            if not group:continue
            pose=min(self.geometry.candidates(tuple(n['finger'] for n in group),tuple(n['pitch'] for n in group),left=bool(side)),
                     key=lambda p:max(p.residuals_m))
            result[off+20:off+22]=pose.wrist_tx_ty;result[off+23]=pose.wrist_pitch
            for n,sample in zip(group,pose.sample_indices):
                ix=off+n['finger']%5*4
                result[ix:ix+4]=self.source['reachable_samples'][n['finger']][sample]['action'][ix:ix+4]
        return self.np.clip(result,self.probe.low,self.probe.high)

    def set_pose(self,action,node):
        p,m,d,np,mj,torch=self.probe,self.m,self.d,self.np,self.mj,self.torch
        requested=np.clip(action,p.low,p.high).copy();requested[[22,24,47,49]]=self.neutral[[22,24,47,49]]
        d.qpos[:]=p.origin;d.qvel[:]=0.;d.qpos[p.qadr]=requested[p.full]
        for n in node['notes']:
            j=m.joint('piano/'+('white' if 'white' in m.geom(self.keys[n['pitch']]).name else 'black')+'_joint_'+str(n['pitch']-21)).id
            d.qpos[m.jnt_qposadr[j]]=m.jnt_range[j,1]
        delta=float('inf')
        for iteration in range(8):
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            mapped,info=p.coupler.apply(torch.tensor(requested[None],dtype=torch.float32),
                actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
            applied=mapped[0].numpy().astype(float)
            delta=float(np.max(abs(applied[p.full]-d.qpos[p.qadr])))
            d.qpos[p.qadr]=applied[p.full]
            if delta<1e-7:break
        mj.mj_fwdPosition(m,d)
        error=float(info['coupling_target_error_deg'].abs().max())
        return applied,dict(coupling_converged=delta<1e-7,coupling_iterations=iteration+1,
                            coupling_error_max_deg=error,coupling_pose_delta_rad=delta)

    def inspect(self,action,node,detail=False):
        from scripts.hpc.piano_contact_planner import terminal_top_error
        np,m,d=self.np,self.m,self.d
        applied,coupling=self.set_pose(action,node)
        wanted={n['finger']:n['pitch'] for n in node['notes']}
        plane=[];lowest=[];top={f:False for f in wanted};penalty=np.zeros(len(self.hand))
        contacts=[];forbidden=0;max_depth=0.;target_overdepth=0.
        for n in node['notes']:
            sid,half=self.key_refs[n['pitch']]
            verts=np.concatenate([self.vertices[g]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g in self.terminal[n['finger']]])
            err,local=terminal_top_error(verts,d.site_xpos[sid],d.site_xmat[sid].reshape(3,3),half,
                                       vertical_offset=-self.settings['penetration'])
            plane.extend(err);lowest.append(local.tolist())
        for c in d.contact[:d.ncon]:
            if c.dist>0:continue
            g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in self.identities and g2 not in self.identities:continue
            valid=False;category='other';local=None;alignment=None
            keyg,handg=(g1,g2) if g1 in self.key_pitch else (g2,g1)
            if keyg in self.key_pitch and handg in self.identities:
                _,finger=self.identities[handg];pitch=self.key_pitch[keyg]
                if wanted.get(finger)==pitch:
                    sid,half=self.key_refs[pitch];rot=d.site_xmat[sid].reshape(3,3)
                    local=(c.pos-d.site_xpos[sid])@rot
                    normal=c.frame[:3] if keyg==g1 else -c.frame[:3]
                    alignment=float(normal@rot[:,2])
                    valid=finite_top_contact(local,half,alignment,handg in self.terminal[finger],float(c.dist))
                    if valid:top[finger]=True
                    category='assigned_terminal_top' if valid else 'assigned_key_side_or_nonterminal'
                else:category='non_target_key'
            elif g1 in self.identities and g2 in self.identities:
                s1,f1=self.identities[g1];s2,f2=self.identities[g2]
                category='cross_hand' if s1!=s2 else 'finger_palm' if f1 is None or f2 is None else 'same_finger' if f1==f2 else 'same_hand_different_fingers'
            depth=max(0.,-float(c.dist))
            deficit=max(0.,depth-.001) if valid else depth+.001
            if not valid:forbidden+=1;max_depth=max(max_depth,depth)
            else:target_overdepth=max(target_overdepth,deficit)
            for geom in (g1,g2):
                if geom in self.hand_rows:penalty[self.hand_rows[geom]]=max(penalty[self.hand_rows[geom]],deficit)
            if detail:contacts.append(dict(geom1=m.geom(g1).name,geom2=m.geom(g2).name,
                category=category,distance_m=float(c.dist),pos=c.pos.tolist(),finite_top=valid,
                key_local=None if local is None else local.tolist(),normal_alignment=alignment))
        passed=all(top.values()) and forbidden==0 and target_overdepth==0 and coupling['coupling_converged'] and coupling['coupling_error_max_deg']<=.05
        check=dict(**coupling,static_passed=passed,native_assigned_top_by_finger=top,
            assigned_top_count=sum(top.values()),assigned_count=len(top),unwanted_native_contacts=forbidden,
            max_unwanted_penetration_m=max_depth,max_target_excess_over_1mm_m=target_overdepth,
            terminal_lowest_key_local=lowest,plane_residual_m=np.asarray(plane).reshape(-1,3).tolist())
        if detail:check['contacts']=contacts
        return applied,check,np.r_[4*np.asarray(plane),10*penalty,.001*max(0.,coupling['coupling_error_max_deg']-.05)]

    def solve(self,node,signature,contract,seed_action=None):
        from scipy.optimize import least_squares
        np=self.np;start=time.process_time();deadline=start+self.settings['node_cpu_budget']
        base=self.seed(node);seeds=[base]
        if seed_action is not None:seeds.insert(0,np.asarray(seed_action))
        template=base.copy();found=False
        for side in (0,1):
            group=sorted((n['finger'],n['pitch']) for n in node['notes'] if n['finger']//5==side)
            for old in self.extra:
                if group and group==sorted((n['finger'],n['pitch']) for n in old['notes'] if n['finger']//5==side):
                    template[side*25:(side+1)*25]=old['action'][side*25:(side+1)*25];found=True;break
        if found:seeds.insert(0,template)
        fingers={n['finger'] for n in node['notes']};sides={f//5 for f in fingers}
        variables=[i for i in self.independent if (i//25 in sides and i%25>=20) or (i//25*5+(i%25)//4 in fingers and i%25<20)]
        lo,hi=self.probe.low[variables],self.probe.high[variables]
        best=None;evals=0;statuses=[]
        class Budget(Exception):pass
        def evaluate(x,seed,allow_expired=False):
            nonlocal best,evals
            if not allow_expired and time.process_time()>=deadline:raise Budget()
            action=seed.copy();action[variables]=x
            applied,check,residual=self.inspect(action,node)
            residual=np.r_[residual,.0001*(x-seed[variables])]
            score=float(residual@residual);evals+=1
            rank=(not check['static_passed'],score)
            if best is None or rank<best[0]:best=(rank,applied.copy(),check)
            return residual
        # One bounded evaluation ensures a diagnostic candidate even on budget expiry.
        evaluate(np.clip(seeds[0][variables],lo,hi),seeds[0],True)
        for seed in seeds:
            if time.process_time()>=deadline:break
            def objective(x):return evaluate(x,seed)
            def jacobian(x):
                baseline=objective(x);cols=[]
                for j,index in enumerate(variables):
                    step=1e-5 if index%25 in (20,21) else 3e-4
                    delta=min(step,hi[j]-x[j])
                    if delta<step/2:delta=max(-step,lo[j]-x[j])
                    moved=x.copy();moved[j]+=delta
                    cols.append((objective(moved)-baseline)/delta)
                return np.stack(cols,axis=1)
            try:
                fit=least_squares(objective,np.clip(seed[variables],lo+1e-8,hi-1e-8),jac=jacobian,bounds=(lo,hi),
                    max_nfev=self.settings['max_nfev'],ftol=1e-6,xtol=1e-6,gtol=1e-7,x_scale='jac')
                statuses.append(int(fit.status))
            except Budget:statuses.append('cpu_budget');break
        action,check,_=self.inspect(best[1],node,True)
        return dict(kind='static_candidate_v1',signature=signature,contract_sha256=digest(contract),
            action=action.tolist(),check=check,representative=node,objective=best[0][1],
            solver_cpu_seconds=time.process_time()-start,solver_statuses=statuses,evaluations=evals,
            worker_pid=os.getpid(),physics_steps=0,dynamic_acceptance=None)


WORKER=None


def initialize_worker(settings):
    global WORKER
    log=Path(settings['output'])/f'worker_{os.getpid()}.stderr'
    stream=log.open('w');sys.stderr=stream
    with contextlib.redirect_stdout(stream):WORKER=StaticModel(settings)
    (Path(settings['output'])/f'worker_{os.getpid()}_source.json').write_text(json.dumps(WORKER.provenance,indent=2))


def worker_group(job):
    signature,nodes,contract,seed_action,cache_candidate=job
    start=time.time()
    if cache_candidate is None:
        candidate=WORKER.solve(nodes[0],signature,contract,seed_action)
    else:
        action=cached_action(cache_candidate,signature,contract)
        applied,check,_=WORKER.inspect(WORKER.np.asarray(action),nodes[0],True)
        candidate={**cache_candidate,'action':applied.tolist(),'check':check,'reused_cache_candidate':True,
                   'worker_pid':os.getpid(),'solver_cpu_seconds':0.}
    records=[]
    for node in nodes:
        applied,check,_=WORKER.inspect(WORKER.np.asarray(candidate['action']),node,True)
        records.append(occurrence_record(node,{**candidate,'action':applied.tolist()},check))
    return dict(candidate=candidate,occurrences=records,wall_seconds=time.time()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fingering',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-model-sha256',required=True);p.add_argument('--expected-archive-sha256',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--node-cpu-budget',type=float,default=5.)
    p.add_argument('--max-nfev',type=int,default=60);p.add_argument('--penetration',type=float,default=.0005)
    p.add_argument('--seed-bank',type=Path,action='append',default=[])
    p.add_argument('--cache',type=Path,action='append',default=[])
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--max-groups',type=int)
    a=p.parse_args()
    if not os.environ.get('SSH_CONNECTION'):p.error('All model and collision work must run remotely')
    if not 1<=a.workers<=8 or not 0<a.node_cpu_budget<=30 or not .0002<=a.penetration<=.001:
        p.error('Unsupported resource or contact target budget')
    os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    from scripts.hpc.self_collision_probe import sha,verify_sources
    source,diagnostic=verify_sources(a.recording.resolve(),a.expected_model_sha256,a.expected_archive_sha256)
    contract=dict(schema='fullscore_static_native_top_v1',model_sha256=a.expected_model_sha256,
        archive_sha256=a.expected_archive_sha256,geometry_sha256=sha(a.geometry),coupling=diagnostic['coupling_options'],
        target_penetration_m=a.penetration,active_axes='Pinned MJB no TZ/Yaw; current physical bounds',
        idle_policy='Inactive independent fingers fixed to neutral; pinky MCP/SW parked; followers recomputed',
        top_contract='native terminal/key pair; normal>=.7; X margin5mm Ymargin.5mm Ztol2mm; all unwanted native contacts fail',
        bank_source_sha256=sha(__file__),contact_helper_sha256=sha(ROOT/'scripts/hpc/piano_contact_planner.py'),
        geometry_seed_code_sha256=sha(ROOT/'scripts/plan_model_fingering.py'),
        metrics_helper_sha256=sha(ROOT/'scripts/self_collision_metrics.py'),probe_sha256=sha(ROOT/'scripts/hpc/self_collision_probe.py'))
    rows=[dict(index=int(r['note_index']),pitch=int(r['pitch']),finger=int(r['finger_id']),
               start=float(r['start_seconds']),end=float(r['end_seconds'])) for r in csv.DictReader(a.fingering.open())]
    nodes=onset_nodes(rows);groups={}
    for node in nodes:groups.setdefault(static_signature(node,contract),[]).append(node)
    cached={}
    for directory in a.cache:
        for path in directory.glob('candidates/*.json'):
            row=json.loads(path.read_text());signature=row['signature']
            if row.get('contract_sha256')==digest(contract):
                cached_action(row,signature,contract);cached[signature]=row
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'candidates').mkdir();(a.output/'groups').mkdir()
    (a.output/'worker_source.py').write_bytes(Path(__file__).read_bytes())
    settings=dict(recording=str(a.recording.resolve()),geometry=str(a.geometry.resolve()),output=str(a.output.resolve()),
        model_sha256=a.expected_model_sha256,archive_sha256=a.expected_archive_sha256,node_cpu_budget=a.node_cpu_budget,
        max_nfev=a.max_nfev,penetration=a.penetration,seed_banks=[str(p.resolve()) for p in a.seed_bank])
    metadata=dict(kind='fullscore_static_contact_bank_v1',contract=contract,contract_sha256=digest(contract),
        arguments={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(a).items()},
        command=sys.argv,pid=os.getpid(),hostname=os.uname().nodename,started_unix_s=time.time(),
        source_fingering_sha256=sha(a.fingering),worker_source_sha256=sha(__file__),source_file_count=len(source['source_sha256']),
        seed_bank_sha256={str(p.resolve()):sha(p) for p in a.seed_bank},requested_onsets=len(nodes),unique_combinations=len(groups),
        physics_steps=0,training_performed=False,dynamic_acceptance=None)
    (a.output/'run.json').write_text(json.dumps(metadata,indent=2))
    jobs=[]
    for signature,occurrences in groups.items():
        old=cached.get(signature);reuse=old is not None and (old['check']['static_passed'] or not a.retry_failed)
        jobs.append((signature,occurrences,contract,None if old is None else old['action'],old if reuse else None))
    if a.max_groups:jobs=jobs[:a.max_groups]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    done=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker,initargs=(settings,)) as pool:
        futures=[pool.submit(worker_group,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();candidate=result['candidate'];signature=candidate['signature']
            (a.output/'candidates'/f'{signature}.json').write_text(json.dumps(candidate,indent=2))
            (a.output/'groups'/f'{signature}.json').write_text(json.dumps(result['occurrences']))
            done.append(result)
            print(json.dumps(dict(completed_groups=len(done),total_groups=len(jobs),signature=signature,
                static_passed=candidate['check']['static_passed'],top=candidate['check']['assigned_top_count'],
                assigned=candidate['check']['assigned_count'],cpu_seconds=candidate['solver_cpu_seconds'])),flush=True)
    records=sorted((row for group in done for row in group['occurrences']),key=lambda r:r['time_seconds'])
    with (a.output/'occurrences.jsonl').open('w') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    report={**metadata,'finished_unix_s':time.time(),'exit_status':0,'completed_combinations':len(done),
        'completed_onsets':len(records),'all_requested_onsets_checked':len(records)==len(nodes),
        'static_passed_combinations':sum(g['candidate']['check']['static_passed'] for g in done),
        'static_passed_onsets':sum(r['static_passed'] for r in records),
        'all_assigned_top_onsets':sum(r['assigned_top_count']==r['assigned_count'] for r in records),
        'reused_cached_combinations':sum(bool(g['candidate'].get('reused_cache_candidate')) for g in done),
        'summed_solver_cpu_seconds':sum(g['candidate']['solver_cpu_seconds'] for g in done),
        'limitations':['Static keys are placed at maximum angle only in scratch data; no physical key was pressed.',
            'Every occurrence rechecks the complete candidate but its held-note context does not simulate entry/release.',
            'Inactive finger policy, finite seeds and budget can cause failed nodes; failure is not proof of kinematic impossibility.',
            'Cached complete poses are candidates only. No interpolation, drive tracking, actual force or dynamic replay is certified.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('completed_onsets','static_passed_onsets','static_passed_combinations','all_assigned_top_onsets')}),flush=True)


if __name__=='__main__':main()
