#!/usr/bin/env python3
"""Isolated remote TZ/Pitch study. Never changes training defaults or fingering."""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ARCHIVE_SHA='1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560'
MODEL_SHA='920475c61dc0469779d9f64769c75044140163151461201609ec181a7509fd4f'
CSV_SHA='0ee06bea4c7711f4d77c4622055d7efcb16650ab0ec07b63dc7422050a1c1de9'
PROTO_SHA='1248acd036ce492ebaa1f8be3410f6f1d8f8e34dc87ed892af92123d6763773e'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def canonical_indices(names,actuators):
    if len(set(names))!=len(names) or len(set(actuators))!=len(actuators):raise ValueError('Duplicate control name')
    if any(n not in names for n in actuators):raise ValueError('Unknown canonical control')
    return [names.index(n) for n in actuators]


def patch_channels(channels,pitch_min,pitch_max):
    if not -.1<=pitch_min<=0 or not .5<=pitch_max<=.9:raise ValueError('Outside authorized narrow study')
    result=copy.deepcopy(channels)
    for c in result:
        if c['name'].endswith('_Palm_Base_Tz'):c.update(physical_min=0.,physical_max=.03)
        if c['name'].endswith('_Palm_Base_Pitch'):c.update(physical_min=pitch_min,physical_max=pitch_max)
    return result


def select_cases(rows,maximum=16):
    groups={};counts=collections.Counter(r['signature'] for r in rows)
    for row in sorted(rows,key=lambda x:x['time_seconds']):groups.setdefault(row['signature'],row)
    buckets=collections.defaultdict(list)
    for sig,r in groups.items():
        pitches=[[n['pitch'] for n in r['notes'] if n['finger']//5==side] for side in (0,1)]
        if r['static_passed']:tags=['static_pass_control']
        else:
            tags=['failed']
            if any(r['action'][i]>=.498 for i in (23,48)):tags.append('upper_pitch')
            if any(r['action'][i]<=.002 and pitches[s] for s,i in enumerate((23,48))):tags.append('lower_pitch')
            if any(len(p)>=4 and max(p)-min(p)>=10 for p in pitches):tags.append('wide_four')
            if any(len(p)>=2 and max(p)-min(p)>=12 for p in pitches):tags.append('octave')
        for tag in tags:buckets[tag].append((sig,r,tags))
    for vals in buckets.values():vals.sort(key=lambda x:(-counts[x[0]],x[1]['time_seconds']))
    chosen={};order=('upper_pitch','lower_pitch','wide_four','octave','static_pass_control','failed')
    for round_index in range(maximum):
        for tag in order:
            vals=[v for v in buckets[tag] if v[0] not in chosen]
            if not vals:continue
            sig,r,tags=vals[0]
            chosen[sig]={**r,'occurrence_count':counts[sig],'selection_reasons':tags}
            if len(chosen)>=maximum:return list(chosen.values())
    return list(chosen.values())


def build_variant(a):
    """Extract exact frozen source, alter only virtual axes, rebuild both structures."""
    import base64
    import difflib
    import shutil
    import tarfile
    assert sha(a.archive)==ARCHIVE_SHA and sha(a.score)==PROTO_SHA
    assert sha(a.recording/'model.mjb')==MODEL_SHA
    assert sha(a.recording/'source_config.json')=='17695e6ed5c39811049161b74d95fd60f8ceeafb1bc08edf1b604fda8596e1b8'
    if (a.variant_root/'study/complete.json').exists():raise ValueError('Completed build cannot be overwritten')
    a.variant_root.mkdir(parents=True,exist_ok=a.resume_build)
    with tarfile.open(a.archive,'r:gz') as tar:tar.extractall(a.variant_root,filter='data')
    for role in ('training','training_mjlab'):
        p=a.variant_root/role;p.mkdir(exist_ok=True)
        if not (p/'.venv').exists():(p/'.venv').symlink_to((ROOT/role/'.venv').resolve(),target_is_directory=True)
    names=('tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py','configs/actions/direct_joint_v1.json')
    changes={};patch=[]
    for name in names:
        path=a.variant_root/name;before=path.read_text()
        if name.endswith('.py'):
            after=before.replace('"forearm_tz": ("slide", (-1, 0, 0), (-0.025, 0),','"forearm_tz": ("slide", (-1, 0, 0), (0, 0.030),')
            after=after.replace('"forearm_pitch": ("hinge", (1, 0, 0), (0.0, 0.5),',f'"forearm_pitch": ("hinge", (1, 0, 0), ({a.pitch_min}, {a.pitch_max}),')
        else:
            doc=json.loads(before);doc['channels']=patch_channels(doc['channels'],a.pitch_min,a.pitch_max)
            after=json.dumps(doc,indent=2)+'\n'
        assert after!=before
        changes[name]={'before':sha(path)};path.write_text(after);changes[name]['after']=sha(path)
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (a.variant_root/'study_source.patch').write_text(''.join(patch))
    sys.path.insert(0,str(a.variant_root));os.chdir(a.variant_root)
    import numpy as np
    import mujoco
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import source_files,build
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.finger_resistance import Resistance
    original=json.loads((a.recording/'source_config.json').read_text());recipe=copy.deepcopy(original)
    recipe['source_sha256']=source_files()
    difference={k for k in original['source_sha256']|recipe['source_sha256'] if original['source_sha256'].get(k)!=recipe['source_sha256'].get(k)}
    assert difference==set(names),difference
    recipe['external_score']={'name':a.score.name,'suffix':a.score.suffix,'sha256':PROTO_SHA,'base64':base64.b64encode(a.score.read_bytes()).decode()}
    recipe['environment']['midi']=a.score.name
    out=a.variant_root/'study';out.mkdir(exist_ok=a.resume_build)
    write(out/'source_config.json',recipe);scene,task,_=build(recipe)
    shutil.copytree(scene,out/'scene',dirs_exist_ok=a.resume_build);shutil.copytree(task,out/'task',dirs_exist_ok=a.resume_build)
    shutil.copy2(a.variant_root/names[1],out/'action_contract.json')
    original_model=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'))
    def matched(m):
        checks=[]
        for kind,fields in (('body',('pos','quat','mass','inertia','gravcomp')),('geom',('type','size','pos','quat','contype','conaffinity','solref','solimp'))):
            count=getattr(original_model,'n'+kind)
            assert getattr(m,'n'+kind)==count
            for i in range(count):
                name=getattr(original_model,kind)(i).name
                for field in fields:
                    left=getattr(original_model,kind+'_'+field)[getattr(original_model,kind)(name).id]
                    right=getattr(m,kind+'_'+field)[getattr(m,kind)(name).id]
                    np.testing.assert_allclose(right,left,atol=1e-12,rtol=0,err_msg=name+'.'+field)
            checks.append(kind+'_named_geometry_and_parameters')
        np.testing.assert_array_equal(m.mesh_vert,original_model.mesh_vert);checks.append('all_mesh_vertices_exact')
        for i in range(original_model.nu):
            old=original_model.actuator(i);new=m.actuator(old.name)
            for f in ('gainprm','biasprm','forcerange','gear'):np.testing.assert_array_equal(getattr(new,f),getattr(old,f))
            if not old.name.endswith('_Pitch'):np.testing.assert_array_equal(new.ctrlrange,old.ctrlrange)
        checks.append('all_original_actuator_gains_bias_force_gear_and_non_pitch_controls_exact')
        for i in range(original_model.njnt):
            old=original_model.joint(i);new=m.joint(old.name)
            for f in ('type','axis','pos'):np.testing.assert_array_equal(getattr(new,f),getattr(old,f))
            if not old.name.endswith('/forearm_pitch'):np.testing.assert_array_equal(new.range,old.range)
        checks.append('all_original_joint_type_axis_pos_and_non_pitch_ranges_exact')
        return checks
    models=[]
    for tz in (False,True):
        b=SceneBundle(out/'scene',disable_wrist_tz=not tz,disable_wrist_yaw=True);m=b.model;d=b.reset_cpu()
        resistance=Resistance(m);resistance.update(d);assert len(resistance.rows)==40
        checks=matched(m);folder=out/('tz_on' if tz else 'tz_off');folder.mkdir(exist_ok=a.resume_build)
        mujoco.mj_saveModel(m,str(folder/'model.mjb'),None)
        full=canonical_indices(b.manifest['action_names'],[m.actuator(i).name for i in range(m.nu)])
        expected=[i for i in range(50) if i not in ((24,49) if tz else (22,24,47,49))]
        assert full==expected
        axes={}
        mujoco.mj_kinematics(m,d)
        for side,off in (('right',0),('left',25)):
            for axis in ('pitch',)+(('tz',) if tz else ()):
                j=m.joint(side+'_/forearm_'+axis);label=side[0].upper()+'_Palm_Base_'+('Pitch' if axis=='pitch' else 'Tz')
                act=m.actuator(label);ran=[a.pitch_min,a.pitch_max] if axis=='pitch' else [0.,.03]
                np.testing.assert_allclose(j.range,ran,atol=1e-12,rtol=0);np.testing.assert_allclose(act.ctrlrange,ran,atol=1e-12,rtol=0)
                axes[label]={'joint_range':j.range.tolist(),'control_range':act.ctrlrange.tolist(),'world_axis':d.xaxis[j.id].tolist(),
                             'qpos_address':int(j.qposadr[0]),'canonical_index':off+(23 if axis=='pitch' else 22)}
        doc=dict(kind='isolated_tz_pitch_variant_v1',parent_archive_sha256=ARCHIVE_SHA,parent_model_sha256=MODEL_SHA,
            model_sha256=sha(folder/'model.mjb'),source_config_sha256=sha(out/'source_config.json'),source_patch_sha256=sha(a.variant_root/'study_source.patch'),
            patched_sources=changes,score_sha256=PROTO_SHA,geometry_match_checks=checks,tz_enabled=tz,pitch_range=[a.pitch_min,a.pitch_max],
            action_contract_sha256=sha(out/'action_contract.json'),scene_manifest_sha256=sha(out/'scene/manifest.json'),
            active_full_indices=full,axes=axes,nq=m.nq,nu=m.nu,physics_hz=1000,control_hz=20,mujoco_version=mujoco.__version__,
            worker_source_sha256=sha(__file__),pid=os.getpid(),command=sys.argv,training_compatible=False,
            limitations='Research virtual axes only; no hardware calibration; 49-channel policy/TZ-Yaw ambiguity deliberately bypassed with name mapping')
        write(folder/'variant.json',doc);models.append(str(folder))
    write(out/'complete.json',dict(models=models,exit_status=0,command=sys.argv,pid=os.getpid(),finished_unix_s=time.time()))
    print(json.dumps(dict(models=models,complete=True)),flush=True)


def variant_model(path,settings):
    """Adapt only the verified isolated model to the frozen bank's geometry evaluator."""
    from types import SimpleNamespace
    import numpy as np
    import mujoco as mj
    import torch
    from scripts.hpc.piano_fullscore_contact_bank import StaticModel
    from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
    from tsensei_v3_2_pianist.fingering_targets import key_reference
    from scripts.self_collision_metrics import finger_identity,validate_coupling_contract
    from scripts.plan_model_fingering import Geometry
    torch.set_num_threads(1)
    path=Path(path);doc=json.loads((path/'variant.json').read_text())
    assert doc['parent_model_sha256']==MODEL_SHA and doc['parent_archive_sha256']==ARCHIVE_SHA
    assert sha(path/'model.mjb')==doc['model_sha256'] and sha(path.parent/'scene/manifest.json')==doc['scene_manifest_sha256']
    assert sha(path.parent/'action_contract.json')==doc['action_contract_sha256']
    assert sha(path.parent/'source_config.json')==doc['source_config_sha256']
    manifest=json.loads((path.parent/'scene/manifest.json').read_text());m=mj.MjModel.from_binary_path(str(path/'model.mjb'))
    assert mj.__version__==doc['mujoco_version']
    names=manifest['action_names'];full=np.array(canonical_indices(names,[m.actuator(i).name for i in range(m.nu)]))
    assert full.tolist()==doc['active_full_indices']
    acts=np.arange(m.nu);joints=m.actuator_trnid[:,0];qadr=m.jnt_qposadr[joints]
    channels={c['name']:c for c in json.loads((path.parent/'action_contract.json').read_text())['channels']}
    low=np.array([channels[n]['physical_min'] for n in names]);high=np.array([channels[n]['physical_max'] for n in names])
    for i,aid in zip(full,acts):
        assert low[i]>=m.actuator_ctrlrange[aid,0]-1e-9 and high[i]<=m.actuator_ctrlrange[aid,1]+1e-9
    origin=np.zeros(m.nq)
    for name,value in zip(manifest['joint_names'],manifest['qpos']):
        j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name)
        if j>=0:origin[m.jnt_qposadr[j]]=value
    neutral=np.array(manifest['neutral_action']);neutral[[22,47]]=0
    def reset():
        d=mj.MjData(m);d.qpos[:]=origin;mj.mj_forward(m,d);return d
    bundle=SimpleNamespace(model=m,manifest=manifest,action_ids=acts,active_action_indices=full,low=np.array(manifest['action_min']),
        high=np.array(manifest['action_max']),neutral_action=neutral,origin_qpos=origin,origin_mocap_pos=np.zeros((0,3)),
        origin_mocap_quat=np.zeros((0,4)),reset_cpu=reset,key_qadr=np.array([m.joint(n).qposadr[0] for n in manifest['piano_joints']]))
    options=json.loads((Path(settings['recording'])/'model_diagnostic.json').read_text())['coupling_options'];validate_coupling_contract(options)
    coupler=FingerCoupler(bundle,device='cpu',options=options,physical_low=low,physical_high=high)
    followers=set(coupler.indices[:,1:].flatten().tolist())
    p=SimpleNamespace(model=m,data=reset(),full=full,qadr=qadr,low=low,high=high,origin=origin,neutral=neutral,coupler=coupler,
                      names=names,np=np,mj=mj,torch=torch,controls=[dict(index=int(i)) for i in full if i not in followers])
    class Variant(StaticModel):
        def seed(self,node):
            x=super().seed(node)
            if self.fixed_tz is not None:x[[22,47]]=self.fixed_tz
            return x
        def set_pose(self,action,node):
            requested=np.clip(action,p.low,p.high).copy();requested[[24,49]]=self.neutral[[24,49]]
            if self.fixed_tz is not None:requested[[22,47]]=self.fixed_tz
            elif not doc['tz_enabled']:requested[[22,47]]=0.
            d=self.d;d.qpos[:]=origin;d.qvel[:]=0.;d.qpos[qadr]=requested[full]
            for n in node['notes']:
                key='piano/'+('white' if 'white' in m.geom(self.keys[n['pitch']]).name else 'black')+'_joint_'+str(n['pitch']-21)
                j=m.joint(key);d.qpos[j.qposadr[0]]=j.range[1]
            delta=float('inf')
            for it in range(8):
                mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
                mapped,info=coupler.apply(torch.tensor(requested[None],dtype=torch.float32),actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
                applied=mapped[0].numpy().astype(float);delta=float(np.max(abs(applied[full]-d.qpos[qadr])))
                d.qpos[qadr]=applied[full]
                if delta<1e-7:break
            mj.mj_fwdPosition(m,d)
            return applied,dict(coupling_converged=delta<1e-7,coupling_iterations=it+1,coupling_error_max_deg=float(info['coupling_target_error_deg'].abs().max()),coupling_pose_delta_rad=delta)
    obj=Variant.__new__(Variant);obj.settings=settings;obj.probe=p;obj.np=np;obj.mj=mj;obj.torch=torch;obj.m=m;obj.d=p.data
    obj.source=json.loads(Path(settings['geometry']).read_text());assert obj.source['source_archive_sha256']==ARCHIVE_SHA
    obj.geometry=Geometry(obj.source);obj.neutral=neutral.copy()
    for off in (16,41):obj.neutral[off:off+4]=[.490541428,.01000039,.389615625,.213344008]
    obj.independent=[c['index'] for c in p.controls];obj.fixed_tz=None;obj.keys={};obj.terminal={};obj.vertices={};obj.hand=[];obj.identities={};obj.extra=[]
    digits=['thumb','index','mid','ring','pinky']
    for g in range(m.ngeom):
        name=m.geom(g).name or '';body=m.body(int(m.geom_bodyid[g])).name or ''
        if name.startswith(('piano/white_key_geom_','piano/black_key_geom_')):obj.keys[int(name.rsplit('_',1)[1])+21]=g
        if name.startswith(('left_/','right_/')) and '_collision_' in name:
            obj.hand.append(g);identity=finger_identity(body);f=None if identity is None else (5 if identity[0]=='left' else 0)+digits.index(identity[1])
            obj.identities[g]=(0 if name.startswith('right_/') else 1,f)
            if f is not None and (('_thumb_PIP_link_' in name) if f%5==0 else '_DIP_link_' in name):
                obj.terminal.setdefault(f,[]).append(g);mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh]);obj.vertices[g]=m.mesh_vert[start:start+count].copy()
    assert len(obj.keys)==88 and len(obj.terminal)==10
    obj.key_pitch={g:pitch for pitch,g in obj.keys.items()};obj.key_refs={pitch:key_reference(m,g) for pitch,g in obj.keys.items()};obj.hand_rows={g:i for i,g in enumerate(obj.hand)}
    return obj,doc,bundle


def scan_worker(job):
    variant,cases,settings=job
    model,doc,_=variant_model(variant,settings);full_independent=model.independent.copy();rows=[]
    levels=[None,0.,.005,.01,.02,.03] if doc['tz_enabled'] else [None]
    from scripts.hpc.piano_fullscore_contact_bank import digest
    for case in cases:
        node={k:case[k] for k in ('time_seconds','notes','held_note_indices','onset_note_indices')}
        for level in levels:
            model.fixed_tz=level;model.independent=[i for i in full_independent if level is None or i not in (22,47)]
            seed=model.np.array(case['action']);seed[[22,47]]=0. if level is None else level
            contract={'variant':doc['model_sha256'],'fixed_tz_m':level,'bank_source_sha256':sha(ROOT/'scripts/hpc/piano_fullscore_contact_bank.py'),'study_source_sha256':sha(__file__)}
            result=model.solve(node,digest((case['signature'],contract)),contract,seed)
            rows.append(dict(case_signature=case['signature'],selection_reasons=case['selection_reasons'],occurrence_count=case['occurrence_count'],
                             variant=str(variant),tz_enabled=doc['tz_enabled'],pitch_range=doc['pitch_range'],fixed_tz_m=level,**result))
    return rows


def scan(a):
    from scripts.hpc.self_collision_probe import verify_sources
    verify_sources(a.recording,MODEL_SHA,ARCHIVE_SHA)
    report=json.loads((a.bank/'report.json').read_text());assert report['source_fingering_sha256']==CSV_SHA
    allrows=[json.loads(s) for s in (a.bank/'occurrences.jsonl').read_text().splitlines()]
    cases=select_cases(allrows,a.maximum_cases)
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'cases.json',cases)
    settings=dict(recording=str(a.recording),geometry=str(a.geometry),penetration=.0005,node_cpu_budget=5.,max_nfev=60,seed_banks=[])
    write(a.output/'run.json',dict(command=sys.argv,pid=os.getpid(),started_unix_s=time.time(),workers=min(8,len(a.variant)),
        source_fingering_sha256=CSV_SHA,bank_report_sha256=sha(a.bank/'report.json'),study_source_sha256=sha(__file__),
        bank_source_sha256=sha(ROOT/'scripts/hpc/piano_fullscore_contact_bank.py'),geometry_sha256=sha(a.geometry),physics_steps=0))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    rows=[]
    with ProcessPoolExecutor(max_workers=min(8,len(a.variant)),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(scan_worker,(str(p),cases,settings)) for p in a.variant]
        for future in as_completed(futures):
            part=future.result();rows.extend(part);print(json.dumps(dict(completed_variants=len({r['variant'] for r in rows}),rows=len(rows))),flush=True)
            write(a.output/'rows_partial.json',rows)
    write(a.output/'rows.json',rows)
    summary=[]
    for path in a.variant:
        subset=[r for r in rows if r['variant']==str(path)]
        passed={r['case_signature'] for r in subset if r['check']['static_passed']}
        summary.append(dict(variant=str(path),cases=len(cases),static_pass_cases=len(passed),all_top_cases=len({r['case_signature'] for r in subset if r['check']['assigned_top_count']==r['check']['assigned_count']}),
            passed_signatures=sorted(passed),rows=len(subset)))
    write(a.output/'report.json',dict(summary=summary,finished_unix_s=time.time(),exit_status=0,physics_steps=0,
        limitations=['Finite seeded static search; failure is not proof of impossibility.','Fixed-height sweeps apply the same TZ to both wrists; free-TZ solves independent right/left TZ.',
                    'Every variant is the same original fingering combination; no physical or transition score is reused.','No measured hardware range or calibration is claimed.']))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build');b.add_argument('--archive',type=Path,required=True);b.add_argument('--score',type=Path,required=True)
    b.add_argument('--variant-root',type=Path,required=True);b.add_argument('--pitch-min',type=float,default=0.);b.add_argument('--pitch-max',type=float,default=.5)
    b.add_argument('--resume-build',action='store_true',help='Rebuild only an incomplete isolated study root from its frozen archive')
    s=sub.add_parser('scan');s.add_argument('--bank',type=Path,required=True);s.add_argument('--geometry',type=Path,required=True)
    s.add_argument('--variant',type=Path,action='append',required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--maximum-cases',type=int,default=16)
    for q in (b,s):q.add_argument('--recording',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SSH_CONNECTION'):p.error('Model construction, loading and physics are remote-only')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
        if isinstance(v,list):setattr(a,k,[x.resolve() if isinstance(x,Path) else x for x in v])
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable')
    if a.mode=='build':build_variant(a)
    elif a.mode=='scan':scan(a)


if __name__=='__main__':main()
