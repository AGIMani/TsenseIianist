#!/usr/bin/env python3
"""Repeat archived-source checkpoint evaluation and record substep diagnostics.

The source archive is temporary and must exactly match checkpoint fingerprints.
Canonical evaluation keeps physics graphs; detailed runs use the existing
read-only substep callback, as the video renderer does, and are labelled apart.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CLEARANCE_INFO_KEYS = (
    'finger_actual_deficit_m','finger_requested_deficit_m','finger_applied_deficit_m',
    'finger_clearance_deficit_m','finger_projection_mask','finger_fallback_mask',
    'finger_pose_reference_feasible','finger_active','finger_terminal_allowed')
ACTION_INFO_KEYS = ('raw_policy_action','requested_physical_action','applied_physical_action',
                    'control_clip_mask','control_slew_mask')


def attach_substep_capture(step, callback):
    """Forward task-specific keywords, including the current assigned keys."""
    def recorded(action, **kwargs):
        if kwargs.get('substep_callback') is not None:
            raise ValueError('Cannot nest two substep recorders')
        kwargs['substep_callback']=callback
        return step(action, **kwargs)
    return recorded


def diagnostic_summary(values, *, frames, physical_action_names):
    """CPU summaries keep target FK deficits distinct from actual-state deficits."""
    import numpy as np
    summary=dict(frames=frames, recorded_fields=list(values), fields={})
    for name,data in values.items():
        if data.shape[0]!=frames:raise ValueError('Incomplete diagnostic field '+name)
        if data.dtype==np.bool_:
            summary['fields'][name]=dict(any_frames=int(data.any(-1).sum()),
                channel_frame_fraction=float(data.mean()),per_channel_frames=data.sum(0).tolist())
        elif name.endswith('_deficit_m'):
            summary['fields'][name]=dict(max_m=float(data.max()),mean_m=float(data.mean()),
                above_1um_frames=int((data>1e-6).any(-1).sum()),
                above_1um_per_finger_frames=(data>1e-6).sum(0).tolist())
    if all(name in values for name in ('requested_physical_action','applied_physical_action')):
        delta=np.abs(values['requested_physical_action']-values['applied_physical_action'])
        summary['requested_applied_difference']=dict(any_frames=int((delta>1e-7).any(-1).sum()),
            per_channel=[dict(name=name,max_absolute_delta=float(delta[:,i].max()),
                changed_frames=int((delta[:,i]>1e-7).sum()),
                unit='m'if '_Palm_Base_T' in name else'pedal'if name=='sustain'else'rad')
                for i,name in enumerate(physical_action_names)],
            scope='Combined ordinary target limits and finger geometry correction; not physical motion')
    return summary


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoints',nargs='+',default=['step-8500.pt','step-9000.pt','step-9500.pt','final.pt'])
    p.add_argument('--repeats',type=int,default=3,help='Canonical episodes per checkpoint (default 3; 1 is a diagnostic spot check only)')
    p.add_argument('--detail-repeats',type=int,default=1)
    p.add_argument('--extended',action='store_true',help='Record actuator force, all contact normals/locations and model index/range metadata')
    p.add_argument('--worker-root',type=Path,help=argparse.SUPPRESS)
    return p.parse_args()


def spans(mask,dt):
    import numpy as np
    padded=np.r_[False,mask,False].astype(np.int8)
    starts=np.flatnonzero(np.diff(padded)==1)
    ends=np.flatnonzero(np.diff(padded)==-1)
    return [dict(start_seconds=float((s+1)*dt),end_seconds=float(min(e+1,len(mask))*dt),
                 samples=int(e-s),sampled_duration_seconds=float((e-s)*dt),
                 present_at_episode_end=bool(e==len(mask))) for s,e in zip(starts,ends)]


def worker(a, *, prepared_probe=None):
    """Record exact-source replay, or an explicitly migrated probe prepared by its caller.

    ``prepared_probe`` is an internal recording API, never a source-check CLI
    bypass. Its caller must build/validate the scene/task and load the network
    through the explicit controller-change migration gate before recording.
    """
    sys.path.insert(0,str(a.worker_root))
    import numpy as np
    import torch
    import mujoco
    from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent,reward_options
    from tsensei_v3_2_pianist.rl_torch.train import evaluate
    from tsensei_v3_2_pianist.preview_audio import events_from_states
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if prepared_probe is None:
        config=json.loads((a.run/'config.json').read_text())
        validate_config=runpy.run_path(str(a.worker_root/'scripts/eval_from_source.py'))['validate_config']
        first=a.run/a.checkpoints[0]
        scene_path,task_path=for_checkpoint(first)
        options=reward_options(first)
        diagnostics=bool(config.get('cli',{}).get('collision_diagnostics',False))
        environment_options={**options}
        if diagnostics:environment_options.setdefault('collision_diagnostics',True)
        env=PianoTaskEnv(scene_path,task_path,device='cuda:0',**environment_options)
    else:
        config=prepared_probe['config'];env=prepared_probe['env']
        scene_path=prepared_probe['scene_path'];task_path=prepared_probe['task_path']
        options=prepared_probe['options']
    s=env.scene;m=s.model;b=s.bundle
    dt=float(env.bundle.manifest['control_dt']);pdt=float(b.manifest['physics_dt'])
    decimation=s.decimation;music_start=round(env.bundle.manifest['contract']['environment']['initial_buffer_time']/dt)
    aids=np.asarray(b.action_ids);jids=m.actuator_trnid[aids,0];dofs=m.jnt_dofadr[jids]
    action_names=[b.manifest['action_names'][i] for i in b.active_action_indices]
    groups={key:np.array([i for i,name in enumerate(action_names) if predicate(name)],dtype=int)
            for key,predicate in {
                'wrist_tx':lambda n:n.endswith('_Palm_Base_Tx'),
                'wrist_ty':lambda n:n.endswith('_Palm_Base_Ty'),
                'wrist_pitch':lambda n:n.endswith('_Palm_Base_Pitch'),
                'finger':lambda n:'_Palm_Base_' not in n}.items()}
    geom_names=[m.geom(i).name or f'geom#{i}' for i in range(m.ngeom)]
    geom_bodies=np.array(m.geom_bodyid)
    body_names=[m.body(i).name or f'body#{i}' for i in range(m.nbody)]
    piano_geoms={m.geom(name).id for name in env.bundle.manifest['key_geoms']}
    key_indices={m.geom(name).id:i for i,name in enumerate(env.bundle.manifest['key_geoms'])}
    if a.extended:
        index_metadata=dict(ngeom=m.ngeom,joints=[dict(action_index=i,name=name,actuator_id=int(aids[i]),
                          joint_id=int(jids[i]),joint_name=m.joint(int(jids[i])).name,
                          qpos_adr=int(m.jnt_qposadr[jids[i]]),qvel_adr=int(dofs[i]),
                          joint_range=m.jnt_range[jids[i]].tolist(),joint_limited=bool(m.jnt_limited[jids[i]]),
                          ctrl_range=m.actuator_ctrlrange[aids[i]].tolist(),
                          force_range=m.actuator_forcerange[aids[i]].tolist(),force_limited=bool(m.actuator_forcelimited[aids[i]]),
                          gain=m.actuator_gainprm[aids[i]].tolist(),bias=m.actuator_biasprm[aids[i]].tolist())
                          for i,name in enumerate(action_names)],
                          geom_names=geom_names,body_names=body_names,geom_body_ids=geom_bodies.tolist(),
                          geom_type=m.geom_type.tolist(),geom_size=m.geom_size.tolist(),
                          site_names=[m.site(i).name or f'site#{i}' for i in range(m.nsite)],
                          fingertip_sites={side:list(b.manifest['hands'][side]['sites']) for side in ('right','left')},
                          key_geom_midi={str(g):i+21 for g,i in key_indices.items()},
                          action_contract=config['action_contract'],control_limits_contract=getattr(s,'control_limits',None),
                          finger_clearance_contract=getattr(s,'finger_clearance',None))
        (a.output/'model_indices.json').write_text(json.dumps(index_metadata,indent=2)+'\n')
        np.savez_compressed(a.output/'score.npz',goal=env.bundle.goal,fingers=env.bundle.fingers)
        # Static CPU forward kinematics is independent of the GPU rollout. Save
        # both poses so geometry clearance can be audited without rebuilding or
        # silently replacing the archived model with current package sources.
        for pose in ('origin','initial'):
            if pose=='initial':data=b.reset_cpu()
            else:
                data=mujoco.MjData(m)
                for field in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
                    getattr(data,field)[:]=getattr(b,'origin_'+field)
                mujoco.mj_forward(m,data)
            np.savez_compressed(a.output/f'{pose}_geometry.npz',
                                **{key:getattr(data,key).copy() for key in
                                   ('qpos','qvel','ctrl','mocap_pos','mocap_quat','site_xpos','geom_xpos','geom_xmat')})
    def side(g):
        name=geom_names[g]
        return 'right' if name.startswith('right_/') else 'left' if name.startswith('left_/') else None
    def category(g1,g2):
        one,two=side(g1),side(g2)
        if one and two:
            return 'same_hand' if one==two else 'cross_hand'
        if one or two:
            return 'hand_key' if g1 in piano_geoms or g2 in piano_geoms else 'hand_other'
        return None
    def detailed(agent,directory):
        directory.mkdir(parents=True,exist_ok=False)
        obs=env.reset()
        initial=dict(qpos=s.data.qpos[0].cpu().numpy().copy(),qvel=s.data.qvel[0].cpu().numpy().copy(),
                     ctrl=s.data.ctrl[0].cpu().numpy().copy(),observation=obs[0].cpu().numpy().copy(),
                     mocap_pos=s.data.mocap_pos[0].cpu().numpy().copy(),mocap_quat=s.data.mocap_quat[0].cpu().numpy().copy())
        physical={key:[] for key in ('qpos','qvel','ctrl','sustain','mocap_pos')}
        if a.extended:
            physical.update({key:[] for key in ('actuator_force','qfrc_constraint','site_xpos','geom_xpos','geom_xmat')})
        controls={key:[] for key in ('observation','actions','goals','actual','reward','reward_terms','qpos','qvel','ctrl')}
        diagnostics={}
        contacts=[];contact_geometry=[]
        tick=0
        def capture(scene):
            nonlocal tick
            wd=scene.sim.wp_data
            count=int(wd.nacon.numpy()[0])
            geom=wd.contact.geom.numpy()[:count]
            dist=wd.contact.dist.numpy()[:count]
            if a.extended:
                points=wd.contact.pos.numpy()[:count];frames=wd.contact.frame.numpy()[:count]
                geom_pos=wd.geom_xpos.numpy()[0];geom_mat=wd.geom_xmat.numpy()[0]
            unique={}
            for ci,(pair,distance) in enumerate(zip(geom,dist)):
                g1,g2=sorted(int(v) for v in pair)
                if distance>0 or category(g1,g2) is None or geom_bodies[g1]==geom_bodies[g2]:continue
                code=g1*m.ngeom+g2
                unique[code]=min(unique.get(code,0.),float(distance))
                if a.extended:
                    key=int(pair[0]) if int(pair[0]) in piano_geoms else int(pair[1]) if int(pair[1]) in piano_geoms else -1
                    normal=frames[ci,0]
                    if key>=0:
                        local_point=(points[ci]-geom_pos[key])@geom_mat[key]
                        outward=normal if int(pair[0])==key else -normal
                        local_normal=outward@geom_mat[key]
                    else:local_point=local_normal=np.full(3,np.nan)
                    contact_geometry.append([tick,int(pair[0]),int(pair[1]),float(distance),
                                             *points[ci],*normal,key,*local_point,*local_normal])
            contacts.extend((tick,code,distance) for code,distance in unique.items())
            for key in ('qpos','qvel','ctrl','mocap_pos'):
                physical[key].append(getattr(scene.data,key)[0].cpu().numpy().copy())
            if a.extended:
                for key in ('actuator_force','qfrc_constraint','site_xpos','geom_xpos','geom_xmat'):
                    physical[key].append(getattr(wd,key).numpy()[0].copy())
            physical['sustain'].append(float(scene.sustain[0]))
            tick+=1
        original_step=s.step
        s.step=attach_substep_capture(original_step,capture)
        try:
            with torch.no_grad():
                for index in range(env.logic.length):
                    action=agent.act(obs,deterministic=True)
                    controls['observation'].append(obs[0].cpu().numpy().copy())
                    result=env.step(action);obs=result.observation
                    if getattr(s,'finger_clearance_guard',None) is not None:
                        missing=[key for key in CLEARANCE_INFO_KEYS if key not in result.info]
                        if missing:raise ValueError('Active clearance guard omitted diagnostics: '+str(missing))
                    for key in (*CLEARANCE_INFO_KEYS,*ACTION_INFO_KEYS):
                        if key in result.info:
                            diagnostics.setdefault(key,[]).append(result.info[key][0].cpu().numpy().copy())
                    values=dict(actions=action[0].cpu().numpy(),goals=result.info['goal_applied'][0].cpu().numpy(),
                                actual=result.info['key_activation'][0].cpu().numpy(),reward=float(result.reward[0]),
                                reward_terms=[float(result.info['reward_terms'][k][0]) for k in env.reward_term_names])
                    values.update({key:getattr(s.data,key)[0].cpu().numpy().copy() for key in ('qpos','qvel','ctrl')})
                    for key,value in values.items():controls[key].append(value)
                    if bool(result.done[0]):break
        finally:s.step=original_step
        controls={key:np.asarray(value) for key,value in controls.items()}
        diagnostics={key:np.asarray(value) for key,value in diagnostics.items()}
        clearance_diagnostics=diagnostic_summary(diagnostics,frames=len(controls['actions']),
                                                 physical_action_names=b.manifest['action_names'])
        physical={key:np.asarray(value) for key,value in physical.items()}
        contact_array=np.asarray(contacts,dtype=float).reshape(-1,3)
        key_limits=s.key_limits.cpu().numpy();threshold=b.manifest['key_activation_fraction']
        key_qpos=physical['qpos'][:,b.key_qadr]
        active=np.clip(key_qpos,key_limits[:,0],key_limits[:,1])>=key_limits[:,1]*threshold
        target=np.repeat(controls['goals'][:,:88].astype(bool),decimation,axis=0)
        assert np.array_equal(active[decimation-1::decimation],controls['actual'])
        events=events_from_states(key_qpos,key_limits,physical['sustain'],pdt,threshold)
        for event in events:
            if event['type']=='note_on':
                index=min(int(np.floor((event['time']-1e-9)/dt)),len(controls['goals'])-1)
                event.update(score_index=index,music_time=event['time']-music_start*dt,
                             wrong=not bool(controls['goals'][index,event['note']-21]))
        wrong=active & ~target
        wrong_spans=[]
        for key in np.flatnonzero(wrong.any(0)):
            for interval in spans(wrong[:,key],pdt):
                wrong_spans.append(dict(midi=int(key)+21,**interval,
                                        start_music_seconds=interval['start_seconds']-music_start*dt))
        pairs=[]
        category_ticks={name:np.zeros(len(active),dtype=bool) for name in ('same_hand','cross_hand','hand_key','hand_other')}
        control_body_sets={name:[set() for _ in controls['actions']] for name in ('same_hand','cross_hand')}
        for code in sorted({int(row[1]) for row in contacts}):
            chosen=contact_array[contact_array[:,1]==code]
            g1,g2=divmod(code,m.ngeom);kind=category(g1,g2)
            ticks=chosen[:,0].astype(int);mask=np.zeros(len(active),bool);mask[ticks]=True
            category_ticks[kind]|=mask
            pair_bodies=tuple(sorted((int(geom_bodies[g1]),int(geom_bodies[g2]))))
            if kind in control_body_sets:
                for control_index in np.unique(ticks//decimation):control_body_sets[kind][control_index].add(pair_bodies)
            pairs.append(dict(geom_ids=[g1,g2],geoms=[geom_names[g1],geom_names[g2]],
                              bodies=[body_names[x] for x in pair_bodies],category=kind,
                              sampled_frames=len(ticks),sampled_duration_seconds=len(ticks)*pdt,
                              max_penetration_m=float(-chosen[:,2].min()),intervals=spans(mask,pdt)))
        metrics={}
        for label,lo,hi in (('preparation',0,music_start),('music',music_start,len(controls['actions']))):
            state20=controls['actual'][lo:hi].astype(bool);goal20=controls['goals'][lo:hi,:88].astype(bool)
            state200=active[lo*decimation:hi*decimation];goal200=target[lo*decimation:hi*decimation]
            def key_metrics(actual,goal):
                tp=(actual&goal).sum(1);fp=(actual&~goal).sum(1);fn=(~actual&goal).sum(1);den=2*tp+fp+fn
                return dict(mean_frame_f1=float(np.divide(2*tp,den,out=np.ones(len(tp)),where=den>0).mean()),
                            micro_f1=float(2*tp.sum()/max(int(den.sum()),1)),wrong_key_frames=int(fp.sum()),
                            frames_with_wrong_keys=int((fp>0).sum()),missed_key_frames=int(fn.sum()))
            phase=dict(control_20hz=key_metrics(state20,goal20),diagnostic_200hz=key_metrics(state200,goal200))
            phase['note_on_events']=[e for e in events if e['type']=='note_on' and lo<=e['score_index']<hi]
            phase['contact_frames_200hz']={k:int(v[lo*decimation:hi*decimation].sum()) for k,v in category_ticks.items()}
            phase['contact_body_pairs_per_control_union']={k:dict(total=sum(len(x) for x in v[lo:hi]),maximum=max((len(x) for x in v[lo:hi]),default=0)) for k,v in control_body_sets.items()}
            metrics[label]=phase
        target_jumps=np.diff(np.vstack((initial['ctrl'],controls['ctrl'])),axis=0)[:,aids]
        motion={}
        for name,indices in groups.items():
            velocity=np.abs(physical['qvel'][:,dofs[indices]])
            jumps=np.abs(target_jumps[:,indices])
            vindex=np.unravel_index(velocity.argmax(),velocity.shape)
            jindex=np.unravel_index(jumps.argmax(),jumps.shape)
            motion[name]=dict(max_velocity_200hz=float(velocity.max()),rms_velocity_200hz=float(np.sqrt(np.mean(velocity**2))),
                              velocity_peak_seconds=float((vindex[0]+1)*pdt),velocity_joint=action_names[indices[vindex[1]]],
                              max_target_jump_20hz=float(jumps.max()),target_jump_command_seconds=float(jindex[0]*dt),
                              target_jump_joint=action_names[indices[jindex[1]]],
                              first_command_max_jump=float(jumps[0].max()),
                              velocity_unit='m/s' if name in ('wrist_tx','wrist_ty') else 'rad/s',
                              target_unit='m' if name in ('wrist_tx','wrist_ty') else 'rad')
        np.savez_compressed(directory/'rollout.npz',**controls)
        if diagnostics:
            np.savez_compressed(directory/'control_diagnostics.npz',**diagnostics,
                command_time_seconds=np.arange(len(controls['actions']))*dt,
                returned_state_time_seconds=(np.arange(len(controls['actions']))+1)*dt)
        (directory/'control_diagnostics.json').write_text(json.dumps(clearance_diagnostics,indent=2)+'\n')
        np.savez_compressed(directory/'physics_200hz.npz',**physical,contacts=contact_array,key_activation=active,
                            key_limits=key_limits,key_activation_fraction=threshold,physics_dt=pdt)
        if a.extended:
            np.savez_compressed(directory/'contact_geometry.npz',contacts=np.asarray(contact_geometry).reshape(-1,17),
                                columns=np.array(['tick','geom0','geom1','distance','position_x','position_y','position_z',
                                    'normal_x','normal_y','normal_z','key_geom','key_local_x','key_local_y','key_local_z',
                                    'key_outward_normal_x','key_outward_normal_y','key_outward_normal_z']))
        np.savez_compressed(directory/'initial.npz',**initial)
        report=dict(scope='deterministic policy with read-only substep callback; physics graph bypassed as in renderer',
                    actual_physics_steps=len(active),control_frames=len(controls['actions']),control_dt=dt,physics_dt=pdt,
                    canonical_metrics_separate=True,metrics=metrics,motion=motion,wrong_note_intervals=wrong_spans,
                    control_diagnostics=clearance_diagnostics,
                    contact_pairs=sorted(pairs,key=lambda p:(p['category'], -p['sampled_duration_seconds'])),
                    audio_events=events,reward_term_names=env.reward_term_names,
                    limitations=['200Hz scores repeat the currently applied 20Hz goal and are diagnostics, not a new training metric.',
                                 'Contact distances belong to contacts solved in the sampled substep; no contact forces were inferred.',
                                 'Detailed callback and canonical graph paths may diverge through floating-point contact differences.'])
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return {key:report[key] for key in ('scope','metrics','motion')}
    report=dict(source_archive=str(a.source_archive) if prepared_probe is None else None,
                archive_sha256=sha256(a.source_archive) if prepared_probe is None else None,
                verified_source_files=len(config['source_runtime']['source_sha256']) if prepared_probe is None else None,
                source_sha256=config['source_runtime']['source_sha256'],
                scene_sha256=sha256(scene_path/'manifest.json'),task_sha256=sha256(task_path/'task.json'),
                reward_options=options,checkpoints=[])
    if prepared_probe is not None:
        report['explicit_migration_probe']=prepared_probe['provenance']
    for name in a.checkpoints:
        path=a.run/name;metadata=torch.load(path,map_location='cpu',weights_only=True)['metadata']
        if prepared_probe is None:
            validate_config(config,metadata)
            assert reward_options(path)==options
            # Older archived loaders predate software control contracts. Only
            # pass the resolved contract when this checkpoint uses that feature.
            load_options={**options}
            if hasattr(s, 'action_contract'):
                load_options['action_contract'] = s.action_contract
            if options.get('control_limits') is not None:
                load_options['control_limits_contract']=s.control_limits
            if options.get('finger_clearance') is not None:
                load_options['finger_clearance_contract']=s.finger_clearance
            if options.get('finger_coupling') is not None:
                load_options['finger_coupling_contract']=s.finger_coupling
                load_options['coupling_observation_contract']=env.coupling_observation_contract
            agent=load_agent(path,scene_sha256=report['scene_sha256'],task_sha256=report['task_sha256'],
                             obs_size=env.observation_size,action_size=env.action_size,device='cuda:0',
                             preparation_contract=b.preparation,**load_options)
        else:
            if len(a.checkpoints)!=1:raise ValueError('A prepared probe records exactly its migrated parent actor')
            agent=prepared_probe['agent']
        canonical=[evaluate(agent,env,1) for _ in range(a.repeats)]
        print(name,'canonical music F1',[r['music_only_f1'] for r in canonical],flush=True)
        item=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),canonical_repeats=canonical,
                  mean_music_f1=float(np.mean([r['music_only_f1'] for r in canonical])),
                  min_music_f1=float(min(r['music_only_f1'] for r in canonical)),
                  max_music_f1=float(max(r['music_only_f1'] for r in canonical)),details=[])
        for repeat in range(a.detail_repeats):
            row=detailed(agent,a.output/name.removesuffix('.pt')/f'detail-{repeat}')
            item['details'].append(row)
            print(name,'detail',repeat,'music F1',row['metrics']['music']['control_20hz']['mean_frame_f1'],flush=True)
        report['checkpoints'].append(item)
        (a.output/'checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Audit complete',a.output,flush=True)


def main():
    a=arguments()
    a.run=a.run.resolve();a.output=a.output.resolve();a.source_archive=a.source_archive.resolve()
    if a.worker_root:
        worker(a);return
    if a.repeats<1 or a.detail_repeats<1:raise ValueError('Use at least one canonical episode and one detailed run')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'checkpoint_audit.json').exists():raise FileExistsError('Choose a new replay audit directory')
    from eval_from_source_archive import extract_archive,validate_sources
    config=json.loads((a.run/'config.json').read_text())
    with tempfile.TemporaryDirectory(prefix='tsensei-v32-checkpoint-audit-') as temporary:
        archive_root=Path(temporary)
        extract_archive(a.source_archive,archive_root)
        validate_sources(archive_root,config['source_runtime']['source_sha256'])
        for role in ('training','training_mjlab'):
            link=archive_root/role/'.venv';link.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to((ROOT/role/'.venv').resolve(),target_is_directory=True)
        env=os.environ.copy();env['PYTHONPATH']=str(archive_root);env['PYTHONDONTWRITEBYTECODE']='1';env['MUJOCO_GL']='egl'
        command=[str(ROOT/'training_mjlab/.venv/bin/python'),str(Path(__file__).resolve()),
                 '--run',str(a.run),'--source-archive',str(a.source_archive),'--output',str(a.output),
                 '--worker-root',str(archive_root),'--repeats',str(a.repeats),'--detail-repeats',str(a.detail_repeats),
                 '--checkpoints',*a.checkpoints]
        if a.extended:command.append('--extended')
        subprocess.run(command,cwd=archive_root,env=env,check=True)


if __name__=='__main__':main()
