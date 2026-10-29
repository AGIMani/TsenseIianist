#!/usr/bin/env python3
"""Retarget the verified TurkishB saved positions to current A3.2 physical servos.

Remote CPU only. The original MJB is used for coordinate verification, never
for the new dynamics. Only the initial robot pose is set; keys are never driven.
The six-DOF wrist extension is an explicit per-export model, not a training default.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=17.)
    p.add_argument('--feedforward',action='store_true',help='Compensate current inertia/damping/friction through the unchanged position servos; reference stays unchanged')
    p.add_argument('--learn-from',type=Path,help='Previous same-reference physical trace for iterative feedforward correction through native capped actuators, never a reference replacement')
    p.add_argument('--feedback',action='store_true',help='Outer trajectory-error correction through unchanged capped position actuators')
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if a.feedback and not a.feedforward:p.error('--feedback requires --feedforward')
    if a.learn_from and (a.feedforward or a.feedback):p.error('--learn-from must be evaluated separately from --feedforward/--feedback')
    if not os.environ.get('SSH_CONNECTION') and not os.environ.get('SLURM_JOB_ID'):
        p.error('New physics and model compilation must run remotely')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'process.json',dict(pid=os.getpid(),hostname=os.uname().nodename,command=sys.argv,status='running'))
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    src=root/'source';sys.path.insert(0,str(src))
    inp=root/'input/TurkishB_2221_best24250';scene=root/'build/scene'
    package=json.loads((inp/'PACKAGE_MANIFEST.json').read_text())
    for name,digest in package['files'].items():
        path=(inp/name).resolve()
        if not path.is_relative_to(inp) or sha(path)!=digest:raise ValueError('Input hash mismatch '+name)
    model_source_sha=sha(root/'source_cpu_replay_v1.tar.gz')
    if model_source_sha!='5a5a21db04d3f969b2052956568bb02d47368b457eb5760ab18dfd29971ceaa1':raise ValueError('Unexpected current source')
    cfg=json.loads((inp/'recording/replay_config.json').read_text())
    if mujoco.__version__!=cfg['mujoco_version']:raise ValueError('Source MJB version mismatch')
    original=mujoco.MjModel.from_binary_path(str(inp/'recording/scene.mjb'))
    reference=mujoco.MjData(original)
    base=mujoco.MjModel.from_binary_path(str(root/'build/current.mjb'))
    base_data=mujoco.MjData(base);mujoco.mj_kinematics(base,base_data)
    with np.load(inp/'recording/actual_rollout.npz',allow_pickle=False) as z:
        original_q=z['q_actual'];original_t=z['playback_time'];original_names=list(z['joint_names'])
    original_qadr=np.array([original.joint(n).qposadr[0] for n in original_names])
    if not np.all(np.diff(original_t)>0) or not np.isfinite(original_q).all():raise ValueError('Invalid input samples')
    mapping=json.loads((root/'joint_mapping_audit.json').read_text())
    if len(mapping)!=40 or any(abs(abs(m['axis_dot'])-1)>1e-8 or m['anchor_difference_m']>1e-8 for m in mapping):raise ValueError('Invalid joint correspondence')
    xml=ET.parse(scene/'scene.xml').getroot();act=xml.find('actuator')
    # Current joints and actuators remain; explicitly unlock TZ/Yaw by retaining
    # their structural joints, and add the missing third orthogonal rotation.
    extensions=[]
    for side in ('left','right'):
        body=next(b for b in xml.iter('body') if b.get('name')==side+'_/base_link')
        pitch=next(j for j in body.findall('joint') if j.get('name').endswith('/palm_base_pitch'))
        roll=ET.Element('joint',name=side+'_/palm_base_roll',type='hinge',axis='0 0 1',pos='0 0 0',range='-0.5 0.5',limited='true',damping=pitch.get('damping'))
        last=max(i for i,n in enumerate(body) if n.tag=='joint');body.insert(last+1,roll)
        oldact=next(n for n in act if n.get('joint')==side+'_/palm_base_pitch')
        newact=copy.deepcopy(oldact);newact.set('name',side[0].upper()+'_Palm_Base_Roll');newact.set('joint',roll.get('name'));act.append(newact)
        # This original wrist records its base frame above the current default
        # fixed height. The explicit replay range admits that source trajectory.
        z=next(j for j in body.findall('joint') if j.get('name').endswith('/palm_base_tz'))
        z.set('range','-0.025 0.04')
        za=next(n for n in act if n.get('joint')==z.get('name'));za.set('ctrlrange','-0.025 0.04')
        extensions.append(dict(side=side,enabled=['tx','ty','tz','pitch','yaw','roll'],roll_axis_local=[0,0,1],roll_range_rad=[-.5,.5],roll_gain_source='current Pitch actuator copied',tz_range_m=[-.025,.04],training_defaults_changed=False))
    # Match this replay's frozen 1000Hz baseline and its 2*dt piano solref.
    xml.find('option').set('timestep','0.001')
    for geom in xml.iter('geom'):
        if (geom.get('name') or '').startswith('piano/'):
            old=base.geom(geom.get('name')).id
            values=base.geom_solref[old].copy();values[0]=.002
            geom.set('solref',' '.join(map(str,values)))
    # Keyframes from a 138-DOF model would silently mislabel the added coordinates.
    if xml.find('keyframe') is not None:xml.remove(xml.find('keyframe'))
    assets={n:(scene/n).read_bytes() for n in json.loads((scene/'manifest.json').read_text())['files'] if n not in ('scene.xml','cpu_reference.npz')}
    xmltext=ET.tostring(xml,encoding='unicode');(out/'model.xml').write_text(xmltext)
    model=mujoco.MjModel.from_xml_string(xmltext,assets)
    assert (model.nq,model.nv,model.nu)==(140,140,52)
    assert model.opt.timestep==.001
    # Verify current physical piano, geometry and finger actuators were retained.
    for j in range(base.njnt):
        n=base.joint(j).name
        if n.startswith('piano/'):
            k=model.joint(n).id;np.testing.assert_array_equal(model.jnt_range[k],base.jnt_range[j])
    for i in range(base.nu):
        n=base.actuator(i).name;k=model.actuator(n).id
        for field in ('actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_forcelimited','actuator_gear'):
            np.testing.assert_array_equal(getattr(model,field)[k],getattr(base,field)[i])
    for g in range(base.ngeom):
        k=model.geom(base.geom(g).name).id
        for field in ('geom_type','geom_bodyid','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_condim','geom_solimp'):
            np.testing.assert_allclose(getattr(model,field)[k],getattr(base,field)[g],atol=1e-12,rtol=0)
    for b in range(base.nbody):
        k=model.body(base.body(b).name).id
        for field in ('body_mass','body_inertia','body_gravcomp'):
            np.testing.assert_allclose(getattr(model,field)[k],getattr(base,field)[b],atol=1e-12,rtol=0)
    for j in range(base.njnt):
        k=model.joint(base.joint(j).name).id
        for field in ('dof_damping','dof_armature','dof_frictionloss'):
            np.testing.assert_allclose(getattr(model,field)[model.jnt_dofadr[k]],getattr(base,field)[base.jnt_dofadr[j]],atol=1e-12,rtol=0)
    write(out/'retained_model_validation.json',dict(passed=True,existing_actuators=base.nu,
        geometry_and_contact_fields_preserved=True,mass_inertia_gravity_preserved=True,
        existing_drive_gains_damping_armature_force_limits_preserved=True,
        explicit_extensions=extensions,physics_hz=1000,piano_solref_time_s=.002,
        source_archive_sha256=model_source_sha))
    data=mujoco.MjData(model);scratch=mujoco.MjData(model)
    targets=np.zeros((len(original_t),model.nu))
    act_joints=model.actuator_trnid[:,0];qadr=model.jnt_qposadr[act_joints]
    aids={model.joint(j).name:i for i,j in enumerate(act_joints)}
    target_bases={side:[] for side in ('left','right')}
    max_anchor_error=0.;max_rotation_error=0.
    for k in range(len(original_t)):
        reference.qpos[original_qadr]=original_q[k];mujoco.mj_kinematics(original,reference)
        for side in ('left','right'):
            oldid=original.body(side+'__base_link').id;bid=base.body(side+'_/base_link').id
            R=base_data.xmat[bid].reshape(3,3);position=reference.xpos[oldid];rotation=reference.xmat[oldid].reshape(3,3)
            directions=np.column_stack([base_data.xaxis[base.joint(side+'_/palm_base_'+s).id] for s in ('tx','ty','tz')])
            slides=np.linalg.solve(directions,position-base_data.xpos[bid]);angles=Rotation.from_matrix(R.T@rotation).as_euler('XYZ')
            wrist=np.r_[slides,angles[0]*(1 if side=='right' else -1),-angles[1],angles[2]]
            for s,q in zip(('tx','ty','tz','pitch','yaw','roll'),wrist):targets[k,aids[side+'_/palm_base_'+s]]=q
            target_bases[side].append(np.r_[position,rotation.ravel()])
        for row in mapping:targets[k,aids[row['current']]]=row['sign']*original_q[k,original_names.index(row['original'])]
        scratch.qpos[qadr]=targets[k];mujoco.mj_kinematics(model,scratch)
        for row in mapping:
            nj=model.joint(row['current']).id;oj=original.joint(row['original']).id
            max_anchor_error=max(max_anchor_error,float(np.linalg.norm(scratch.xanchor[nj]-reference.xanchor[oj])))
        for side in ('left','right'):
            ni=model.body(side+'_/base_link').id;oi=original.body(side+'__base_link').id
            max_rotation_error=max(max_rotation_error,float(np.max(abs(scratch.xmat[ni]-reference.xmat[oi]))))
    if max_anchor_error>1e-7 or max_rotation_error>1e-7:raise ValueError(f'FK mapping failed {max_anchor_error} {max_rotation_error}')
    low=model.actuator_ctrlrange[:,0].copy();high=model.actuator_ctrlrange[:,1].copy()
    # The input is a measured trajectory, not a policy action. Preserve it.
    # Native actuator/joint/force limits remain in MuJoCo and are audited below;
    # do not reapply the policy MCP range, coupling or target-motion limiter.
    effective=np.clip(targets,low,high)
    mapping_report=dict(max_all_finger_anchor_error_m=max_anchor_error,max_base_rotation_matrix_error=max_rotation_error,joint_mapping=mapping,extensions=extensions,targets_outside_current_actuator_limits=int(np.count_nonzero(targets!=effective)),max_native_actuator_clamp=float(np.max(abs(targets-effective))),reference_modified=False,source_sample_hz=60,interpolation='linear interpolation of verified mapped joint position samples; final sample held to 17s',state_projection=False,finger_coupling='not applied: source supplies all 40 physical finger coordinates; no policy action mapping',wrist_target_limits='no additional trajectory rate limiter; native model joint/actuator/force limits retained')
    write(out/'mapping_report.json',mapping_report)
    np.savez_compressed(out/'mapped_input.npz',time=original_t,unclipped_target=targets,target=targets,actuator_names=np.array([model.actuator(i).name for i in range(model.nu)]),**{side+'_base_pose':np.array(v) for side,v in target_bases.items()})
    data.qpos[qadr]=targets[0];data.ctrl[:]=targets[0]
    # Initial key state is the resting current piano, never the saved key trace.
    data.qvel[:]=0
    from tsensei_v3_2_pianist.finger_resistance import Resistance
    resistance=Resistance(model);assert len(resistance.rows)==40
    key_joints=[j for j in range(model.njnt) if model.joint(j).name.startswith('piano/')]
    key_joints=sorted(key_joints,key=lambda j:int(model.joint(j).name.rsplit('_',1)[1]));assert len(key_joints)==88
    keyq=model.jnt_qposadr[key_joints];key_limits=model.jnt_range[key_joints]
    # Map all physical geoms attached to key bodies; count only force-carrying contacts.
    key_bodies={int(model.jnt_bodyid[j]):k for k,j in enumerate(key_joints)}
    key_geom={g:key_bodies[int(model.geom_bodyid[g])] for g in range(model.ngeom) if int(model.geom_bodyid[g]) in key_bodies}
    geom_side={g:0 if model.geom(g).name.startswith('right_') else 1 for g in range(model.ngeom) if (model.geom(g).name or '').startswith(('right_','left_'))}
    dt=float(model.opt.timestep);duration=min(a.duration,17.);times=np.arange(round(duration/dt)+1)*dt
    desired=np.column_stack([np.interp(times,original_t,targets[:,i]) for i in range(model.nu)])
    # Derivatives are estimates from the 60 Hz recording, used only for the
    # force feedforward. They never smooth, offset or retime the reference.
    sampled_velocity=np.gradient(targets,original_t,axis=0,edge_order=2)
    sampled_acceleration=np.gradient(sampled_velocity,original_t,axis=0,edge_order=2)
    reference_velocity=np.column_stack([np.interp(times,original_t,sampled_velocity[:,i]) for i in range(model.nu)])
    reference_acceleration=np.column_stack([np.interp(times,original_t,sampled_acceleration[:,i]) for i in range(model.nu)])
    reference_velocity[times>original_t[-1]]=0.;reference_acceleration[times>original_t[-1]]=0.
    kp=model.actuator_gainprm[:,0].copy();kd=-model.actuator_biasprm[:,2].copy()
    vadr=model.jnt_dofadr[act_joints]
    learned_commands=None;learning_source=None
    if a.learn_from:
        from scipy.signal import savgol_filter
        with np.load(a.learn_from,allow_pickle=False) as prior:
            np.testing.assert_array_equal(prior['time'],times)
            np.testing.assert_array_equal(prior['desired_ctrl'],desired)
            error=desired-prior['qpos'][:,qadr]
            # Correct repeatable tracking error through existing servo commands.
            # Filtering applies only to the correction, never to the reference.
            smoothed=savgol_filter(error,41,3,axis=0)
            derivative=savgol_filter(error,41,3,deriv=1,delta=dt,axis=0)
            correction=.8*(smoothed+derivative*(model.dof_damping[vadr]+kd)/kp)
            learned_commands=prior['ctrl']+correction
        learning_source=dict(trace=str(a.learn_from),sha256=sha(a.learn_from),gain=.8,
                             correction_window_seconds=.041,reference_unchanged=True)
    ffdata=mujoco.MjData(model);mass_acceleration=np.zeros(model.nv)
    full_mass=np.zeros((model.nv,model.nv))
    def command_for(k):
        if learned_commands is not None:return learned_commands[k].copy()
        if not a.feedforward:return desired[k].copy()
        ffdata.qpos[qadr]=desired[k];ffdata.qvel[:]=0.;ffdata.qvel[vadr]=reference_velocity[k]
        resistance.update(ffdata);mujoco.mj_forward(model,ffdata)
        acceleration=np.zeros(model.nv);acceleration[vadr]=reference_acceleration[k]
        mujoco.mj_mulM(model,ffdata,mass_acceleration,acceleration)
        # Only non-contact dynamics: contacts and joint limits stay in the
        # physical solver. Never supply inverse-constraint or external forces.
        torque=mass_acceleration+ffdata.qfrc_bias-ffdata.qfrc_passive
        torque[vadr]+=model.dof_frictionloss[vadr]*np.tanh(reference_velocity[k]/.02)
        command=desired[k]+(torque[vadr]+kd*reference_velocity[k])/kp
        if a.feedback:
            # An outer reference tracker, not a model/drive parameter change.
            # All force, position and joint limits remain in the native plant.
            mujoco.mj_fullM(model,ffdata,full_mass)
            extra_damping=np.maximum(0.,2*np.sqrt(4*kp*np.diag(full_mass)[vadr])-(model.dof_damping[vadr]+kd))
            command+=3*(desired[k]-data.qpos[qadr])+extra_damping*(reference_velocity[k]-data.qvel[vadr])/kp
        return command
    data.ctrl[:]=command_for(0)
    baseids=[model.body(side+'_/base_link').id for side in ('right','left')]
    tipnames=[side+'_/'+finger+'_tip' for side in ('right','left') for finger in ('thumb','index','mid','ring','pinky')]
    tipids=[model.site(n).id for n in tipnames]
    base_actual=[];base_reference=[];orientation_error=[];tip_actual=[];tip_reference=[]
    # Engineering display gate, declared before simulation. It is not a
    # hardware safety certificate or a guarantee of matching the source score.
    thresholds=dict(wrist_position_p95_m=.005,wrist_position_max_m=.015,
                    wrist_rotation_p95_rad=float(np.deg2rad(3)),
                    fingertip_position_p95_m=.005,fingertip_position_max_m=.015)
    write(out/'tracking_gate_definition.json',thresholds)
    Q=[];V=[];C=[];forces=[];ncontacts=[];cross=[];tracking=[];actuator_forces=[];start=time.monotonic()
    for k,t in enumerate(times):
        if k:
            data.ctrl[:]=command_for(k)
            resistance.update(data);mujoco.mj_step(model,data)
            if abs(data.time-t)>1e-8:raise RuntimeError('Physics time reset or drift')
        resistance.update(data);mujoco.mj_forward(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise RuntimeError('Nonfinite physics')
        # A separate kinematic scratch state defines reference geometry only.
        # It never supplies positions or forces to the physical simulation.
        scratch.qpos[qadr]=desired[k];mujoco.mj_kinematics(model,scratch)
        base_actual.append(data.xpos[baseids].copy());base_reference.append(scratch.xpos[baseids].copy())
        tip_actual.append(data.site_xpos[tipids].copy());tip_reference.append(scratch.site_xpos[tipids].copy())
        orientation_error.append([float(Rotation.from_matrix(scratch.xmat[b].reshape(3,3).T@data.xmat[b].reshape(3,3)).magnitude()) for b in baseids])
        force=np.zeros((2,88));cross_force=0.;cnum=0
        for ci in range(data.ncon):
            c=data.contact[ci]
            if c.efc_address<0 or c.dist>1e-6:continue
            g1,g2=int(c.geom1),int(c.geom2);cf=np.zeros(6);mujoco.mj_contactForce(model,data,ci,cf)
            if cf[0]<=1e-8:continue
            if g1 in geom_side and g2 in key_geom:force[geom_side[g1],key_geom[g2]]+=cf[0];cnum+=1
            elif g2 in geom_side and g1 in key_geom:force[geom_side[g2],key_geom[g1]]+=cf[0];cnum+=1
            elif g1 in geom_side and g2 in geom_side and geom_side[g1]!=geom_side[g2]:cross_force+=cf[0]
        Q.append(data.qpos.copy());V.append(data.qvel.copy());C.append(data.ctrl.copy());forces.append(force);ncontacts.append(cnum);cross.append(cross_force);tracking.append(data.qpos[qadr]-data.ctrl)
        actuator_forces.append(data.actuator_force.copy())
        if k%2000==0:print(json.dumps(dict(time_s=float(t),wall_s=time.monotonic()-start,active=int(np.sum(data.qpos[keyq]>=key_limits[:,1])),contacts=cnum,wrist_position_error_m=np.linalg.norm(base_actual[-1]-base_reference[-1],axis=-1).tolist(),max_tip_error_m=float(np.linalg.norm(tip_actual[-1]-tip_reference[-1],axis=-1).max()))),flush=True)
    Q=np.array(Q);V=np.array(V);C=np.array(C);forces=np.array(forces)
    if not a.feedforward and learned_commands is None:np.testing.assert_array_equal(C,desired)
    base_actual=np.asarray(base_actual);base_reference=np.asarray(base_reference)
    tip_actual=np.asarray(tip_actual);tip_reference=np.asarray(tip_reference)
    orientation_error=np.asarray(orientation_error)
    def stats(x):
        return dict(rms=float(np.sqrt(np.mean(x*x))),p95=float(np.percentile(abs(x),95)),maximum=float(abs(x).max()))
    wrist_errors=np.linalg.norm(base_actual-base_reference,axis=-1)
    tip_errors=np.linalg.norm(tip_actual-tip_reference,axis=-1)
    checks=dict(wrist_position_p95_m=float(np.max(np.percentile(wrist_errors,95,axis=0))),
                wrist_position_max_m=float(wrist_errors.max()),
                wrist_rotation_p95_rad=float(np.max(np.percentile(orientation_error,95,axis=0))),
                fingertip_position_p95_m=float(np.max(np.percentile(tip_errors,95,axis=0))),
                fingertip_position_max_m=float(tip_errors.max()))
    tracking_report=dict(thresholds=thresholds,measurements=checks,
        passed=all(checks[n]<=v for n,v in thresholds.items()),reference_modified=False,
        feedforward_applied=a.feedforward,reference_derivatives='60 Hz central finite differences, used only for feedforward; not measured source velocities',
        learned_feedforward_source=learning_source,
        outer_feedback=a.feedback,
        command_reference_max_error=float(abs(C-desired).max()),
        native_actuator_clamp_max=float(abs(C-np.clip(C,low,high)).max()),
        native_actuator_clamp_samples=int(np.count_nonzero(C!=np.clip(C,low,high))),
        wrist={side:dict(position_m=stats(wrist_errors[:,i]),orientation_rad=stats(orientation_error[:,i])) for i,side in enumerate(('right','left'))},
        fingertip={n:stats(tip_errors[:,i]) for i,n in enumerate(tipnames)},
        joint={model.actuator(i).name:stats(Q[:,qadr[i]]-desired[:,i]) for i in range(model.nu)},
        reference_definition='source joint samples transformed by verified FK, linearly interpolated at unchanged timestamps; current fingertip task sites',
        scope=f'entire {duration:g} s requested recording including initial transient; no phase alignment')
    write(out/'tracking_validation.json',tracking_report)
    write(out/'target_rate_validation.json',dict(rate_limits_applied=False,reference_preservation_passed=True,command_reference_max_error=float(abs(C-desired).max()),feedforward_applied=a.feedforward,note='Reference is unchanged. Commands may compensate current drive inertia/damping/friction. Native physical limits remain.'))
    actual=Q[:,keyq]>=key_limits[:,1];mujoco.mj_saveModel(model,str(out/'model.mjb'))
    np.savez_compressed(out/'physical_trace.npz',time=times,qpos=Q,qvel=V,ctrl=C,desired_ctrl=desired,key_qpos=Q[:,keyq],key_limits=key_limits,actual_keys=actual,key_activation_fraction=1.,hand_key_normal_force=forces,hand_key_contact_count=ncontacts,cross_hand_normal_force=cross,tracking_error=tracking,physics_dt=dt,base_actual=base_actual,base_reference=base_reference,orientation_error=orientation_error,tip_actual=tip_actual,tip_reference=tip_reference,actuator_force=actuator_forces)
    violations=np.maximum(model.jnt_range[:,0]-Q,0)+np.maximum(Q-model.jnt_range[:,1],0)
    report=dict(controller='position trajectory replay through current PD/friction/contact dynamics',physics_stepped=True,robot_state_projection=False,key_qpos_driven=False,source_checkpoint_in_package=False,source_selected_update=24250,source_archive_sha256=model_source_sha,model_sha256=sha(out/'model.mjb'),script_sha256=sha(__file__),duration_s=float(times[-1]),physics_hz=1/dt,source_hz=60.,source_recorded_preparation_s=.5,original_midi_offset_s=.5,initial_state='mapped first recorded hand position; zero initial velocity; resting current piano',num_hand_key_contact_samples=int(np.count_nonzero(np.array(ncontacts))),max_hand_key_force_n=float(forces.max()),activated_key_count=int(actual.any(0).sum()),active_physical_samples=int(actual.sum()),max_joint_limit_violation=float(violations.max()),warnings=[dict(name=mujoco.mjtWarning(i).name,count=int(w.number),lastinfo=int(w.lastinfo)) for i,w in enumerate(data.warning)],dependency_versions={'mujoco':mujoco.__version__,'numpy':np.__version__},mapping=mapping_report,wall_seconds=time.monotonic()-start)
    report['tracking_validation']=tracking_report
    report['controller']='unchanged current position servos, with non-contact dynamics feedforward' if a.feedforward else report['controller']
    if learned_commands is not None:report['controller']='unchanged current position servos, with iterative feedforward tracking correction; reference unchanged'
    if a.feedback:report['controller']='unchanged capped current position servos, with dynamics feedforward and outer trajectory-error feedback; reference unchanged'
    report['process']=dict(pid=os.getpid(),hostname=os.uname().nodename,command=sys.argv,exit_code=0)
    write(out/'replay_report.json',report);print('PHYSICS_DONE',flush=True)
    write(out/'process.json',dict(**report['process'],status='completed'))

if __name__=='__main__':main()
