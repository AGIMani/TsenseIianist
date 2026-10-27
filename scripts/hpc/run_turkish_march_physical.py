#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""Physical MuJoCo replay driven by MIDI fingering assignments.

Unlike the kinematic diagnostic, this runner never writes piano-key qpos.  The
hand position actuators receive 20 Hz targets and MuJoCo advances the scene at
200 Hz; key travel, contact forces, and activation are all produced by
``mujoco.mj_step``.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import mujoco

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--score', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--duration', type=float, default=10.)
    ap.add_argument('--original-midi', type=Path, help='Unquantized original events for exact-time evaluation')
    ap.add_argument('--build', type=Path, help='Reuse a verified scene/task build for diagnosis; does not change its model')
    ap.add_argument('--render', action='store_true', help='Render only inside Slurm')
    a=ap.parse_args()
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):
        ap.error('Model loading and physics must run on volc-a32, including short tests; local execution is disabled')
    if a.render and not os.environ.get('SLURM_JOB_ID'):
        ap.error('--render requires a Slurm allocation via scripts/hpc/submit.sh')
    out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(out)
    if a.build is None:
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build as build_source
        recipe=make_recipe(midi=str(a.score), world_x=.14, world_z=.10, palm_base_yaw=np.pi/6,
            control_hz=20., physics_hz=200., lookahead=10, frame_stack=1,
            initial_buffer_time=0., release_tail_time=0., score_validation='legacy',
            gravity_compensation=True, action_reward_observation=True,
            disable_forearm_reward=False, task_profile='tsensei_collision_v1',
            collision_profile='convex_hull_v1', drive_profile='shadow_v1',
            preparation_time=0., exact_note_timing=False)
        recipe['external_score']['name']=a.score.name
        (out/'source_config.json').write_text(json.dumps(recipe,indent=2)+'\n')
        # The public builder sets the recipe's pose/model environment in a
        # compiler subprocess and verifies source hashes. Calling _compile in
        # this process bypassed that setup and could compile a different pose.
        import shutil
        generated_scene, generated_task, _ = build_source(recipe)
        build=out/'build'; build.mkdir()
        shutil.copytree(generated_scene,build/'scene')
        shutil.copytree(generated_task,build/'task')
        shutil.copy2(generated_scene.parent/'score.proto',build/'score.proto')
    else:
        build=a.build.resolve()
    embedded=build/'score.proto'
    if not embedded.exists() or sha(embedded) != sha(a.score):
        raise ValueError('Build score differs from --score; rebuild for this exact fingering')
    scene=build/'scene'; task=build/'task'
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    bundle=SceneBundle(scene)
    model=bundle.model; data=bundle.reset_cpu()
    manifest=json.loads((task/'task.json').read_text()); score=np.load(task/'score.npz',allow_pickle=False)
    fingers=score['fingers']; control_dt=float(manifest['control_dt']); physics_dt=float(manifest['physics_dt'])
    duration=min(float(a.duration),fingers.shape[0]*control_dt); total=int(round(duration/physics_dt)); substeps=int(round(control_dt/physics_dt))
    sites=list(manifest['fingertip_sites']); site_ids=[model.site(n).id for n in sites]
    qids=[]; actuator_ids=[]; qpos_to_joint={int(model.jnt_qposadr[j]):j for j in range(model.njnt)}
    actuator_by_qadr={int(model.jnt_qposadr[int(model.actuator_trnid[aid,0])]):aid for aid in range(model.nu) if int(model.actuator_trnid[aid,0])>=0}
    for side, hand in zip(('right','left'), manifest['hand_joints']):
        names=list(hand)
        joints=[model.joint(n).id for n in names if model.joint(n).id>=0]
        qids.append([int(model.jnt_qposadr[j]) for j in joints]); actuator_ids.append([actuator_by_qadr[q] for q in qids[-1]])
    key_geom_ids=[model.geom(n).id for n in manifest['key_geoms']]
    base_key_rgba=model.geom_rgba[key_geom_ids].copy()
    key_joint_ids=[]
    for gid in key_geom_ids:
        body=int(model.geom_bodyid[gid]); key_joint_ids.append(next(j for j in range(model.njnt) if int(model.jnt_bodyid[j])==body))
    finger_geom_ids=[]; fg_to_f={}
    for f,sid in enumerate(site_ids):
        body=int(model.site_bodyid[sid]); gs=[i for i in range(model.ngeom) if int(model.geom_bodyid[i])==body and '_collision_' in model.geom(i).name]
        finger_geom_ids.append(gs); [fg_to_f.__setitem__(g,f) for g in gs]
    key_to_k={g:k for k,g in enumerate(key_geom_ids)}
    # Include every collision link of each finger in same-hand diagnostics,
    # while fingertip-to-key success remains restricted to terminal links.
    all_finger_geoms={}
    for gid in range(model.ngeom):
        name=model.geom(gid).name or ''
        if '_collision_' not in name:
            continue
        for side, prefix in enumerate(('right_', 'left_')):
            if name.startswith(prefix):
                for local, finger in enumerate(('thumb', 'index', 'mid', 'ring', 'pinky')):
                    if '_'+finger+'_' in name:
                        all_finger_geoms[gid]=side*5+local
    finger_pair_trace=[]; finger_pair_force_trace=[]; full_contact_trace=[]
    from tsensei_v3_2_pianist.fingering_targets import target_position, target_for_key, THUMB_GLOBAL_INDICES
    q0=bundle.reset_qpos.copy()
    # Preserve the exported initial pose, position servos, limits and contacts.
    mujoco.mj_forward(model,data)
    # Use current key geometry as the alignment surface.  Keys remain fully
    # physical afterwards; only hand actuator targets are generated here.
    target_ctrl=data.ctrl.copy(); frames=[]; actual_keys=[]; assigned_contacts=[]; assigned_forces=[]; qtrace=[]; vtrace=[]; ctrls=[]
    renderer=mujoco.Renderer(model,height=720,width=1280) if a.render else None
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    camera=mujoco.MjvCamera(); camera_settings=configure_camera(camera,model,view='front',framing='full',distance=1.05,elevation=-30)
    # A separate scratch state lets the IK compute actuator targets without
    # teleporting the physical state used by mj_step.
    ik=mujoco.MjData(model)
    def update_targets(ci, lift=0.0):
        """Solve a hand pose from the physical state and issue only ctrl targets."""
        ik.qpos[:]=data.qpos; ik.qvel[:]=0; mujoco.mj_forward(model,ik); mujoco.mj_forward(model,data)
        for side in (0,1):
            active_f=[f for f in range(side*5,side*5+5) if np.any(fingers[ci,f]>0)]
            # Wrist commands are needed for changing register; the physical
            # state is moved by its original actuators, never by writing qpos.
            joint_qids=qids[side]
            joint_aids=actuator_ids[side]
            cols=[int(model.jnt_dofadr[qpos_to_joint[q]]) for q in joint_qids]
            # First align the fingers to their MIDI-assigned keys.  The
            # resulting qpos is only converted to actuator targets.
            released_targets=ik.site_xpos[site_ids].copy()
            released_targets[:,2]+=0.025 if lift == 0.0 else 0.018
            for _ in range(80):
                mujoco.mj_forward(model,ik); blocks=[]; tb=[]
                for f in range(side*5, side*5+5):
                    if f in active_f:
                        keys=np.flatnonzero(fingers[ci,f]>0); k=int(keys[0]); gid=key_geom_ids[k]
                        target=target_for_key(model, ik, gid,thumb=f in THUMB_GLOBAL_INDICES,task_profile='tsensei_collision_v1')
                        target=np.asarray(target); target[2]+=lift
                        if lift == 0.0:
                            # A shallow penetration lets the real contact
                            # solver load the key without injecting a large
                            # impulse into the hand.
                            target[2]-=0.003
                    else:
                        # Release fingers that have no MIDI assignment on this
                        # control frame.  Keeping the previous target would
                        # leave an old key physically held and inflate false
                        # positives in the actual-key trace.
                        target=released_targets[f]
                    jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,ik,jp,jr,site_ids[f]); blocks.append(jp[:,cols]); tb.append(target-ik.site_xpos[site_ids[f]])
                J=np.vstack(blocks); err=np.concatenate(tb)
                if np.linalg.norm(err)<1e-6: break
                dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(J.shape[0]),err)
                for q,dqv in zip(joint_qids,dq):
                    jid=qpos_to_joint[q]; ik.qpos[q]=np.clip(ik.qpos[q]+.18*dqv,model.jnt_range[jid,0],model.jnt_range[jid,1])
            # Once the hand is in the neighbourhood, refine the actuator
            # target from the actual closest points of the fingertip hull and
            # assigned key.  This makes the subsequent key motion come from
            # contact dynamics rather than from a manually chosen endpoint.
            if lift == 0.0:
                for _ in range(48):
                    mujoco.mj_forward(model,ik); blocks=[]; tb=[]
                    for f in active_f:
                        k=int(np.flatnonzero(fingers[ci,f]>0)[0]); ft=np.zeros(6,dtype=np.float64)
                        dist=mujoco.mj_geomDistance(model,ik,finger_geom_ids[f][0],key_geom_ids[k],1.0,ft)
                        if dist <= 1e-5: continue
                        jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,ik,jp,jr,site_ids[f]); blocks.append(jp[:,cols]); tb.append(np.clip(ft[3:]-ft[:3],-0.025,0.025))
                    if not blocks: break
                    J=np.vstack(blocks); err=np.concatenate(tb); dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(J.shape[0]),err)
                    for q,dqv in zip(joint_qids,dq):
                        jid=qpos_to_joint[q]; ik.qpos[q]=np.clip(ik.qpos[q]+.12*dqv,model.jnt_range[jid,0],model.jnt_range[jid,1])
            for q,aid in zip(joint_qids,joint_aids): target_ctrl[aid]=ik.qpos[q]
    # Dynamic preparation gives both hands time to align above the first
    # assigned keys.  The lift keeps this phase from pressing any key; the
    # score clock and all reported frames start only after preparation.
    preparation_s=2.0; preparation_steps=int(round(preparation_s/physics_dt)); ctrl_start=data.ctrl.copy()
    for pre in range(preparation_steps):
        if pre % substeps == 0:
            ctrl_start=data.ctrl.copy(); update_targets(0, lift=.018)
        alpha=float((pre % substeps)+1)/substeps
        data.ctrl[:]=ctrl_start + alpha*(target_ctrl[:model.nu]-ctrl_start); mujoco.mj_step(model,data)
    for step in range(total+1):
        t=step*physics_dt; ci=min(step//substeps,fingers.shape[0]-1)
        if step % substeps == 0:
            ctrl_start=data.ctrl.copy(); update_targets(ci, lift=0.0)
        alpha=float((step % substeps)+1)/substeps
        data.ctrl[:]=ctrl_start + alpha*(target_ctrl[:model.nu]-ctrl_start)
        # Sample at exact state time t, before advancing [t,t+dt).
        # mj_forward evaluates current-state contacts with matching qpos.
        mujoco.mj_forward(model,data)
        # Key state comes from the dynamic qpos after mj_step, never from the
        # MIDI command.  Contacts and normal forces are collected at 200 Hz.
        active=np.zeros(88,dtype=bool); lohi=[]
        for k,jid in enumerate(key_joint_ids):
            adr=int(model.jnt_qposadr[jid]); lo,hi=model.jnt_range[jid]; active[k]=np.clip(data.qpos[adr],lo,hi)>=hi*float(bundle.manifest['key_activation_fraction']); lohi.append(float(data.qpos[adr]))
            model.geom_rgba[key_geom_ids[k]]=np.array([0.08,.9,.2,1.],dtype=np.float32) if active[k] else base_key_rgba[k]
        actual_keys.append(active)
        assign=fingers[ci]>0; contacts=np.zeros((10,88),dtype=bool); forces=np.zeros((10,88),dtype=float)
        finger_pairs=np.zeros((10,10),dtype=bool); finger_pair_forces=np.zeros((10,10))
        for i in range(int(data.ncon)):
            c=data.contact[i]; g1,g2=int(c.geom1),int(c.geom2); f=fg_to_f.get(g1); k=key_to_k.get(g2)
            if c.efc_address < 0 or float(c.dist)>1e-6:
                continue
            cf=np.zeros(6); mujoco.mj_contactForce(model,data,i,cf)
            if float(cf[0]) <= 1e-8:
                continue
            f1,f2=all_finger_geoms.get(g1),all_finger_geoms.get(g2)
            if f1 is not None and f2 is not None and f1 != f2 and f1//5 == f2//5:
                low,high=sorted((f1,f2)); finger_pairs[low,high]=True
                finger_pair_forces[low,high]+=float(cf[0])
            if f is None or k is None: f=fg_to_f.get(g2); k=key_to_k.get(g1)
            # Physical contact requires a solver constraint and normal force.
            if f is None or k is None or c.efc_address < 0 or float(c.dist)>1e-6: continue
            cf=np.zeros(6); mujoco.mj_contactForce(model,data,i,cf)
            if float(cf[0]) <= 1e-8: continue
            contacts[f,k]=True; forces[f,k]+=float(cf[0])
        assigned_contacts.append(contacts & assign); assigned_forces.append(forces)
        full_contact_trace.append(contacts); finger_pair_trace.append(finger_pairs)
        finger_pair_force_trace.append(finger_pair_forces)
        qtrace.append(data.qpos.copy()); vtrace.append(data.qvel.copy()); ctrls.append(data.ctrl.copy())
        if renderer is not None and step%5==0:
            renderer.update_scene(data,camera=camera); frames.append(renderer.render().copy())
        if step < total:
            mujoco.mj_step(model,data)
    import imageio.v2 as imageio
    if renderer is not None:
        renderer.close()
        imageio.mimsave(out/'turkish_march_first10s_physical.mp4',frames,fps=40,codec='libx264',quality=7); imageio.imwrite(out/'frame_0000.png',frames[0])
    actual=np.asarray(actual_keys); target=np.asarray([fingers[min(int(round(t/physics_dt))//substeps,fingers.shape[0]-1)].sum(axis=0)>0 for t in np.arange(total+1)*physics_dt]); tp=int(np.sum(actual&target)); fp=int(np.sum(actual&~target)); fn=int(np.sum(~actual&target)); f1=2*tp/max(2*tp+fp+fn,1)
    ac=np.asarray(assigned_contacts); af=np.asarray(assigned_forces); assign=(np.asarray([fingers[min(int(round(t/physics_dt))//substeps,fingers.shape[0]-1)]>0 for t in np.arange(total+1)*physics_dt])); ctp=int(np.sum(ac)); cfn=int(np.sum(assign & ~ac)); recall=ctp/max(ctp+cfn,1)
    key_qadr=model.jnt_qposadr[key_joint_ids]
    np.savez_compressed(out/'physical_trace.npz',time=np.arange(total+1)*physics_dt,qpos=np.asarray(qtrace),qvel=np.asarray(vtrace),ctrl=np.asarray(ctrls),actual_keys=actual,target_keys=target,assigned_contact=ac,contact_force=af,
                        key_qpos=np.asarray(qtrace)[:,key_qadr],key_limits=model.jnt_range[key_joint_ids],
                        key_activation_fraction=float(bundle.manifest['key_activation_fraction']),
                        finger_key_contact=np.asarray(full_contact_trace),finger_pair_contact=np.asarray(finger_pair_trace),
                        finger_pair_force=np.asarray(finger_pair_force_trace))
    report={'kind':'midi_fingering_physical_mujoco_replay_v1','score':str(a.score.resolve()),'score_sha256':sha(a.score),'duration_s':duration,'control_hz':1/control_dt,'physics_hz':1/physics_dt,'frames':total+1,'frame_time_semantics':'qpos and contacts at k * physics_dt; terminal sample included','key_tp':tp,'key_fp':fp,'key_fn':fn,'key_f1':f1,'acceptance_passed':False,'assigned_contact_tp':ctp,'assigned_contact_fn':cfn,'assigned_contact_recall':recall,'contact_force_max_N':float(af.max(initial=0)),'contact_force_sum_Ns':float(af.sum()*physics_dt),'camera':camera_settings,'physical_key_qpos_written_by_script':False,'physical_stepping':'mujoco.mj_step; key travel and contact forces are dynamic','actuator_parameters_modified':False,'initial_state_source':'SceneBundle.reset_cpu','preparation_s':preparation_s,'key_activation_fraction':float(bundle.manifest['key_activation_fraction']),'scene_manifest_sha256':sha(scene/'manifest.json'),'task_manifest_sha256':sha(task/'task.json'),'runner_sha256':sha(__file__),'score_reference':'20Hz control grid; not original MIDI exact-time F1','contact_definition':'c.efc_address >= 0 and c.dist <= 1e-6','limitations':'Actuator-driven MuJoCo replay, not a learned policy or hardware trajectory.'}
    fpairs=np.asarray(finger_pair_trace)
    report['index_middle_contact_samples']={'right':int(fpairs[:,1,2].sum()),'left':int(fpairs[:,6,7].sum())}
    report['assigned_finger_and_key_active_samples']=int((ac & actual[:,None,:]).sum())
    if a.original_midi:
        # Parse original note times independently of the control-grid score.
        import pretty_midi
        midi=pretty_midi.PrettyMIDI(str(a.original_midi))
        events=np.array([[n.start,n.end,n.pitch] for inst in midi.instruments for n in inst.notes])
        times=np.arange(total+1)*physics_dt
        exact=np.zeros_like(actual)
        for start,end,pitch in events:
            if 21 <= int(pitch) <= 108:
                exact[:,int(pitch)-21] |= (times>=start) & (times<end)
        etp=int((actual&exact).sum()); efp=int((actual&~exact).sum()); efn=int((~actual&exact).sum())
        report['original_midi_exact_time']={'sha256':sha(a.original_midi),'tp':etp,'fp':efp,'fn':efn,
                                           'f1':2*etp/max(2*etp+efp+efn,1)}
        np.savez_compressed(out/'original_midi_comparison.npz',time=times,actual=actual,target=exact,original_events=events)
    (out/'replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    import shutil; shutil.copy2(scene/'scene.xml',out/'scene.xml'); shutil.copy2(scene/'manifest.json',out/'scene_manifest.json'); shutil.copy2(task/'task.json',out/'task.json'); shutil.copy2(task/'score.npz',out/'score.npz')
    (out/'build_reference.json').write_text(json.dumps({'build':str(build.resolve()),'reused_build':a.build is not None,'assets_required':str(scene.resolve()),'standalone_scene':False},indent=2)+'\n')
    print(json.dumps({'output':str(out),'video':str(out/'turkish_march_first10s_physical.mp4') if a.render else None,'key_f1':f1,'acceptance_passed':False,'assigned_contact_recall':recall,'contact_force_max_N':float(af.max(initial=0))},indent=2))

if __name__=='__main__': main()
