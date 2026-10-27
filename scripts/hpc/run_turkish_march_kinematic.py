#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""Render a first-10-second, finger-assignment-driven MuJoCo replay.

This is a kinematic diagnostic: it solves fingertip IK toward the registered
score's assigned keys and visually presses active keys. It is not a policy or
hardware trajectory.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--score', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--duration', type=float, default=10.0)
    a=ap.parse_args(); out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(out)
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, _compile
    recipe=make_recipe(midi=str(a.score), world_x=.14, world_z=.10, palm_base_yaw=np.pi/6,
        control_hz=20., physics_hz=200., lookahead=10, frame_stack=1,
        initial_buffer_time=0., release_tail_time=0., score_validation='legacy',
        gravity_compensation=True, action_reward_observation=True,
        disable_forearm_reward=False, task_profile='tsensei_collision_v1',
        collision_profile='convex_hull_v1', drive_profile='shadow_v1',
        preparation_time=0., exact_note_timing=False)
    recipe['external_score']['name']=a.score.name
    recipe_path=out/'source_config.json'; recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    build=out/'build'; build.mkdir()
    _compile(recipe_path, build)
    scene=build/'scene'; task=build/'task'
    model=mujoco.MjModel.from_xml_path(str(scene/'scene.xml')); data=mujoco.MjData(model)
    manifest=json.loads((task/'task.json').read_text())
    score=np.load(task/'score.npz',allow_pickle=False)
    fingers=score['fingers']; control_dt=float(manifest['control_dt']); physics_dt=float(manifest['physics_dt'])
    duration=min(float(a.duration), fingers.shape[0]*control_dt)
    sites=list(manifest['fingertip_sites']); site_ids=[model.site(n).id for n in sites]
    # Manifest order is right five sites followed by left five sites.
    qids=[]
    for side, hand in zip(('right','left'), manifest['hand_joints']):
        names=list(hand)+[f'{side}_/palm_base_tx',f'{side}_/palm_base_ty',f'{side}_/palm_base_tz',f'{side}_/palm_base_pitch',f'{side}_/palm_base_yaw']
        qids.append([model.joint(n).id for n in names if model.joint(n).id >= 0])
    qpos_ids=[]; dof_ids=[]
    qpos_to_joint={int(model.jnt_qposadr[j]):j for j in range(model.njnt)}
    for hand in qids:
        qpos_ids.append([int(model.jnt_qposadr[j]) for j in hand])
        dof_ids.append([int(model.jnt_dofadr[j]) for j in hand])
    key_geom_ids=[model.geom(n).id for n in manifest['key_geoms']]
    base_key_rgba=model.geom_rgba[key_geom_ids].copy()
    key_joint_ids=[]
    for gid in key_geom_ids:
        body=int(model.geom_bodyid[gid]); found=None
        for jid in range(model.njnt):
            if int(model.jnt_bodyid[jid])==body: found=jid; break
        key_joint_ids.append(found)
    # fingertip target uses the same geometric contract as the task.
    from tsensei_v3_2_pianist.fingering_targets import target_position, target_for_key, THUMB_GLOBAL_INDICES
    q0=model.qpos0.copy(); data.qpos[:]=q0; mujoco.mj_forward(model,data)
    rest_sites=data.site_xpos[site_ids].copy()
    key_targets=np.zeros((10,88,3))
    for f in range(10):
        for k,gid in enumerate(key_geom_ids):
            key_targets[f,k]=target_for_key(model, data, gid,
                thumb=f in THUMB_GLOBAL_INDICES, task_profile='tsensei_collision_v1')
    # The collision geom attached to each fingertip is the authoritative
    # contact primitive.  We keep the target-point IK as a warm start, then
    # solve it tightly enough that the assigned fingertip actually intersects
    # its assigned key geometry.  Key activation alone is not accepted as
    # evidence of a finger press.
    finger_geom_ids=[]
    finger_geom_to_index={}
    for f,sid in enumerate(site_ids):
        body=int(model.site_bodyid[sid])
        geoms=[i for i in range(model.ngeom)
               if int(model.geom_bodyid[i])==body and '_collision_' in model.geom(i).name]
        if not geoms:
            raise RuntimeError(f'No fingertip collision geom for {sites[f]}')
        finger_geom_ids.append(geoms)
        for gid in geoms: finger_geom_to_index[gid]=f
    key_geom_to_index={gid:k for k,gid in enumerate(key_geom_ids)}
    finger_contact=[]; contact_tp=contact_fp=contact_fn=0
    rows=[]; frames=[]; errors=[]; actual_keys=[]
    total=int(round(duration/physics_dt)); render_stride=5
    renderer=mujoco.Renderer(model,height=720,width=1280)
    # Keep the replay identical while using the project's canonical front
    # camera so that the piano spans the frame and the finger-to-key gap is
    # visible.  Camera settings affect rendering only, never the MuJoCo state.
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    camera = mujoco.MjvCamera()
    camera_settings = configure_camera(camera, model, view='front', framing='full',
                                       distance=1.05, elevation=-30)
    for step in range(total+1):
        t=step*physics_dt; ci=min(int(t/control_dt),fingers.shape[0]-1)
        active=fingers[ci].sum(axis=1)>0
        for side in (0,1):
            active_f=[f for f in range(side*5,side*5+5) if np.any(fingers[ci,f]>0)]
            if not active_f: continue
            cols=dof_ids[side]
            # More damped iterations are intentional: the old 24-step solve
            # left long-span notes hovering above the key.  This contact-first
            # solve is still kinematic, but it materially raises the signed
            # MuJoCo finger/key contact rate without changing the score.
            for _ in range(200):
                mujoco.mj_forward(model,data)
                blocks=[]; target_blocks=[]
                for f in active_f:
                    sid=site_ids[f]; jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,jp,jr,sid)
                    blocks.append(jp[:,cols])
                    keys=np.flatnonzero(fingers[ci,f]>0)
                    target_blocks.append(key_targets[f,int(keys[0])])
                J=np.vstack(blocks); err=np.concatenate(target_blocks)-np.concatenate([data.site_xpos[site_ids[f]] for f in active_f])
                if float(np.linalg.norm(err)) < 1e-7: break
                dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(J.shape[0]),err)
                for j,d in zip(qpos_ids[side],dq):
                    jid=qpos_to_joint[j]; data.qpos[j]=np.clip(data.qpos[j]+.15*d,model.jnt_range[jid,0],model.jnt_range[jid,1])
        mujoco.mj_forward(model,data)
        # Show score-active keys in their pressed position; this is visualized
        # kinematic state, while the scene XML remains the original physical model.
        for k,jid in enumerate(key_joint_ids):
            if jid is None: continue
            key_on=bool(fingers[ci,:,k].sum()>0)
            qadr=int(model.jnt_qposadr[jid]); lo,hi=model.jnt_range[jid]
            target=lo + (.82*(hi-lo) if key_on else .02*(hi-lo))
            # Smooth key travel at 200 Hz; the rendered green state is read
            # from this interpolated qpos, never from the target command.
            data.qpos[qadr] += .8*(target-data.qpos[qadr])
        mujoco.mj_forward(model,data)
        # Contact refinement is the actual press stage.  It uses the closest
        # points of the fingertip collision hull and the assigned key, rather
        # than a hand-authored target point.  The preceding IK pass merely
        # brings the finger into the key's neighbourhood so this refinement
        # remains stable when several fingers share one wrist.
        for side in (0,1):
            active_f=[f for f in range(side*5,side*5+5) if np.any(fingers[ci,f]>0)]
            if not active_f: continue
            cols=dof_ids[side]
            for _ in range(48):
                mujoco.mj_forward(model,data)
                blocks=[]; target_blocks=[]
                for f in active_f:
                    keys=np.flatnonzero(fingers[ci,f]>0)
                    if len(keys)==0: continue
                    k=int(keys[0]); ft=np.zeros(6,dtype=np.float64)
                    dist=mujoco.mj_geomDistance(model,data,finger_geom_ids[f][0],key_geom_ids[k],1.0,ft)
                    if dist <= 1e-5: continue
                    sid=site_ids[f]; jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,jp,jr,sid)
                    blocks.append(jp[:,cols]); target_blocks.append(np.clip(ft[3:]-ft[:3],-0.025,0.025))
                if not blocks: break
                J=np.vstack(blocks); err=np.concatenate(target_blocks)
                dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(J.shape[0]),err)
                for j,d in zip(qpos_ids[side],dq):
                    jid=qpos_to_joint[j]; data.qpos[j]=np.clip(data.qpos[j]+.12*d,model.jnt_range[jid,0],model.jnt_range[jid,1])
        mujoco.mj_forward(model,data)
        active_now=np.zeros(88,dtype=bool)
        for k,jid in enumerate(key_joint_ids):
            if jid is None: continue
            lo,hi=model.jnt_range[jid]; q=float(data.qpos[int(model.jnt_qposadr[jid])])
            active_now[k]=abs(np.clip(q,lo,hi)-hi)<=float(manifest.get('key_tolerance',.003))
            gid=key_geom_ids[k]
            model.geom_rgba[gid]=np.array([0.08,.9,.2,1.],dtype=np.float32) if active_now[k] else base_key_rgba[k]
        actual_keys.append(active_now)
        assignment=fingers[ci]>0
        contact_now=np.zeros((10,88),dtype=bool)
        # Signed distance is measured after key interpolation, so this test
        # cannot be satisfied by a target command or by a green key alone.
        for f in np.flatnonzero(assignment.any(axis=1)):
            for k in np.flatnonzero(assignment[f]):
                dist=min(mujoco.mj_geomDistance(model,data,fg,key_geom_ids[int(k)],1.0,None)
                         for fg in finger_geom_ids[int(f)])
                contact_now[int(f),int(k)]=dist <= 1e-5
        target_pairs=assignment
        contact_tp += int(np.sum(contact_now & target_pairs))
        contact_fn += int(np.sum((~contact_now) & target_pairs))
        # False-positive finger/key contacts are read from MuJoCo's contact
        # list.  They are counted only for actual penetrating/touching pairs;
        # broad-phase proximity is not treated as a press.
        for ci_contact in range(int(data.ncon)):
            c=data.contact[ci_contact]; g1,g2=int(c.geom1),int(c.geom2)
            f=finger_geom_to_index.get(g1); k=key_geom_to_index.get(g2)
            if f is None or k is None:
                f=finger_geom_to_index.get(g2); k=key_geom_to_index.get(g1)
            if f is not None and k is not None and float(c.dist) <= 1e-5 and not target_pairs[f,k]:
                contact_fp += 1
        finger_contact.append(contact_now)
        errmax=0.
        for f in range(10):
            keys=np.flatnonzero(fingers[ci,f]>0)
            if len(keys):
                target=key_targets[f,int(keys[0])]
                errmax=max(errmax,float(np.linalg.norm(target-data.site_xpos[site_ids[f]])))
        errors.append(errmax)
        if step%render_stride==0:
            renderer.update_scene(data, camera=camera); frames.append(renderer.render().copy())
        if step%10==0: rows.append({'time_s':t,'control_frame':ci,'active_keys':[int(k) for k in np.flatnonzero(active)],'max_tip_error_m':errmax})
    import imageio.v2 as imageio
    imageio.mimsave(out/'turkish_march_first10s.mp4',frames,fps=40,codec='libx264',quality=7)
    imageio.imwrite(out/'frame_0000.png',frames[0])
    actual_arr=np.asarray(actual_keys); target_arr=np.asarray([fingers[min(int(t/control_dt),fingers.shape[0]-1)].sum(axis=0)>0 for t in np.arange(total+1)*physics_dt])
    tp=int(np.sum(actual_arr&target_arr)); fp=int(np.sum(actual_arr&~target_arr)); fn=int(np.sum(~actual_arr&target_arr)); f1=2*tp/max(2*tp+fp+fn,1)
    finger_contact_recall=contact_tp/max(contact_tp+contact_fn,1)
    finger_contact_precision=contact_tp/max(contact_tp+contact_fp,1)
    finger_contact_f1=2*contact_tp/max(2*contact_tp+contact_fp+contact_fn,1)
    np.savez_compressed(out/'kinematic_trace.npz',time=np.arange(total+1)*physics_dt,
                        max_tip_error_m=np.asarray(errors),actual_keys=actual_arr,
                        target_keys=target_arr,finger_contact=np.asarray(finger_contact))
    (out/'replay_report.json').write_text(json.dumps({'kind':'kinematic_fingering_contact_diagnostic_v4','score':str(a.score.resolve()),'score_sha256':sha(a.score),'duration_s':duration,'control_hz':20.,'physics_hz':200.,'frames':total+1,'max_tip_error_m':float(max(errors)),'mean_tip_error_m':float(np.mean(errors)),'key_tp':tp,'key_fp':fp,'key_fn':fn,'key_f1':f1,'finger_contact_tp':contact_tp,'finger_contact_fp':contact_fp,'finger_contact_fn':contact_fn,'assigned_finger_contact_recall':finger_contact_recall,'finger_contact_precision':finger_contact_precision,'finger_contact_f1':finger_contact_f1,'smooth_key_interpolation':True,'green_active_key_render':True,'camera':camera_settings,'limitations':'Two-stage contact-first kinematic replay: target-point IK is only a warm start, then closest-point signed MuJoCo finger/key geometry drives the press refinement; not a learned policy or hardware trajectory.'},indent=2)+'\n')
    import shutil; shutil.copy2(scene/'scene.xml',out/'scene.xml'); shutil.copy2(scene/'manifest.json',out/'scene_manifest.json'); shutil.copy2(task/'task.json',out/'task.json'); shutil.copy2(task/'score.npz',out/'score.npz')
    print(json.dumps({'output':str(out),'video':str(out/'turkish_march_first10s.mp4'),'max_tip_error_m':float(max(errors))},indent=2))

if __name__=='__main__': main()
