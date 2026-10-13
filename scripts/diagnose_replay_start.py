#!/usr/bin/env python3
"""Reapply recorded controls on GPU and inspect real contacts, without changing training."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import warp as wp
from mujoco_warp._src.support import contact_force
from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
from tsensei_v3_2_pianist.mjlab_env.bundle import sha256


def saved_motion_stats(scene, rollout):
 b=scene.bundle;dt=b.manifest['control_dt'];m=scene.model;aid=b.action_ids;jid=m.actuator_trnid[aid,0];qadr=m.jnt_qposadr[jid]
 names=b.manifest['action_names'][:-1];initial=b.reset_qpos[qadr];q=rollout['qpos'][:,qadr];u=rollout['ctrl'][:,aid]
 first=[]
 for i,name in enumerate(names):
  first.append(dict(name=name,initial=float(initial[i]),first_target=float(u[0,i]),
                    first_actual=float(q[0,i]),initial_error=float(u[0,i]-initial[i]),
                    initial_unclipped_servo_force=float(m.actuator_gainprm[aid[i],0]*(u[0,i]-initial[i])),
                    force_limit=float(m.actuator_forcerange[aid[i],1]),
                    unit='m' if m.jnt_type[jid[i]]==mujoco.mjtJoint.mjJNT_SLIDE else 'rad'))
 differences=np.diff(np.concatenate((initial[None],q)),axis=0)
 intervals=[]
 for start,end in [(0,1),(1,3),(3,7.9)]:
  x=differences[round(start/dt):round(end/dt)]
  intervals.append(dict(start=start,end=end,mean_abs_finger_step_deg=float(np.degrees(abs(x[:,[i for i in range(50) if i%25<20]]).mean())),mean_abs_wrist_translation_step_mm=float(abs(x[:,[20,21,22,45,46,47]]).mean()*1000)))
 return dict(first_action=first,control_dt=dt,mean_motion_per_control_step=intervals)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=60)
 p.add_argument('--sample-every-substeps',type=int,default=10,help='Contact sampling stride; 1 inspects every 0.5 ms physics step')
 a=p.parse_args()
 if a.output.exists():p.error('Output exists')
 if a.sample_every_substeps < 1:p.error('sample-every-substeps must be positive')
 r=np.load(a.recording/'rollout.npz')
 if not 1<=a.steps<=len(r['actions']):p.error('steps must be between 1 and the recorded episode length')
 metadata=json.loads((a.recording/'report.json').read_text())
 from tsensei_v3_2_pianist.mjlab_env.paths import for_recording
 scene_path,_=for_recording(metadata)
 scene=PianoScene(scene_path,action_profile=metadata.get('action_profile','legacy'));b=scene.bundle;m=scene.model
 names=[m.geom(i).name or '' for i in range(m.ngeom)];jointids=m.actuator_trnid[b.action_ids,0];qadr=m.jnt_qposadr[jointids]
 def category(pair):
  sides=[n.split('/')[0] if n.startswith(('right_/','left_/')) else 'piano_key' if 'key_geom' in n else 'other' for n in pair]
  if sides[0]==sides[1] and sides[0] in ('left_','right_'):return 'self_collision'
  if set(sides)=={'right_','left_'}:return 'between_hands'
  if 'piano_key'in sides:return 'hand_key'
  return 'hand_other'
 def run(label,actions):
  scene.reset();records={};tick=0;states=[]
  def capture(s):
   nonlocal tick
   tick+=1
   if tick%a.sample_every_substeps:return
   wd=s.sim.wp_data;n=int(wd.nacon.numpy()[0]);geom=wd.contact.geom.numpy()[:n];dist=wd.contact.dist.numpy()[:n]
   if n:
    ids=wp.array(np.arange(n,dtype=np.int32),dtype=wp.int32,device=str(scene.device));forces=wp.empty(n,dtype=wp.spatial_vector,device=str(scene.device));contact_force(s.sim.wp_model,wd,ids,False,forces);f=forces.numpy()
    for j,(g1,g2) in enumerate(geom):
     pair=(names[g1],names[g2])
     if not any(name.startswith(('right_/','left_/')) for name in pair) or dist[j]>0:continue
     key=' | '.join(sorted(pair));time=tick*b.manifest['physics_dt']
     row=records.setdefault(key,dict(pair=pair,category=category(pair),first_time=time,min_dist=0.,max_normal_force=0.,force_peak_time=time,samples=0))
     row['min_dist']=min(row['min_dist'],float(dist[j]));row['samples']+=1
     if f[j,0]>row['max_normal_force']:row['max_normal_force']=float(f[j,0]);row['force_peak_time']=time
   states.append(dict(time=tick*b.manifest['physics_dt'],qpos=s.data.qpos[0,qadr].cpu().tolist(),qvel=s.data.qvel[0,m.jnt_dofadr[jointids]].cpu().tolist()))
  positions=[]
  for action in actions:
   scene.step(action[None],substep_callback=capture);positions.append(scene.data.qpos[0].cpu().numpy().copy())
  return dict(label=label,pairs=sorted(records.values(),key=lambda x:x['max_normal_force'],reverse=True),samples=states),np.asarray(positions)
 result,q=run('saved_video_actions_open_loop',r['actions'][:a.steps])
 neutral,_=run('neutral_control_baseline',np.tile(scene.policy_action(b.neutral_action).cpu().numpy(),(round(.5/b.manifest['control_dt']),1)))
 saved=r['qpos'][:a.steps];qerror=abs(q-saved)
 # Exact saved trajectory: geometry overlap only, no dynamic force reconstruction.
 data=b.reset_cpu();offline=[]
 for i in range(-1,a.steps):
  data.qpos[:]=b.reset_qpos if i<0 else r['qpos'][i];mujoco.mj_forward(m,data)
  for c in data.contact:
   pair=[names[int(g)] for g in c.geom]
   if c.dist<0 and any(n.startswith(('right_/','left_/')) for n in pair):offline.append(dict(time=(i+1)*b.manifest['control_dt'],pair=pair,category=category(pair),distance=float(c.dist)))
 report=dict(recording=str(a.recording.resolve()),sampling_interval_s=a.sample_every_substeps*b.manifest['physics_dt'],steps=a.steps,
             note='GPU force from actual integrated substeps; re-applied saved actions, not a new policy evaluation. Contact sampling interval is recorded in sampling_interval_s. Offline geometry uses exact saved video qpos, no inferred forces.',
             open_loop_max_slide_error=float(qerror[:,np.where(m.jnt_type==mujoco.mjtJoint.mjJNT_SLIDE)[0]].max()),
             open_loop_max_hinge_error=float(qerror[:,np.where(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)[0]].max()),
             gpu=result,neutral=neutral,exact_video_geometry=offline,saved_motion=saved_motion_stats(scene,r))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('gpu','neutral','exact_video_geometry')},indent=2))
 print('TOP_GPU_CONTACTS',json.dumps(result['pairs'][:12],indent=2));print('NEUTRAL',json.dumps(neutral['pairs'][:8],indent=2))

if __name__=='__main__':main()
