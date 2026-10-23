#!/usr/bin/env python3
"""Non-learning, actuator-only piano trajectory search and physical validation.

All model work must run remotely. The scratch IK state is never the physical
state. Original MIDI times and supplied finger assignments remain independent
of the training control grid.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fk(model, data):
    # Jacobians need comPos, but IK does not need collision/force solving.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def read_events(score, midi):
    code = '''import json,sys,pretty_midi
from tsensei_v3_2_pianist.music.midi_file import MidiFile
seq=MidiFile.from_file(sys.argv[1]).seq
original=pretty_midi.PrettyMIDI(sys.argv[2])
print(json.dumps(dict(events=sorted((n.start_time,n.end_time,n.pitch-21,n.part) for n in seq.notes),
 raw=sorted((n.start,n.end,n.pitch-21) for inst in original.instruments for n in inst.notes))))
'''
    data=json.loads(subprocess.check_output([str(ROOT/'training/.venv/bin/python'),'-c',code,str(score),str(midi)],text=True))
    return np.array(data['events'],float),np.array(data['raw'],float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--score', type=Path, required=True)
    ap.add_argument('--original-midi', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build', type=Path)
    ap.add_argument('--duration', type=float, default=1000)
    ap.add_argument('--control-hz', type=float, default=20.)
    ap.add_argument('--prepare', type=float, default=2.)
    ap.add_argument('--lead', type=float, default=.05)
    ap.add_argument('--release-lead', type=float, default=.035)
    ap.add_argument('--hover', type=float, default=.012)
    ap.add_argument('--idle-height', type=float, default=.045)
    ap.add_argument('--depth', type=float, default=.006)
    ap.add_argument('--finger-press-z-weight', type=float, default=1.,
                    help='Non-thumb vertical press weight; horizontal alignment retains unit weight')
    ap.add_argument('--anticipation', type=float, default=.20)
    ap.add_argument('--iterations', type=int, default=35)
    ap.add_argument('--collision-weight', type=float, default=0.)
    ap.add_argument('--solver', choices=['damped','bounded'], default='bounded')
    ap.add_argument('--finger-ff', type=float, default=0.)
    ap.add_argument('--wrist-ff', type=float, default=0.)
    ap.add_argument('--free-key-x', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--white-front-min', type=float, default=.022,
                    help='Minimum longitudinal landing coordinate for white keys, in metres')
    ap.add_argument('--tracking-gain', type=float, default=0.)
    ap.add_argument('--nonthumb-tracking-gain', type=float, default=0.,
                    help='Extra measured-position feedback in the eight non-thumb actuator targets')
    ap.add_argument('--note-adjustments', type=Path)
    ap.add_argument('--wrist-reference', type=Path)
    ap.add_argument('--actual-wrist-reference', type=Path,
                    help='Recorded physical wrists used for offline future-pose IK alignment')
    ap.add_argument('--issued-wrist-reference', type=Path,
                    help='Reuse a recorded wrist command sequence without smoothing or shifting')
    ap.add_argument('--wrist-lead', type=float, default=.15)
    ap.add_argument('--wrist-smoothing', type=float, default=.12)
    ap.add_argument('--wrist-prediction', type=float, default=.03)
    ap.add_argument('--alignment-gate', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--alignment-tolerance', type=float, default=.006)
    ap.add_argument('--smooth-reference', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--wrist-regularization', type=float, default=.02)
    ap.add_argument('--idle-clearance', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--contact-objective', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--contact-penetration', type=float, default=.002)
    ap.add_argument('--future-weight', type=float, default=.6)
    ap.add_argument('--tip-angle-weight', type=float, default=0.)
    ap.add_argument('--park-thumbs', action='store_true',
                    help='Require a score without thumb assignments and use an explicit parking pose')
    ap.add_argument('--thumb-parking-pose', type=Path,
                    help='JSON mapping all eight thumb joint names to physical radians')
    ap.add_argument('--initialize-thumb-parking', action='store_true',
                    help='Use the parking pose as the episode initial thumb configuration')
    ap.add_argument('--thumb-idle-pose', type=Path,
                    help='Explicit initial-pose reference for inactive thumbs; permits thumb notes')
    ap.add_argument('--initialize-thumb-idle', action='store_true',
                    help='Use the idle thumb reference at episode reset, before physical preparation')
    ap.add_argument('--thumb-idle-weight', type=float, default=.005)
    ap.add_argument('--thumb-key-clearance-weight', type=float, default=0.)
    ap.add_argument('--thumb-landing-weight', type=float, default=0.)
    ap.add_argument('--thumb-collision-weight', type=float, default=1.)
    ap.add_argument('--thumb-cross-clearance-weight', type=float, default=0.)
    ap.add_argument('--thumb-depth', type=float, default=.008)
    ap.add_argument('--thumb-top-approach', action='store_true')
    ap.add_argument('--thumb-press-seeds', type=Path,
                    help='Static candidate thumb configurations used only to initialize scratch IK')
    ap.add_argument('--top-approach', action='store_true',
                    help='Lift above the key tops before transferring to a different key')
    ap.add_argument('--coupling', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--coupled-ik', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--build-only', action='store_true')
    a = ap.parse_args()
    if a.actual_wrist_reference is not None and not (a.wrist_reference or a.issued_wrist_reference):
        ap.error('--actual-wrist-reference requires a wrist command reference')
    if a.wrist_reference is not None and a.issued_wrist_reference is not None:
        ap.error('Choose either planned or issued wrist commands')
    if a.park_thumbs != (a.thumb_parking_pose is not None):
        ap.error('--park-thumbs requires --thumb-parking-pose; the old inward parking pose is unsafe')
    if a.initialize_thumb_parking and not a.park_thumbs:
        ap.error('--initialize-thumb-parking requires --park-thumbs')
    if a.thumb_idle_pose and a.park_thumbs:
        ap.error('Choose active-thumb idle reference or fully parked thumbs')
    if a.initialize_thumb_idle and not a.thumb_idle_pose:
        ap.error('--initialize-thumb-idle requires --thumb-idle-pose')
    if a.thumb_idle_pose and a.solver!='bounded':
        ap.error('Thumb idle reference requires bounded IK')
    if not 0<a.finger_press_z_weight<=1:
        ap.error('--finger-press-z-weight must be in (0,1]')
    if not 0<=a.nonthumb_tracking_gain<=4:
        ap.error('--nonthumb-tracking-gain must be in [0,4]')
    if not (os.environ.get('SSH_CONNECTION') or os.environ.get('SLURM_JOB_ID')):
        ap.error('Model compilation and physics require remote execution')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'arguments.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                               for k,v in vars(a).items()}, indent=2))
    if a.build is None:
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        recipe = make_recipe(midi=str(a.score.resolve()), height=.06, ty=.04,
            yaw=np.pi/6, control_hz=20, physics_hz=200,
            initial_buffer_time=0, release_tail_time=.25, score_validation='legacy',
            gravity_compensation=True, task_profile='tsensei_collision_v1',
            collision_profile='convex_hull_v1', drive_profile='shadow_v1')
        (out/'source_config.json').write_text(json.dumps(recipe, indent=2))
        scene, task, _ = build(recipe)
        bdir = out/'build'; bdir.mkdir()
        shutil.copytree(scene, bdir/'scene'); shutil.copytree(task, bdir/'task')
        shutil.copy2(a.score, bdir/'score.proto')
    else:
        bdir = a.build.resolve()
    if sha(bdir/'score.proto') != sha(a.score):
        raise ValueError('Mismatched build score')
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    bundle = SceneBundle(bdir/'scene', disable_wrist_tz=True, disable_wrist_yaw=True)
    m = bundle.model; d = bundle.reset_cpu(); ik = bundle.reset_cpu()
    control_dt=1/a.control_hz
    if a.control_hz<=0 or not np.isclose(control_dt/m.opt.timestep,round(control_dt/m.opt.timestep)):
        raise ValueError('Controller period must be an integer number of physics steps')
    task = json.loads((bdir/'task/task.json').read_text())
    # Independent exact-time reference, retaining the original leading silence.
    events,raw = read_events(a.score,a.original_midi)
    if a.park_thumbs and np.any(np.isin(events[:,3], [0,5])):
        raise ValueError('Thumb parking requires an explicitly revised score without thumb notes')
    adjustments=np.zeros((len(events),4))
    thumb_press_seeds={}
    if a.thumb_press_seeds:
        for row in json.loads(a.thumb_press_seeds.read_text())['selected']:
            key=(round(row['start_s'],8),round(row['end_s'],8),row['pitch'],row['part'])
            thumb_press_seeds[key]=np.asarray(row['static_seed']['q'],float)
    if a.note_adjustments:
        adjustments=np.asarray(json.loads(a.note_adjustments.read_text())['adjustments'],float)
        if adjustments.shape!=(len(events),4):raise ValueError('Wrong note adjustment dimensions')
    ordered = events[np.lexsort((events[:,2], events[:,1], events[:,0])),:3]
    if raw.shape != ordered.shape or not np.allclose(raw, ordered, atol=1e-8, rtol=0):
        raise ValueError('Score must preserve original MIDI note times and pitches')
    ids = np.array([m.site(n).id for n in task['fingertip_sites']])
    keygeom = np.array([m.geom(n).id for n in task['key_geoms']])
    finger_geoms = {g:f for f,sid in enumerate(ids) for g in range(m.ngeom)
                   if m.geom_bodyid[g] == m.site_bodyid[sid] and '_collision_' in m.geom(g).name}
    geomkeys = {int(g):k for k,g in enumerate(keygeom)}
    hand_geoms = {g:(0 if m.geom(g).name.startswith('right_') else 1)
                  for g in range(m.ngeom) if '_collision_' in m.geom(g).name
                  and m.geom(g).name.startswith(('right_', 'left_'))}
    hand_geom_rows={g:i for i,g in enumerate(hand_geoms)}
    from scripts.hpc.piano_ik_geometry import (thumb_meshes, lower_support, keyboard_clearance, landing_clearance,
        hand_collision_meshes, thumb_cross_pairs, cross_clearance)
    import scripts.hpc.piano_ik_geometry as geometry_module
    geometry_source=Path(geometry_module.__file__).read_bytes()
    (out/'geometry_helper.py').write_bytes(geometry_source)
    thumb_collision_meshes=thumb_meshes(m)
    full_hand_meshes=hand_collision_meshes(m)
    thumb_opposite_pairs=thumb_cross_pairs(m,full_hand_meshes)
    aid = bundle.action_ids
    fullids = bundle.active_action_indices
    joints = m.actuator_trnid[aid,0]
    qids = m.jnt_qposadr[joints]; vids = m.jnt_dofadr[joints]
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
    low, high = physical_bounds('direct_joint_v1', bundle.manifest['action_names'], bundle.low, bundle.high)
    low, high = np.array(low), np.array(high)
    qlow, qhigh = low[fullids], high[fullids]
    from tsensei_v3_2_pianist.mjlab_env.joint_limits import resolve_robot_limits
    joint_contract=resolve_robot_limits(m,key_ids=bundle.key_ids,action_ids=aid,
        active_indices=fullids,physical_low=low,physical_high=high)
    stopq=np.array([r['qpos_address'] for r in joint_contract['joints']])
    stopv=np.array([r['dof_address'] for r in joint_contract['joints']])
    stoplo=np.array([r['float32_min'] for r in joint_contract['joints']])
    stophi=np.array([r['float32_max'] for r in joint_contract['joints']])
    stopcounts=np.zeros(len(stopq),dtype=int)
    def enforce_stops():
        # Exact CPU equivalent of joint_limits_warp.project; keys excluded.
        old=d.qpos[stopq].copy();new=np.clip(old,stoplo,stophi);speed=d.qvel[stopv]
        outward=((new<=stoplo)&(speed<0))|((new>=stophi)&(speed>0))
        changed=(old!=new)|outward
        d.qpos[stopq]=new;d.qvel[stopv[outward]]=0
        d.qacc_warmstart[stopv[changed]]=0;stopcounts[:]+=changed
    enforce_stops()
    rest = d.qpos.copy(); resttips = d.site_xpos[ids].copy()
    parked_columns = []
    parking_pose = {}
    if a.park_thumbs or a.thumb_idle_pose:
        pose_path=a.thumb_parking_pose or a.thumb_idle_pose
        parking_pose=json.loads(pose_path.read_text())['joints_rad']
        expected={m.joint(j).name for j in joints if '_thumb_' in m.joint(j).name}
        if set(parking_pose)!=expected:raise ValueError('Parking pose must name all eight thumb joints')
        for col,j in enumerate(joints):
            name=m.joint(j).name
            if '_thumb_' in name:
                if a.park_thumbs:parked_columns.append(col)
                value=float(parking_pose[name])
                if not np.isfinite(value) or not qlow[col]<=value<=qhigh[col]:
                    raise ValueError(f'Parking pose outside physical bounds: {name}')
                rest[qids[col]]=value
                if a.initialize_thumb_parking or a.initialize_thumb_idle:d.qpos[qids[col]]=value
        if a.initialize_thumb_parking or a.initialize_thumb_idle:
            enforce_stops()
            mujoco.mj_forward(m,d)
    keypos = d.geom_xpos[keygeom].copy(); keysize = m.geom_size[keygeom].copy()
    distal={}
    for f,sid in enumerate(ids):
        if f%5==0:continue
        side='right' if f<5 else 'left';name=('thumb','index','mid','ring','pinky')[f%5]
        j=m.joint(f'{side}_/{side}_hand_{name}_DIP_joint').id
        direction=m.site_pos[sid]-m.jnt_pos[j]
        distal[f]=(int(m.site_bodyid[sid]),direction/np.linalg.norm(direction))
    targets = keypos.copy(); targets[:,0] += .35*keysize[:,0]
    targets[:,2] += keysize[:,2]
    coupling = None
    if a.coupling:
        import torch
        torch.set_num_threads(1)
        from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
        opts = dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False,
                    angle_ranges_deg=json.loads((ROOT/'configs/finger_coupling/angle_ranges_60_80.json').read_text()))
        coupling = FingerCoupler(bundle, device='cpu', options=opts, physical_low=low, physical_high=high)
    independent=np.arange(len(qids))
    if coupling is not None and a.coupled_ik:
        followers=set(coupling.indices[:,1:].flatten().numpy().tolist())
        independent=np.array([i for i,j in enumerate(fullids) if int(j) not in followers])
    diagnostic = dict(nq=m.nq,nu=m.nu, joints=[m.joint(j).name for j in joints],
        rest_tips=resttips.tolist(), key_centers=keypos.tolist(), key_half_sizes=keysize.tolist(),
        qpos=rest[qids].tolist(), ranges=np.stack((qlow,qhigh),1).tolist(),
        physics_dt=m.opt.timestep, score_sha256=sha(a.score), original_midi_sha256=sha(a.original_midi),
        build=str(bdir), source_runner_sha256=sha(__file__), coupling_enabled=a.coupling,
        disabled_wrist_axes=['TZ','Yaw'], physical_collision_response=True)
    diagnostic['joint_state_contract']=joint_contract
    diagnostic['controlled_qpos_addresses']=qids.tolist()
    diagnostic['controlled_actuator_ids']=aid.tolist()
    diagnostic['initial_physical_controlled_qpos']=d.qpos[qids].tolist()
    diagnostic['ik_reference_controlled_qpos']=rest[qids].tolist()
    diagnostic['thumb_parking_enabled']=a.park_thumbs
    diagnostic['thumb_parking_actuator_columns']=parked_columns
    diagnostic['initial_thumb_pose']=parking_pose if a.initialize_thumb_parking or a.initialize_thumb_idle else {}
    if a.thumb_parking_pose:diagnostic['thumb_parking_pose_sha256']=sha(a.thumb_parking_pose)
    if a.thumb_idle_pose:
        diagnostic['thumb_idle_pose_sha256']=sha(a.thumb_idle_pose)
        diagnostic['thumb_idle_reference']=parking_pose
        diagnostic['thumb_idle_weight']=a.thumb_idle_weight
    diagnostic['thumb_key_clearance_weight']=a.thumb_key_clearance_weight
    diagnostic['thumb_landing_weight']=a.thumb_landing_weight
    diagnostic['thumb_collision_weight']=a.thumb_collision_weight
    diagnostic['thumb_cross_clearance_weight']=a.thumb_cross_clearance_weight
    diagnostic['geometry_helper_sha256']=hashlib.sha256(geometry_source).hexdigest()
    diagnostic['thumb_top_approach_enabled']=a.thumb_top_approach
    diagnostic['thumb_idle_return']='lift_terminal_clear_then_initial' if a.thumb_top_approach and a.thumb_idle_pose else 'direct_reference'
    diagnostic['thumb_depth']=a.thumb_depth
    diagnostic['finger_press_z_weight']=a.finger_press_z_weight
    diagnostic['nonthumb_tracking_gain']=a.nonthumb_tracking_gain
    if a.thumb_press_seeds:diagnostic['thumb_press_seeds_sha256']=sha(a.thumb_press_seeds)
    diagnostic['white_front_min']=a.white_front_min
    for name in ('actual_wrist_reference','issued_wrist_reference'):
        path=getattr(a,name)
        if path is not None:diagnostic[name+'_sha256']=sha(path)
    (out/'model_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
    if a.build_only:
        return

    # Use the same final wrist target limiter as training.
    from tsensei_v3_2_pianist.mjlab_env.target_motion_limits import TargetMotionLimiter
    names=bundle.manifest['action_names']; rows={}
    for i,n in enumerate(names):
        axis=n.rsplit('_',1)[-1]
        if axis in ('Tx','Ty','Pitch'):
            rows[n]=dict(index=i,min=low[i],max=high[i],max_velocity=.99 if axis!='Pitch' else 3.1416,
                         max_acceleration=1.99 if axis!='Pitch' else 15.708,
                         position_unit='m' if axis!='Pitch' else 'rad',velocity_unit='',acceleration_unit='')
    import torch
    limiter=TargetMotionLimiter(dict(control_dt=control_dt,channels=rows),num_envs=1,device='cpu',
        reset_action=bundle.neutral_action,physical_low=low,physical_high=high)
    indexed_events=np.c_[events,np.arange(len(events))]
    fnotes=[indexed_events[events[:,3]==f] for f in range(10)]
    duration=min(a.duration,float(events[:,1].max())+.25)
    dt=float(m.opt.timestep); stride=round(control_dt/dt)
    previous=rest[qids].copy()
    previous_plan=previous.copy()
    ff=np.array([a.wrist_ff if 'forearm_' in m.joint(j).name else a.finger_ff for j in joints])
    tracking=np.array([a.tracking_gain+(a.nonthumb_tracking_gain
                       if '_thumb_' not in m.joint(j).name and 'forearm_' not in m.joint(j).name else 0.)
                       for j in joints])
    wrist_cols=np.array([i for i,j in enumerate(joints) if 'forearm_' in m.joint(j).name])
    wrist_reference=None
    actual_wrist_reference=None
    issued_wrist_reference=None
    if a.actual_wrist_reference:
        actual_ref=np.load(a.actual_wrist_reference,allow_pickle=False)
        actual_ref_time=actual_ref['time']
        actual_wrist_reference=actual_ref['qpos'][:,qids[wrist_cols]]
    if a.issued_wrist_reference:
        issued_ref=np.load(a.issued_wrist_reference,allow_pickle=False)
        issued_ref_time=issued_ref['all_command_times']
        issued_wrist_reference=issued_ref['all_commands'][:,fullids[wrist_cols]]
    if a.wrist_reference:
        from scipy.ndimage import gaussian_filter1d
        ref=np.load(a.wrist_reference,allow_pickle=False)
        ref_time=ref['all_command_times']
        wrist_reference=gaussian_filter1d(ref['all_plans'][:,wrist_cols],
            max(a.wrist_smoothing/float(np.median(np.diff(ref_time))),.01),axis=0,mode='nearest')
    max_delta=np.array([.012 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else .12 for j in joints])
    tiperr=[]; records=[]; qtrace=[]; vtrace=[]; ctrltrace=[]; contacttrace=[]; forcetrace=[]; collisiontrace=[]
    cmdtrace=[]; times=[]; wanted_trace=[]; ik_tip_trace=[]; actual_tip_trace=[]
    all_command_times=[];all_commands=[];all_plans=[]
    approach_keys=np.full(10,-1,dtype=int)
    approach_phase=['align']*10
    approach_counts=dict(lift=0,align=0,ready=0,release_lift=0)
    previous_thumb_event={0:None,5:None}
    last_thumb_key={0:None,5:None}
    start_wall=time.monotonic()
    for step in range(-round(a.prepare/dt), round(duration/dt)+1):
        t=step*dt
        if step % stride == 0:
            mujoco.mj_forward(m,d)
            actual_q=torch.tensor(d.qpos[None],dtype=torch.float32)
            actual_x=torch.tensor(d.xmat[None],dtype=torch.float32)
            ik.qpos[:]=rest
            ik.qpos[qids]=previous_plan
            solve_low=qlow.copy();solve_high=qhigh.copy()
            if a.park_thumbs:
                # Bounds constrain only scratch IK targets, never physical state.
                for col in parked_columns:
                    value=np.clip(rest[qids[col]],qlow[col]+2e-8,qhigh[col]-2e-8)
                    solve_low[col]=value-1e-8;solve_high[col]=value+1e-8
            wrist_command=None
            if wrist_reference is not None or issued_wrist_reference is not None:
                if issued_wrist_reference is not None:
                    ix=int(np.clip(np.searchsorted(issued_ref_time,t+1e-8,side='right')-1,0,len(issued_ref_time)-1))
                    wrist_command=issued_wrist_reference[ix].copy()
                else:
                    wrist_command=np.array([np.interp(t+a.wrist_lead,ref_time,wrist_reference[:,i]) for i in range(len(wrist_cols))])
                predicted=np.clip(d.qpos[qids[wrist_cols]]+a.wrist_prediction*d.qvel[vids[wrist_cols]],
                                  qlow[wrist_cols]+1e-7,qhigh[wrist_cols]-1e-7)
                if actual_wrist_reference is not None and t+a.wrist_prediction>=actual_ref_time[0]:
                    predicted=np.array([np.interp(t+a.wrist_prediction,actual_ref_time,actual_wrist_reference[:,i])
                                        for i in range(len(wrist_cols))])
                    predicted=np.clip(predicted,qlow[wrist_cols]+1e-7,qhigh[wrist_cols]-1e-7)
                solve_low[wrist_cols]=predicted-1e-8;solve_high[wrist_cols]=predicted+1e-8
                ik.qpos[qids[wrist_cols]]=predicted
            fk(m,ik)
            wanted=ik.site_xpos[ids].copy(); weights=np.zeros((10,3)); allowed=set(); selected={}
            selected_events={}
            for f,notes in enumerate(fnotes):
                if a.park_thumbs and f in (0,5):
                    continue
                changes=adjustments[notes[:,4].astype(int)]
                now=notes[(notes[:,0]<=t+a.lead+changes[:,0]) & (notes[:,1]>t+a.release_lead+changes[:,1])]
                future=notes[notes[:,0]>t+a.lead+changes[:,0]]
                if len(now):
                    ni=int(now[-1,4]);k=int(now[-1,2]); wanted[f]=targets[k]
                    wanted[f,2]-=(a.thumb_depth if f in (0,5) and a.thumb_idle_pose else a.depth)+adjustments[ni,3]
                    wanted[f,1]+=adjustments[ni,2]
                    weights[f]=[1.,1.,a.finger_press_z_weight if f%5 else 1.]
                    aligned=abs(d.site_xpos[ids[f],1]-keypos[k,1])<=min(a.alignment_tolerance,.8*keysize[k,1])
                    if a.alignment_gate and not aligned:
                        wanted[f,2]=targets[k,2]+a.hover
                    else:
                        allowed.add((f,k))
                    selected[f]=k
                    selected_events[f]=ni
                elif len(future) and future[0,0] <= max(0,t)+a.anticipation:
                    ni=int(future[0,4]);k=int(future[0,2]); wanted[f]=targets[k]; wanted[f,2]+=a.hover
                    wanted[f,1]+=adjustments[ni,2]
                    weights[f]=[a.future_weight,a.future_weight,a.future_weight]
                    selected[f]=k
                    selected_events[f]=ni
                else:
                    wanted[f,2]=a.idle_height
                    weights[f]=[0,0,.2]
                    if a.thumb_idle_pose and f in (0,5):weights[f]=0
            releasing_thumbs=set()
            if a.thumb_top_approach and a.thumb_idle_pose:
                for f in (0,5):
                    if f in selected:
                        last_thumb_key[f]=selected[f]
                        continue
                    k=last_thumb_key[f]
                    if k is None:continue
                    terminal=[g for g in thumb_collision_meshes
                              if m.geom_bodyid[g]==m.site_bodyid[ids[f]]]
                    bottom=min(lower_support(d,g,thumb_collision_meshes[g])[2] for g in terminal)
                    safe_height=float(np.max(targets[:,2]))+.006
                    if bottom>=safe_height:
                        last_thumb_key[f]=None
                        continue
                    # Keep the measured lateral position while lifting the
                    # whole terminal mesh clear; only then seek the idle pose.
                    wanted[f]=d.site_xpos[ids[f]]
                    wanted[f,2]+=safe_height+.006-bottom
                    weights[f]=1.
                    selected[f]=k
                    releasing_thumbs.add(f)
                    approach_counts['release_lift']+=1
            for f in (0,5):
                ni=selected_events.get(f)
                if ni is not None and ni!=previous_thumb_event[f]:
                    s,e,k,_=events[ni]
                    key=(round(float(s),8),round(float(e),8),int(k)+21,f)
                    if key in thumb_press_seeds:
                        side='right_' if f==0 else 'left_'
                        cols=[col for col,j in enumerate(joints)
                              if m.joint(j).name.startswith(side) and '_thumb_' in m.joint(j).name]
                        ik.qpos[qids[cols]]=np.clip(thumb_press_seeds[key],qlow[cols],qhigh[cols])
                previous_thumb_event[f]=ni
            fk(m,ik)
            lifting=set(releasing_thumbs)
            if a.top_approach or a.thumb_top_approach:
                clearance_height=float(np.max(targets[:,2]))+.006
                hover_height=max(clearance_height+.006,float(np.max(targets[:,2]))+a.hover)
                for f,k in selected.items():
                    if f in releasing_thumbs:continue
                    if not a.top_approach and f not in (0,5):continue
                    pos=d.site_xpos[ids[f]]
                    if approach_keys[f]!=k:
                        approach_keys[f]=k
                        approach_phase[f]='lift' if pos[2]<clearance_height else 'align'
                    if approach_phase[f]=='lift' and pos[2]>=clearance_height:
                        approach_phase[f]='align'
                    xmin=keypos[k,0]-.8*keysize[k,0];xmax=keypos[k,0]+.8*keysize[k,0]
                    if 'white' in m.geom(int(keygeom[k])).name:xmin=max(xmin,a.white_front_min)
                    aligned=(xmin-.001<=pos[0]<=xmax+.001 and
                             abs(pos[1]-keypos[k,1])<=.75*keysize[k,1])
                    if approach_phase[f]=='align' and aligned:
                        approach_phase[f]='ready'
                    phase=approach_phase[f];approach_counts[phase]+=1
                    if phase!='ready':
                        allowed.discard((f,k))
                        wanted[f,2]=hover_height
                        weights[f]=1.
                        if phase=='lift':
                            # Hold the measured lateral location until the
                            # physical fingertip clears the black-key tops.
                            wanted[f,:2]=pos[:2]
                            lifting.add(f)
            if a.contact_objective:
                for f,k in allowed:
                    # Desired depressed keys belong only to the scratch IK.
                    ik.qpos[bundle.key_qadr[k]]=m.jnt_range[bundle.key_ids[k],1]
                fk(m,ik)
            if a.solver=='bounded' and coupling is None:
                from scipy.optimize import least_squares
                regularization=np.full(len(qids),.0005)
                reference=rest[qids]
                if a.smooth_reference:
                    reference=.98*previous_plan+.02*rest[qids]
                    regularization[:]=.002
                    regularization[wrist_cols]=a.wrist_regularization
                    for col,j in enumerate(joints):
                        name=m.joint(j).name
                        for f,fname in enumerate(('thumb','index','mid','ring','pinky')*2):
                            side='right' if f<5 else 'left'
                            if name.startswith(side+'_') and '_'+fname+'_' in name and f not in selected:
                                reference[col]=rest[qids[col]];regularization[col]=.003
                if a.thumb_idle_pose:
                    for col,j in enumerate(joints):
                        name=m.joint(j).name
                        part=0 if name.startswith('right_') else 5
                        if '_thumb_' in name and part not in selected:
                            reference[col]=rest[qids[col]]
                            regularization[col]=a.thumb_idle_weight
                permitted_thumb_terminals={int(m.site_bodyid[ids[f]]) for f,k in allowed if f in (0,5)}
                def thumb_clearance(with_jac=False):
                    return a.thumb_key_clearance_weight*keyboard_clearance(
                        m,ik,thumb_collision_meshes,keypos,keysize,
                        permitted_terminal_bodies=permitted_thumb_terminals,
                        dofs=vids if with_jac else None)
                def thumb_cross_clearance(with_jac=False):
                    return a.thumb_cross_clearance_weight*cross_clearance(
                        m,ik,full_hand_meshes,thumb_opposite_pairs,
                        dofs=vids if with_jac else None)
                def thumb_landing(with_jac=False):
                    values=[]
                    for f in (0,5):
                        if f not in selected:
                            values.append(np.zeros((2,len(qids))) if with_jac else np.zeros(2))
                            continue
                        k=selected[f]
                        g=next(g for g in thumb_collision_meshes if m.geom_bodyid[g]==m.site_bodyid[ids[f]])
                        values.append(landing_clearance(m,ik,g,thumb_collision_meshes[g],
                            keypos[k],keysize[k],a.white_front_min,dofs=vids if with_jac else None))
                    return a.thumb_landing_weight*np.concatenate(values,axis=0)
                def tip_residual():
                    error=ik.site_xpos[ids]-wanted
                    if a.idle_clearance:
                        for f in range(10):
                            if f not in selected:error[f,2]=min(0,error[f,2])
                    if a.free_key_x:
                        for f,k in selected.items():
                            if f in lifting:continue
                            lo=keypos[k,0]-.8*keysize[k,0]; hi=keypos[k,0]+.8*keysize[k,0]
                            if 'white' in m.geom(int(keygeom[k])).name: lo=max(lo,a.white_front_min)
                            x=ik.site_xpos[ids[f],0]
                            error[f,0]=x-np.clip(x,lo,hi)
                    if a.contact_objective:
                        for f,k in allowed:
                            g=next(g for g,fi in finger_geoms.items() if fi==f)
                            dist=mujoco.mj_geomDistance(m,ik,g,int(keygeom[k]),1.,None)
                            error[f,2]=dist+a.contact_penetration
                    return error*weights
                def clearance_residual(with_jac=False):
                    values=np.zeros(len(hand_geoms)); rows=np.zeros((len(hand_geoms),len(qids)))
                    mujoco.mj_collision(m,ik)
                    for ci in range(ik.ncon):
                        c=ik.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
                        f,k=finger_geoms.get(g1),geomkeys.get(g2)
                        if f is None or k is None:f,k=finger_geoms.get(g2),geomkeys.get(g1)
                        if (f,k) in allowed:continue
                        deficit=.0005-float(c.dist)
                        for g in (g1,g2):
                            if g not in hand_geom_rows:continue
                            row=hand_geom_rows[g]
                            if deficit<=values[row]:continue
                            values[row]=deficit
                            if with_jac:
                                j1=np.zeros((3,m.nv));j2=np.zeros((3,m.nv))
                                mujoco.mj_jac(m,ik,j1,None,c.pos,int(m.geom_bodyid[g1]))
                                mujoco.mj_jac(m,ik,j2,None,c.pos,int(m.geom_bodyid[g2]))
                                rows[row]=-c.frame[:3]@(j2-j1)[:,vids]
                    factors=np.array([a.thumb_collision_weight if g in thumb_collision_meshes else 1.
                                      for g in hand_geoms])
                    return (a.collision_weight*rows*factors[:,None] if with_jac
                            else a.collision_weight*values*factors)
                def angle_residual(with_jac=False):
                    values=[];rows=[]
                    for f,(body,local) in distal.items():
                        direction=ik.xmat[body].reshape(3,3)@local
                        dot=direction[2];lo=-np.sin(np.deg2rad(80));hi=-np.sin(np.deg2rad(60))
                        error=dot-np.clip(dot,lo,hi) if f in selected else 0.
                        values.append(error)
                        jr=np.zeros((3,m.nv))
                        if with_jac and error:
                            mujoco.mj_jacBody(m,ik,None,jr,body)
                        rows.append(np.cross(direction,[0,0,1])@jr[:,vids])
                    return a.tip_angle_weight*np.array(rows if with_jac else values)
                def residual(q):
                    ik.qpos[qids]=q; fk(m,ik)
                    c=clearance_residual() if a.collision_weight else []
                    angles=angle_residual() if a.tip_angle_weight else []
                    thumb=thumb_clearance() if a.thumb_key_clearance_weight else []
                    landing=thumb_landing() if a.thumb_landing_weight else []
                    cross=thumb_cross_clearance() if a.thumb_cross_clearance_weight else []
                    return np.r_[tip_residual().ravel(), regularization*(q-reference),c,angles,thumb,landing,cross]
                def jacobian(q):
                    ik.qpos[qids]=q; fk(m,ik)
                    rows=[]
                    for f,sid in enumerate(ids):
                        jp=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,ik,jp,None,int(sid))
                        if a.idle_clearance and f not in selected and ik.site_xpos[sid,2]>wanted[f,2]:jp[2]=0
                        if a.free_key_x and f in selected and f not in lifting:
                            k=selected[f];lo=keypos[k,0]-.8*keysize[k,0];hi=keypos[k,0]+.8*keysize[k,0]
                            if 'white' in m.geom(int(keygeom[k])).name:lo=max(lo,a.white_front_min)
                            if lo<ik.site_xpos[sid,0]<hi:jp[0]=0
                        if a.contact_objective and (f,selected.get(f)) in allowed:
                            k=selected[f];g=next(g for g,fi in finger_geoms.items() if fi==f)
                            points=np.zeros(6);dist=mujoco.mj_geomDistance(m,ik,g,int(keygeom[k]),1.,points)
                            normal=(points[:3]-points[3:])/dist if abs(dist)>1e-8 else np.array([0.,0.,1.])
                            jac=np.zeros((3,m.nv));mujoco.mj_jac(m,ik,jac,None,points[:3],int(m.geom_bodyid[g]))
                            jp[2]=normal@jac
                        rows.append(jp[:,vids]*weights[f,:,None])
                    rows.append(np.diag(regularization))
                    if a.collision_weight:rows.append(clearance_residual(True))
                    if a.tip_angle_weight:rows.append(angle_residual(True))
                    if a.thumb_key_clearance_weight:rows.append(thumb_clearance(True))
                    if a.thumb_landing_weight:rows.append(thumb_landing(True))
                    if a.thumb_cross_clearance_weight:rows.append(thumb_cross_clearance(True))
                    return np.vstack(rows)
                sol=least_squares(residual,np.clip(ik.qpos[qids],solve_low+1e-10,solve_high-1e-10),
                    jac=jacobian,bounds=(solve_low,solve_high),max_nfev=a.iterations,
                    ftol=1e-7,xtol=1e-7,gtol=1e-8,x_scale='jac')
                ik.qpos[qids]=sol.x; fk(m,ik)
            for it in range(a.iterations if a.solver=='damped' or coupling is not None else 0):
                C=np.eye(len(qids))[:,independent]
                if coupling is not None and a.coupled_ik:
                    physical=np.array(bundle.neutral_action)
                    physical[fullids]=ik.qpos[qids]
                    # Solve the self-consistent commanded FK, including the
                    # effect of requested swing/Pitch on the follower angles.
                    # Runtime coupling still uses the real state each substep.
                    count=1+len(independent)
                    batch=np.repeat(physical[None],count,axis=0)
                    batch[np.arange(1,count),fullids[independent]]+=.001
                    qbatch=[]; xbatch=[]
                    for row in batch:
                        ik.qpos[qids]=row[fullids]; fk(m,ik)
                        qbatch.append(ik.qpos.copy()); xbatch.append(ik.xmat.copy())
                    mapped,_=coupling.apply(torch.tensor(batch,dtype=torch.float32),
                        actual_qpos=torch.tensor(np.array(qbatch),dtype=torch.float32),
                        body_xmat=torch.tensor(np.array(xbatch),dtype=torch.float32))
                    mapped=mapped.numpy()
                    ik.qpos[qids]=mapped[0,fullids]
                    for col in range(len(independent)):
                        for follower in coupling.indices[:,1:].flatten().numpy():
                            row=np.flatnonzero(fullids==follower)[0]
                            C[row,col]=np.clip((mapped[col+1,follower]-mapped[0,follower])/.001,-10,10)
                fk(m,ik)
                J=[]; err=[]
                for f,sid in enumerate(ids):
                    jp=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,ik,jp,None,int(sid))
                    J.append(jp[:,vids]*weights[f,:,None]); err.extend((wanted[f]-ik.site_xpos[sid])*weights[f])
                if a.collision_weight:
                    mujoco.mj_collision(m,ik)
                    for ci in range(ik.ncon):
                        c=ik.contact[ci]; g1,g2=int(c.geom1),int(c.geom2)
                        if not (g1 in hand_geoms or g2 in hand_geoms): continue
                        f,k=finger_geoms.get(g1),geomkeys.get(g2)
                        if f is None or k is None: f,k=finger_geoms.get(g2),geomkeys.get(g1)
                        if (f,k) in allowed: continue
                        deficit=.001-float(c.dist)
                        if deficit<=0: continue
                        j1=np.zeros((3,m.nv)); j2=np.zeros((3,m.nv))
                        mujoco.mj_jac(m,ik,j1,None,c.pos,int(m.geom_bodyid[g1]))
                        mujoco.mj_jac(m,ik,j2,None,c.pos,int(m.geom_bodyid[g2]))
                        J.append((a.collision_weight*c.frame[:3]@(j2-j1)[:,vids])[None])
                        err.append(a.collision_weight*deficit)
                J=np.vstack(J); err=np.array(err)
                # Small reference cost selects a stable joint posture in the IK nullspace.
                reg=.000003
                J=J@C
                delta=np.linalg.solve(J.T@J+(reg+1e-6)*np.eye(len(independent)),
                    J.T@err+reg*(rest[qids[independent]]-ik.qpos[qids[independent]]))
                nextq=ik.qpos[qids].copy()
                nextq[independent]+=np.clip(delta,-max_delta[independent],max_delta[independent])*.7
                ik.qpos[qids]=np.clip(nextq,qlow,qhigh)
                if np.max(np.abs(err))<.00015:
                    break
            physical=np.array(bundle.neutral_action)
            plan=ik.qpos[qids].copy()
            physical[fullids]=plan+ff*(plan-previous_plan)/control_dt+tracking*(plan-d.qpos[qids])
            if wrist_command is not None:physical[fullids[wrist_cols]]=wrist_command
            previous_plan=plan
            if coupling is not None:
                physical_t, info=coupling.apply(torch.tensor(physical[None],dtype=torch.float32),
                    actual_qpos=actual_q,body_xmat=actual_x)
                physical=physical_t[0].numpy()
            physical=limiter.apply(torch.tensor(np.clip(physical,low,high)[None],dtype=torch.float32))[0][0].numpy()
            d.ctrl[aid]=physical[fullids]
            previous=physical[fullids].copy()
            all_command_times.append(t);all_commands.append(physical.copy());all_plans.append(plan.copy())
            if step>=0:
                cmdtrace.append(physical.copy()); tiperr.append(float(np.max(np.linalg.norm((wanted-ik.site_xpos[ids])*weights,axis=1))))
                wanted_trace.append(wanted.copy()); ik_tip_trace.append(ik.site_xpos[ids].copy())
            if step % round(1/dt)==0:
                print(json.dumps(dict(t=round(t,3),wall_s=round(time.monotonic()-start_wall,1),ik_error_m=tiperr[-1] if tiperr else None)),flush=True)
        if step>=0:
            mujoco.mj_forward(m,d)
            actual=bundle.key_activation(d.qpos)
            contact=np.zeros((10,88),bool); forces=np.zeros((10,88))
            collision=[0,0]
            for ci in range(d.ncon):
                c=d.contact[ci]; g1,g2=int(c.geom1),int(c.geom2)
                if c.efc_address<0 or c.dist>1e-6: continue
                force=np.zeros(6); mujoco.mj_contactForce(m,d,ci,force)
                if force[0]<=1e-8: continue
                f,k=finger_geoms.get(g1),geomkeys.get(g2)
                if f is None or k is None: f,k=finger_geoms.get(g2),geomkeys.get(g1)
                if f is not None and k is not None: contact[f,k]=True; forces[f,k]+=force[0]
                if g1 in hand_geoms and g2 in hand_geoms:
                    collision[int(hand_geoms[g1]!=hand_geoms[g2])]+=1
            times.append(t); records.append(actual); contacttrace.append(contact); forcetrace.append(forces); collisiontrace.append(collision)
            actual_tip_trace.append(d.site_xpos[ids].copy())
            qtrace.append(d.qpos.copy()); vtrace.append(d.qvel.copy()); ctrltrace.append(d.ctrl.copy())
        if step<round(duration/dt):
            if coupling is not None:
                mujoco.mj_forward(m,d)
                applied,_=coupling.apply(torch.tensor(physical[None],dtype=torch.float32),
                    actual_qpos=torch.tensor(d.qpos[None],dtype=torch.float32),
                    body_xmat=torch.tensor(d.xmat[None],dtype=torch.float32))
                d.ctrl[aid]=applied[0].numpy()[fullids]
            mujoco.mj_step(m,d)
            enforce_stops()
    times=np.array(times); actual=np.array(records,dtype=bool)
    target=np.zeros_like(actual); assign=np.zeros((len(times),10,88),bool)
    for st,en,k,f in events:
        mask=(times>=st)&(times<en); target[mask,int(k)]=True; assign[mask,int(f),int(k)]=True
    tp=int((actual&target).sum()); fp=int((actual&~target).sum()); fn=int((~actual&target).sum())
    contacts=np.array(contacttrace); collisions=np.array(collisiontrace)
    report=dict(kind='actuator_only_ik_physical_v1',duration_s=duration,physics_hz=1/dt,control_hz=a.control_hz,
        original_midi_micro_f1=2*tp/max(2*tp+fp+fn,1),tp=tp,fp=fp,fn=fn,
        assigned_contact_recall=float((contacts&assign).sum()/max(assign.sum(),1)),
        same_hand_contact_samples=int((collisions[:,0]>0).sum()),cross_hand_contact_samples=int((collisions[:,1]>0).sum()),
        max_tip_error_m=max(tiperr),mean_tip_error_m=float(np.mean(tiperr)),
        physical_key_qpos_written_after_reset=False,
        robot_state_projection='current training hard_v1 stops only; keys excluded',
        hard_stop_counts=stopcounts.tolist(),physical_control='unchanged position actuators; mj_step; hard_v1 robot stops',
        full_piece=bool(duration>=events[:,1].max()),acceptance_passed=False,
        top_approach_enabled=a.top_approach,approach_control_samples=approach_counts,
        source=diagnostic,wall_seconds=time.monotonic()-start_wall)
    report['acceptance_passed']=report['full_piece'] and report['original_midi_micro_f1']>=.95
    np.savez_compressed(out/'physical_trace.npz',time=times,qpos=qtrace,qvel=vtrace,ctrl=ctrltrace,
        actual_keys=actual,target_keys=target,assignment=assign,finger_key_contact=contacts,
        contact_force=forcetrace,hand_contacts=collisions,original_events=events,
        commands=cmdtrace,wanted_tips=wanted_trace,ik_tips=ik_tip_trace,actual_tips=actual_tip_trace,
        all_command_times=all_command_times,all_commands=all_commands,all_plans=all_plans,note_adjustments=adjustments,
        key_qpos=np.asarray(qtrace)[:,bundle.key_qadr],key_limits=m.jnt_range[bundle.key_ids],
        key_activation_fraction=bundle.manifest['key_activation_fraction'])
    (out/'replay_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='source'},indent=2),flush=True)


if __name__=='__main__':
    main()
