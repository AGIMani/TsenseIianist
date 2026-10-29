#!/usr/bin/env python3
"""Remote CPU build of a camera-only preview from the pinned training recipe.

Uses the training source compiler and structural wrist removal, but does not
run first-note preparation, policy inference, mj_step, or rendering. The saved
state is the original posture before preparation, not the training reset state.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def model_summary(m, d):
    import numpy as np
    joints = {}
    for j in range(m.njnt):
        name = m.joint(j).name
        if name and name.startswith(('right_/', 'left_/')):
            v = m.jnt_dofadr[j]
            joints[name.replace('forearm_', 'palm_base_')] = dict(
                range=m.jnt_range[j].tolist(), axis=m.jnt_axis[j].tolist(),
                qpos=float(d.qpos[m.jnt_qposadr[j]]), damping=float(m.dof_damping[v]),
                armature=float(m.dof_armature[v]), frictionloss=float(m.dof_frictionloss[v]))
    acts = {}
    for a in range(m.nu):
        name = m.actuator(a).name
        acts[name.replace('Forearm_', 'Palm_Base_')] = dict(
            kp=float(m.actuator_gainprm[a,0]), kd=float(-m.actuator_biasprm[a,2]),
            force_range=m.actuator_forcerange[a].tolist(), ctrl_range=m.actuator_ctrlrange[a].tolist())
    hands = {}
    for side in ('right','left'):
        bid = m.body(side+'_/base_link').id
        ids = [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(side+'_/') and (m.geom_contype[i] or m.geom_conaffinity[i])]
        hands[side] = dict(root_position=d.xpos[bid].tolist(),root_rotation=d.xmat[bid].tolist(),
                           collision_geoms=len(ids),palm_collision_geoms=int(sum(m.geom_bodyid[i]==bid for i in ids)))
    keys = {}
    for color, index, length in [('white',0,.15),('black',1,.09)]:
        j = m.joint(f'piano/{color}_joint_{index}').id
        keys[color] = dict(max_angle_rad=float(m.jnt_range[j,1]), nominal_travel_mm=float(np.tan(m.jnt_range[j,1])*length*1000))
    return dict(nq=m.nq,nv=m.nv,nu=m.nu,ngeom=m.ngeom,nmesh=m.nmesh,
                physics_dt=m.opt.timestep, integrator=int(m.opt.integrator),
                hands=hands,keys=keys,joints=joints,actuators=acts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch-command',type=Path,required=True)
    p.add_argument('--score',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--training-entry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-model',type=Path)
    p.add_argument('--previous-state',type=Path)
    p.add_argument('--scene-only',action='store_true',help='Compile only model; music task is explicitly not validated')
    a=p.parse_args()
    assert sha(a.source_archive)==a.source_sha256, 'Wrong frozen source archive'
    # Verify the exact files extracted from that archive before compiling.
    manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())['files']
    for name,row in manifest.items():
        assert sha(ROOT/name)==row['sha256'], 'Frozen source changed: '+name
    command=json.loads(a.launch_command.read_text())
    command=command[command.index('train')+1:]
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    rp=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for name,param in inspect.signature(make_recipe).parameters.items():
        default=param.default
        if isinstance(default,bool):
            rp.add_argument('--'+name.replace('_','-'),default=default,action=argparse.BooleanOptionalAction)
        else:
            rp.add_argument('--'+name.replace('_','-'),default=default,type=int if name=='lookahead' else type(default))
    fields=vars(rp.parse_known_args(command)[0]);fields['midi']=str(a.score.resolve())
    assert '--no-arms' in command and '--disable-wrist-tz' in command and '--disable-wrist-yaw' in command
    recipe=make_recipe(**fields)
    assert recipe['preparation']['mode']=='first_note_v1', 'This static export is for the current no-arm first-note model'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.scene_only:
        # Use the exact scene exporter called by source_runtime._compile. Keep
        # task rejection visible; static geometry does not require music targets.
        workspace=tempfile.TemporaryDirectory(prefix='static-training-scene-')
        scene=Path(workspace.name)/'scene';source=out/'source_config.json'
        write(source,recipe)
        env={**os.environ,'PYTHONPATH':str(ROOT),'TSENSEI_V32_EXPORT_POSE':json.dumps(recipe['pose']),
             'TSENSEI_V32_COLLISION_PROFILE':recipe['model']['collision_profile'],
             'TSENSEI_V32_DRIVE_PROFILE':recipe['model']['drive_profile']}
        subprocess.run([str(ROOT/'training/.venv/bin/python'),str(Path(__file__).resolve()),
                        '--compile-scene-only',str(source),str(scene)],env=env,check=True)
    else:
        scene,task,source=build(recipe)
        from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
        from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
        from tsensei_v3_2_pianist.mjlab_env.exact_score import ExactScore
        original_bundle=SceneBundle(scene,preparation=None)
        source_slide_ranges={}
        for side in ('right','left'):
            for axis,bounds in [('ty',[-.05,.05]),('tz',[-.6,.46])]:
                joint=original_bundle.model.joint(side+'_/palm_base_'+axis)
                assert list(joint.range)==bounds
                source_slide_ranges[joint.name]=list(joint.range)
        music=TaskBundle(task,original_bundle)
        exact=ExactScore(music.exact_note_events,music.exact_sustain_events,
            length=music.manifest['length'],control_dt=music.manifest['control_dt'],
            physics_dt=music.manifest['physics_dt'],lookahead_steps=fields['lookahead'],
            score_validation=music.score_validation,device='cpu')
        report=json.loads((task/'score_coverage.json').read_text())
        assert report['physical_coverage']==music.physical_coverage==exact.physical_coverage
        shutil.copyfile(task/'score_coverage.json',out/'score_coverage.json')
        shutil.copyfile(task/'task.json',out/'task_manifest.json')
        shutil.copyfile(task/'score.npz',out/'score.npz')
        write(out/'score_acceptance.json',dict(status='passed',score_validation=music.score_validation,
            length=music.manifest['length'],release_tail_steps=music.release_tail_steps,
            source_notes=len(exact.events),coverage_valid=report['valid'],
            missing_note_samples=len(exact.physical_coverage['missing_note_samples']),
            missing_release_samples=len(exact.physical_coverage['missing_release_samples']),
            export_load_runtime_coverage_identical=True,physical_steps=0))
        del original_bundle,music,exact
    import mujoco
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.fingering_targets import key_reference,target_for_key
    from tsensei_v3_2_pianist.finger_resistance import Resistance
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
    # Preparation changes the state, not the first_note_v1 model. Deliberately
    # keep the pre-preparation source posture for this static viewer.
    bundle=SceneBundle(scene,disable_wrist_tz=True,disable_wrist_yaw=True,preparation=None)
    m=bundle.model;d=bundle.reset_cpu()
    resistance=Resistance(m);resistance.update(d)
    assert len(resistance.rows)==40
    summary=model_summary(m,d)
    assert m.nq==136 and m.nu==48 and np.isclose(m.opt.timestep,1/recipe['environment']['physics_hz'])
    for color,expected in [('white',12),('black',10)]:
        assert np.isclose(summary['keys'][color]['nominal_travel_mm'],expected)
    for side in ('right','left'):
        assert np.allclose(m.joint(side+'_/palm_base_ty').range,[-.05,.05])
        assert np.allclose(m.joint(side+'_/palm_base_pitch').range,[-.3,.6])
        assert np.allclose(m.joint(side+'_/palm_base_roll').range,[-.3,.5])
        assert summary['hands'][side]['palm_collision_geoms']==79
        assert summary['hands'][side]['collision_geoms']==100
    low,high=physical_bounds('direct_joint_v1',bundle.manifest['action_names'],bundle.low,bundle.high)
    targets={name:[float(low[i]),float(high[i])] for i,name in enumerate(bundle.manifest['action_names']) if name.endswith('_MCP')}
    assert len(targets)==10 and all(np.allclose(bounds,[0,1.4]) for bounds in targets.values())
    summary.update(source_slide_ranges=source_slide_ranges if not a.scene_only else None,mcp_policy_targets=targets,fitted_resistance_joint_count=len(resistance.rows),
                   control_hz=recipe['environment']['control_hz'],physics_hz=recipe['environment']['physics_hz'])
    mujoco.mj_saveModel(m,str(out/'model.mjb'))
    np.savez(out/'state.npz',qpos=d.qpos,mocap_pos=d.mocap_pos,mocap_quat=d.mocap_quat)
    keys=[]
    for i in range(88):
        color='black' if (i+21)%12 in (1,3,6,8,10) else 'white'
        b=m.body(f'piano/{color}_key_{i}');g=m.geom(f'piano/{color}_key_geom_{i}')
        _,size=key_reference(m,g.id)
        keys.append(dict(index=i,midi=i+21,color=color,center_mm=float(b.pos[1]*1000),width_mm=float(size[1]*2000),
                         body_position_mm=(b.pos*1000).tolist(),target_position_mm=(target_for_key(m,d,g.id)*1000).tolist()))
    parent=int(m.body('piano/base').parentid[0])
    source_names={name for name in manifest if name.startswith(('tsensei_v3_2_pianist/assets/','tsensei_v3_2_pianist/models/','tsensei_v3_2_pianist/suite/','configs/actions/'))}
    source_names.update(['tsensei_v3_2_pianist/fingering_targets.py','tsensei_v3_2_pianist/finger_resistance.py','tsensei_v3_2_pianist/drive_settings.py','tsensei_v3_2_pianist/physics_settings.py','tsensei_v3_2_pianist/mjlab_env/bundle.py'])
    record=dict(kind='static_model_preview',mujoco=mujoco.__version__,source_archive_sha256=a.source_sha256,
                training_entry_sha256=sha(a.training_entry),units='mm',total_width=1224,white_width=22.5,
                black_top_width=10,black_bottom_width=12,black_top_length=90,black_bottom_length=95,black_height=18,
                white_key_travel_mm=12,black_key_travel_mm=10,physics_hz=recipe['environment']['physics_hz'],control_hz=recipe['environment']['control_hz'],
                piano_origin_world_m=d.xpos[parent].tolist(),piano_rotation_world=d.xmat[parent].reshape(3,3).tolist(),
                keys=keys,physics_simulation=False,training=False,
                state_phase='original pose before physical first-note preparation; not policy reset state',
                layout_source_sha256={name:sha(ROOT/name) for name in sorted(source_names)},
                files={name:sha(out/name) for name in ('model.mjb','state.npz')})
    write(out/'preview.json',record)
    if source.resolve() != (out/'source_config.json').resolve():
        shutil.copyfile(source,out/'source_config.json')
    shutil.copyfile(scene/'manifest.json',out/'scene_manifest.json')
    shutil.copyfile(a.launch_command,out/'launch_command.json')
    write(out/'model_audit.json',summary)
    differences={}
    if a.previous_model:
        old=mujoco.MjModel.from_binary_path(str(a.previous_model));od=mujoco.MjData(old)
        with np.load(a.previous_state,allow_pickle=False) as state:od.qpos[:]=state['qpos']
        mujoco.mj_forward(old,od)
        previous=model_summary(old,od);write(out/'previous_model_audit.json',previous)
        for section in ('physics_dt','integrator','keys','hands','joints','actuators'):
            if previous[section]!=summary[section]:differences[section]=dict(before=previous[section],after=summary[section])
        write(out/'model_changes.json',differences)
    write(out/'acceptance.json',dict(status='passed',compiled_from_frozen_training_source=True,training_source_files_verified=len(manifest),
        physical_steps=0,policy_inference=False,training_started=False,state_phase=record['state_phase'],
        full_music_task_built=not a.scene_only,
        matched=['piano geometry and 12/10mm travel','bilateral A3.2 URDF and 200 collision components','Pitch range -0.3..0.6',
                 'structurally removed TZ/Yaw','source root pose','explicit control/physics frequencies','shadow_v1 per-axis drive and fitted finger resistance',
                 'MCP target bounds 0..1.4'],
        intentional_differences=['no first-note physical preparation, so static state differs from policy reset',
             'no policy/actions, coupling execution, motion limiting, reward accumulation or dynamics',
             'native MuJoCo static display rather than batched GPU Warp solver'],
        remaining=['old self-collision inspector remains separately pinned; this export updates only the ordinary static viewer']))
    write(out/'SHA256SUMS.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(status='passed',output=str(out),mujoco=mujoco.__version__,nq=m.nq,nu=m.nu,model_sha256=sha(out/'model.mjb')),indent=2),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--compile-scene-only':
        recipe=json.loads(Path(sys.argv[2]).read_text());e=recipe['environment']
        assert recipe['model'].get('wrist_pitch_range_scale',1)==1
        from tsensei_v3_2_pianist.mjlab_env.bundle import export_scene
        export_scene(Path(sys.argv[3]),gravity_compensation=e['gravity_compensation'],
                     control_timestep=1/e['control_hz'],physics_timestep=1/e['physics_hz'],build_reference=False)
    else:
        main()
