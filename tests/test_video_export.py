"""Export contract tests: CLI, source selection, timing and quaternion playback."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'preview/scripts'))
import export_videos as cli
import export_video_job as job
from export_score_annotations import align_score


def test_user_syntax_aliases_and_multiview():
    args=cli.parse_args(['--pt','a.pt','--videonums','2','--video1','arms','backend',
                         '--video2','noarms','forhead','top','layout=horizontal','--plots','arm-joints','tcp'])
    assert args.videos[0]['arms'] and args.videos[0]['views']==['back']
    assert args.videos[1]['views']==['front','top'] and args.videos[1]['layout']=='horizontal'
    assert args.plots==['arm-joints','tcp']
    assert cli.parse_args(['--pt','a.pt']).videos[0]['arms'] is False


def test_joint_plots_default_to_both_arms_and_hands_with_explicit_opt_out():
    assert cli.parse_args(['--pt','a.pt']).plots==['arm-joints','hand-joints']
    assert cli.parse_args(['--pt','a.pt','--plots','none']).plots==[]
    assert cli.parse_args(['--pt','a.pt','--plots','hand-joints']).plots==['hand-joints']
    assert cli.parse_args(['--pt','a.pt','--plots','all']).plots==list(cli.PLOTS)


@pytest.mark.parametrize('count', [1, 2])
def test_gpu_count_reaches_submission_and_saved_plan(tmp_path, monkeypatch, count):
    files = {'tsensei_v3_2_pianist/test.py': b'frozen source'}
    archive = tmp_path / 'source.tar.gz'
    make_archive(archive, files)
    checkpoint = tmp_path / 'final.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    config = {'source_runtime': {'source_sha256': {
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    output = tmp_path / 'export'
    monkeypatch.setattr(sys, 'argv', ['export_videos.py', '--pt', str(checkpoint),
        '--source-archive', str(archive), '--output', str(output), '--gpus', str(count)])
    commands = []
    def fake_submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='1234\n', stderr='')
    monkeypatch.setattr(cli.subprocess, 'run', fake_submit)
    cli.main()
    assert commands == [['bash', str(ROOT / 'scripts/hpc/submit.sh'), '--gpus', str(count),
                         'export', '--plan', str(output / 'export_plan.json')]]
    plan = json.loads((output / 'export_plan.json').read_text())
    assert plan['gpus'] == count
    assert Path(plan['checkpoint']).read_bytes() == checkpoint.read_bytes()
    assert plan['source_archive'] == str(archive)


@pytest.mark.parametrize('args',[
    ['--gpus','0'], ['--gpus','3'],
    ['--videonums','2','--video1','arms','front'],
    ['--video2','arms','front'],
    ['--video1','arms','front','--video1','arms','back'],
    ['--video1','arms','front','top','azimuth=90'],
    ['--video1','arms','front','name=../bad'],
    ['--video1','arms','front','name=same','--video2','noarms','top','name=same'],
    ['--video1','arm','front'], ['--video1','arms','frnot'],
    ['--video1','arms','front','front'], ['--plots','none','tcp'],
    ['--size','639x360'], ['--start','2','--end','1'], ['--speed','0'],['--fps','nan'],
    ['--video1','arms','front','elevation=91'],
    ['--piano-fill','0'], ['--piano-fill','1.1'],
])
def test_invalid_requests_fail_before_submission(args):
    with pytest.raises(SystemExit):cli.parse_args(['--pt','a.pt',*args])


def make_archive(path, files):
    with tarfile.open(path,'w:gz') as archive:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))


def test_archive_selection_checks_bytes_and_extra_contract_files(tmp_path):
    files={'tsensei_v3_2_pianist/test.py':b'old','configs/actions/profile.json':b'{}'}
    expected={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    path=tmp_path/'source.tar.gz';make_archive(path,files)
    assert cli.select_archive({'source_runtime':{'source_sha256':expected}},path)==path
    make_archive(path,{**files,'tsensei_v3_2_pianist/extra.py':b'new'})
    assert not cli.archive_matches(path,expected)
    with pytest.raises(ValueError,match='No matching'):cli.select_archive({'source_runtime':{'source_sha256':expected}},path)


def test_recording_hash_mismatch_and_no_overwrite(tmp_path):
    pt=tmp_path/'best.pt';pt.write_bytes(b'checkpoint')
    config={'source_runtime':{'source_sha256':{'x':'y'}}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    source=tmp_path/'recording';(source/'kinematic_scene').mkdir(parents=True)
    (source/'kinematic_scene/scene.xml').write_text('<mujoco/>')
    (source/'physics_trace.npz').write_bytes(b'trace')
    report={'checkpoint_sha256':'wrong','source_runtime':config['source_runtime']}
    (source/'report.json').write_text(json.dumps(report))
    args=cli.parse_args(['--pt',str(pt),'--recording',str(source),'--no-audio','--no-fingering',
                         '--output',str(tmp_path/'new')])
    with pytest.raises(ValueError,match='does not match'):cli.build_plan(args)
    report['checkpoint_sha256']=cli.sha256(pt);(source/'report.json').write_text(json.dumps(report))
    plan=cli.build_plan(args)
    assert plan['source_archive'] is None and not Path(plan['output']).exists()
    Path(plan['output']).mkdir()
    with pytest.raises(ValueError,match='already exists'):cli.build_plan(args)


def test_clip_interpolation_and_audio_tempo_use_original_time():
    time=np.array([0.,.1,.2,.3]);values=np.column_stack((time,2*time))
    np.testing.assert_allclose(job.interpolate(values,time,.15),[.15,.3])
    np.testing.assert_allclose(job.crop_times(time,.05,.27),[.05,.1,.2,.27])
    filters=job.audio_filters(2,4,.125,.5)
    assert filters.startswith('atrim=start=2:end=4.0625') and filters.count('atempo=0.5')==3
    assert job.layout_shape(3,'grid')==(2,2)
    assert job.layout_shape(2,'vertical')==(2,1)


def test_mocap_quaternion_slerp_and_camera_configuration():
    import mujoco
    from scipy.spatial.transform import Rotation
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="piano/base" type="box" size=".1 .6 .02"/><body mocap="true" name="m"><geom size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    trace={'time':np.array([0.,1.]),'qpos':np.zeros((2,0)),
           'mocap_pos':np.array([[[0,0,0]],[[2,0,0]]]),
           'mocap_quat':np.roll(Rotation.from_euler('z',[[0],[90]],degrees=True).as_quat(),1,axis=1)[:,None,:]}
    job.set_state(model,data,trace,.5)
    np.testing.assert_allclose(data.mocap_pos[0],[1,0,0])
    np.testing.assert_allclose(Rotation.from_quat(np.roll(data.mocap_quat[0],-1)).as_euler('xyz',degrees=True),[0,0,45],atol=1e-10)
    top,_=job.camera_for('top',model,True,{})
    assert top.elevation==-90
    front,_=job.camera_for('front',model,False,{'distance':2.3})
    assert front.distance==2.3 and front.azimuth==0


def test_worker_rejects_execution_outside_slurm(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm|submit.sh'):job.run({'output':str(tmp_path)})


def test_hand_joint_selection_excludes_piano_and_handles_numpy_enums():
    import mujoco
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="right_/palm_base_tx" type="slide"/><geom size=".1"/></body>
      <body><joint name="right_/finger" type="hinge"/><geom size=".1"/></body>
      <body><joint name="left_/finger"/><geom size=".1"/></body>
      <body><joint name="piano/key"/><geom size=".1"/></body>
      </worldbody></mujoco>''')
    assert job.hand_joint_ids(model,'right')==[0,1]
    assert job.hand_joint_ids(model,'left')==[2]


def test_fixed_preparation_prefix_joins_without_duplicate_endpoint(tmp_path):
    quat=np.array([[1.,0,0,0]])
    np.savez(tmp_path/'physics_trace.npz',physics_dt=.1,qpos=np.array([[1.],[2.]]),
             initial_qpos=np.array([0.]),mocap_pos=np.zeros((2,1,3)),initial_mocap_pos=np.zeros((1,3)),
             mocap_quat=np.tile(quat,(2,1,1)),initial_mocap_quat=quat)
    np.savez(tmp_path/'preparation.npz',time=np.array([0.,.1]),qpos=np.array([[-1.],[0.]]),
             mocap_pos=np.zeros((2,1,3)),mocap_quat=np.tile(quat,(2,1,1)))
    (tmp_path/'report.json').write_text(json.dumps({'preparation_prefix_seconds':.1}))
    trace,_=job.load_trace(tmp_path)
    np.testing.assert_allclose(trace['time'],[0,.1,.2,.3])
    np.testing.assert_allclose(trace['qpos'][:,0],[-1,0,1,2])


@pytest.mark.parametrize('size',[(1280,720),(640,480),(480,640)])
@pytest.mark.parametrize('view',['front','back','overview','top','left'])
def test_orthographic_camera_piano_fills_actual_gl_viewport(view,size):
    import mujoco
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="piano/base" type="box" size=".13 .61 .02" pos="-.05 0 .02"/>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <body pos="4 4 4"><geom name="right_/hand" size=".5"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    camera,settings=job.camera_for(view,model,True,{},*size)
    scene=mujoco.MjvScene(model,maxgeom=20)
    mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    gl=scene.camera[0]
    assert gl.orthographic==1 and settings['projection']=='orthographic'
    screen_right=np.cross(gl.forward,gl.up); screen_right/=np.linalg.norm(screen_right)
    center=(scene.camera[0].pos+scene.camera[1].pos)/2
    horizontal=(job.piano_corners(model)-center) @ screen_right
    frustum_width=(gl.frustum_top-gl.frustum_bottom)*size[0]/size[1]
    # Verify actual MuJoCo projection, not just our saved fill metadata.
    np.testing.assert_allclose([horizontal.min(),horizontal.max()],[-frustum_width/2,frustum_width/2],atol=2e-7)
    farther,other=job.camera_for(view,model,True,{'distance':8.},*size)
    assert farther.distance==8. and other['ortho_height_m']==settings['ortho_height_m']


def test_fill_ratio_and_multiview_scales_are_independent():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="piano/base" type="box" size=".13 .61 .02"/></worldbody></mujoco>')
    _,full=job.camera_for('front',model,False,{},1280,720,1.)
    _,wide=job.camera_for('front',model,False,{},1280,720,.5)
    _,side=job.camera_for('left',model,False,{},1280,720,1.)
    assert wide['ortho_height_m']==2*full['ortho_height_m']
    assert side['ortho_height_m']<full['ortho_height_m']


def test_recording_reference_resolves_clean_export_without_model_files(tmp_path):
    cache=tmp_path/'logs/job/recording';cache.mkdir(parents=True)
    (cache/'physics_trace.npz').write_bytes(b'trace')
    output=tmp_path/'export';output.mkdir()
    (output/'report.json').write_text(json.dumps({'recording':str(cache)}))
    assert cli.resolve_recording(output)==cache
    assert cli.resolve_recording(cache)==cache


def test_fingering_switches_are_independent_of_general_overlay():
    args=cli.parse_args(['--pt','best.pt','--no-overlay'])
    assert args.fingering and not args.overlay
    assert not cli.parse_args(['--pt','best.pt','--no-fingering']).fingering


def annotated_score():
    fingers=np.zeros((5,10,88),dtype=np.float32)
    fingers[1:3,0,39]=1  # right thumb C4
    fingers[1:3,7,27]=1  # left middle C3, simultaneously
    fingers[3,4,43]=1
    goal=np.column_stack((fingers.sum(axis=1)>0,np.zeros(5))).astype(np.float32)
    return goal,fingers


@pytest.mark.parametrize('prefix',[0.,1.])
def test_fingering_matches_scored_endpoints_preparation_and_release(prefix):
    goal,fingers=annotated_score()
    # First policy frame is preparation; final frame is release, both empty.
    score=align_score(goal,fingers,np.arange(1,6)*.025,goal,.025,prefix)
    score['metadata']={'preparation_prefix_seconds':prefix}
    assert not job.fingering_at(score,prefix).any()
    assert not job.fingering_at(score,prefix+.025).any()
    np.testing.assert_array_equal(job.fingering_at(score,prefix+.026),fingers[1])
    np.testing.assert_array_equal(job.fingering_at(score,prefix+.075),fingers[2])
    np.testing.assert_array_equal(job.fingering_at(score,prefix+.076),fingers[3])
    # Arbitrary clip start and playback speed sample the same original timeline.
    np.testing.assert_array_equal(job.fingering_at(score,prefix+.026+.02*2),fingers[2])
    assert not job.fingering_at(score,prefix+.125).any()
    assert not job.fingering_at(score,prefix+10).any()


def test_annotation_reconstruction_rejects_other_goals_and_time_shift():
    goal,fingers=annotated_score()
    changed=goal.copy();changed[1,39]=0
    with pytest.raises(ValueError,match='differ from recorded'):
        align_score(goal,fingers,np.arange(1,6)*.025,changed,.025,0.)
    with pytest.raises(ValueError,match='control times'):
        align_score(goal,fingers,np.arange(5)*.025,goal,.025,0.)
    missing=fingers.copy();missing[1,0,39]=0
    with pytest.raises(ValueError,match='inconsistent'):
        align_score(goal,missing,np.arange(1,6)*.025,goal,.025,0.)


def test_exact_events_do_not_replace_applied_shadow_fingering():
    from export_score_annotations import recorded_score_targets
    # Sub-frame onset and exact endpoint release differ from the legacy grid.
    time=np.arange(1,5)*.025
    old_goal=np.zeros((4,89),dtype=np.float32)
    old_fingers=np.zeros((4,10,88),dtype=np.float32)
    old_goal[1:3,39]=1
    old_fingers[1:3,0,39]=1
    score=dict(goal=old_goal,fingers=old_fingers,
        exact_note_events=np.array([[.01,.05,39,0],[.05,.09,43,4]]),
        exact_sustain_events=np.array([[.025,1.],[.075,0.]]))
    goal,fingers=recorded_score_targets(score,time)
    assert goal[:,39].tolist()==[0,1,1,0]
    assert goal[:,43].tolist()==[0,0,0,0]
    assert goal[:,-1].tolist()==[0,0,0,0]
    assert fingers[:,0,39].tolist()==[0,1,1,0]
    assert fingers[:,4,43].tolist()==[0,0,0,0]
    align_score(goal,fingers,time,old_goal,.025,0.)
    from tsensei_v3_2_pianist.mjlab_env.exact_score import sample_targets
    exact_goal,_=sample_targets(score['exact_note_events'],score['exact_sustain_events'],time)
    with pytest.raises(ValueError,match='differ from recorded'):
        align_score(goal,fingers,time,exact_goal,.025,0.)
    legacy=recorded_score_targets(dict(goal=old_goal,fingers=old_fingers),time)
    assert legacy[0] is old_goal and legacy[1] is old_fingers


def test_legacy_recording_fingering_uses_verified_archive_and_hashes_rollout(tmp_path):
    files={'tsensei_v3_2_pianist/test.py':b'old'}
    expected={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    archive=tmp_path/'source.tar.gz';make_archive(archive,files)
    pt=tmp_path/'best.pt';pt.write_bytes(b'checkpoint')
    config={'source_runtime':{'source_sha256':expected}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    recording=tmp_path/'recording';(recording/'kinematic_scene').mkdir(parents=True)
    (recording/'kinematic_scene/scene.xml').write_text('<mujoco/>')
    (recording/'physics_trace.npz').write_bytes(b'trace')
    (recording/'rollout.npz').write_bytes(b'original applied goals')
    (recording/'audio_substeps.npz').write_bytes(b'original actual keys')
    (recording/'report.json').write_text(json.dumps(dict(checkpoint_sha256=cli.sha256(pt),**config)))
    args=cli.parse_args(['--pt',str(pt),'--recording',str(recording),'--source-archive',str(archive),
                         '--no-audio','--output',str(tmp_path/'export')])
    plan=cli.build_plan(args)
    assert plan['fingering'] and plan['source_archive']==str(archive)
    assert plan['recording_sha256']['rollout.npz']==cli.sha256(recording/'rollout.npz')
    assert plan['recording_sha256']['audio_substeps.npz']==cli.sha256(recording/'audio_substeps.npz')


def test_fingering_footer_reserves_vertical_space_without_zooming_out():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="piano/base" type="box" size=".13 .61 .02"/></worldbody></mujoco>')
    _,plain=job.camera_for('front',model,False,{})
    camera,settings=job.camera_for('front',model,False,{},caption_height=54,footer_height=92)
    assert settings['ortho_height_m']==plain['ortho_height_m']
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    scene=mujoco.MjvScene(model,maxgeom=20)
    mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    gl=scene.camera[0]
    up=(job.piano_corners(model)-gl.pos) @ gl.up
    pixels=720*(.5-up/settings['ortho_height_m'])
    assert pixels.min()>54 and pixels.max()<720-92


def test_native_trained_arms_require_explicit_native_export(tmp_path,monkeypatch):
    pt=tmp_path/'best.pt';pt.write_bytes(b'fixture')
    (tmp_path/'config.json').write_text(json.dumps({'arm_follow':{'controller':'wuji_cartesian_v1'},'source_runtime':{'source_sha256':{'x':'y'}}}))
    monkeypatch.setattr(cli,'select_archive',lambda *a,**k:pt)
    args=cli.parse_args(['--pt',str(pt),'--video1','native','front','--plots','hand-joints'])
    plan=cli.build_plan(args)
    assert plan['videos'][0]['native_arms'] and plan['videos'][0]['arms']
    args=cli.parse_args(['--pt',str(pt),'--video1','noarms','front','--plots','hand-joints'])
    with pytest.raises(ValueError,match='requires native'):cli.build_plan(args)
    args=cli.parse_args(['--pt',str(pt),'--video1','native','front'])
    with pytest.raises(ValueError,match='hand-joints/tcp'):cli.build_plan(args)


def test_native_recording_binary_keeps_geometry_when_xml_mesh_is_missing(tmp_path):
    import mujoco
    from replay_arm_ik import read_targets
    folder=tmp_path/'kinematic_scene';folder.mkdir()
    mesh=b'v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    xml='''<mujoco><asset><mesh name="arm" file="right_Link1_R.obj"/></asset>
      <worldbody><body name="right_/base_link"><joint type="slide" axis="1 0 0"/>
      <geom type="mesh" mesh="arm"/></body><body name="left_/base_link" pos="0 1 0">
      <joint type="slide" axis="0 1 0"/><geom type="mesh" mesh="arm"/></body></worldbody></mujoco>'''
    original=mujoco.MjModel.from_xml_string(xml,assets={'right_Link1_R.obj':mesh})
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),original)
    mujoco.mj_saveModel(original,str(folder/'model.mjb'))
    with pytest.raises(ValueError,match='right_Link1_R.obj'):
        mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    np.testing.assert_array_equal(job.load_recorded_model(tmp_path,native=False).mesh_vert,original.mesh_vert)
    model=job.load_recorded_model(tmp_path,native=True)
    np.testing.assert_array_equal(model.mesh_vert,original.mesh_vert)
    np.testing.assert_array_equal(model.mesh_face,original.mesh_face)
    trace=dict(time=np.array([0.,.005]),qpos=np.array([[0.,0.],[.01,.02]]),
               mocap_pos=np.empty((2,0,3)),mocap_quat=np.empty((2,0,4)))
    positions,rotations=read_targets(tmp_path,trace,model=model)
    np.testing.assert_allclose(positions,[[[0,0,0],[0,1,0]],[[.01,0,0],[0,1.02,0]]])
    np.testing.assert_allclose(rotations,np.broadcast_to(np.eye(3),(2,2,3,3)))


def test_native_model_does_not_silently_fall_back_to_xml(tmp_path):
    folder=tmp_path/'kinematic_scene';folder.mkdir()
    (folder/'scene.xml').write_text('<mujoco/>')
    with pytest.raises(FileNotFoundError,match='compiled model'):
        job.load_recorded_model(tmp_path,native=True)


def test_recorded_runtime_uses_venv_prefix_for_native_and_hand_only(tmp_path,monkeypatch):
    root=tmp_path/'project';environment=tmp_path/'learner_env'
    (environment/'bin').mkdir(parents=True)
    (environment/'bin/python').write_text('fixture executable')
    (root/'training_mjlab').mkdir(parents=True)
    (root/'training_mjlab/.venv').symlink_to(environment,target_is_directory=True)
    config=tmp_path/'config.json';config.write_text(json.dumps({'arm_follow':{'enabled':True}}))
    plan={'config':str(config)}
    monkeypatch.setattr(job,'ROOT',root)
    monkeypatch.setattr(job.sys,'prefix',str(tmp_path/'composer'))
    assert job.native_render_python(plan)==root/'training_mjlab/.venv/bin/python'
    monkeypatch.setattr(job.sys,'prefix',str(environment))
    assert job.native_render_python(plan) is None
    config.write_text('{}')
    monkeypatch.setattr(job.sys,'prefix',str(tmp_path/'composer'))
    assert job.native_render_python(plan)==root/'training_mjlab/.venv/bin/python'


def test_hand_only_uses_binary_bounds_instead_of_rounded_xml(tmp_path):
    import mujoco
    folder=tmp_path/'kinematic_scene';folder.mkdir()
    bound=np.pi/12
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/><worldbody>
      <body><joint name="right_/palm_base_yaw" range="0 {bound:.17g}"/>
      <geom size=".1"/></body></worldbody></mujoco>''')
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),model)
    mujoco.mj_saveModel(model,str(folder/'model.mjb'))
    xml=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    captured=job.load_recorded_model(tmp_path,native=False)
    measured=float(np.nextafter(np.float32(bound), np.float32(-np.inf)))
    assert xml.jnt_range[0,1] < measured < captured.jnt_range[0,1]
    assert captured.jnt_range[0,1] == bound


def test_native_runtime_reexec_happens_before_recording_or_workspace(tmp_path,monkeypatch):
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps({'output':str(tmp_path)}))
    python=tmp_path/'learner/bin/python'
    monkeypatch.setattr(sys,'argv',['export_video_job.py','--plan',str(planfile)])
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setattr(job,'native_render_python',lambda plan:python)
    monkeypatch.setattr(job,'run',lambda plan:pytest.fail('run called before learner re-exec'))
    class Reentered(BaseException):pass
    calls=[]
    def execute(executable,arguments):
        calls.append((executable,arguments));raise Reentered
    monkeypatch.setattr(job.os,'execv',execute)
    with pytest.raises(Reentered):job.main()
    assert calls==[(str(python),[str(python),str(Path(job.__file__).resolve()),'--plan',str(planfile)])]


def test_recovery_reuses_hashed_recording_without_rerunning_policy(tmp_path,monkeypatch):
    recording=tmp_path/'recording';recording.mkdir()
    (recording/'physics_trace.npz').write_bytes(b'original replay')
    plan={'recording':str(recording),'recording_sha256':{'physics_trace.npz':cli.sha256(recording/'physics_trace.npz')}}
    monkeypatch.setattr(job.subprocess,'run',lambda *a,**k:pytest.fail('recovery reran policy'))
    assert job.record_once(plan,tmp_path/'new_job')==recording
