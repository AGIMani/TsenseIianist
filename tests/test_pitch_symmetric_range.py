"""Pitch range contract; physical integration is only run inside Slurm."""
import ast
import json
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('profile', ['direct_joint_v1', 'wuji_cartesian_v1'])
def test_pitch_contract(profile):
    rows = json.loads((ROOT / f'configs/actions/{profile}.json').read_text())['channels']
    pitch = [r for r in rows if r['name'].endswith('_Palm_Base_Pitch')]
    assert len(pitch) == 2
    for r in pitch:
        assert [r['physical_min'], r['physical_max']] == [-.5, .5]
        assert [r['policy_min'], r['policy_max']] == ([0., 2.] if profile == 'direct_joint_v1' else [-1., 1.])
    # Inspect constants without importing or constructing a MuJoCo scene locally.
    module = ast.parse((ROOT/'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py').read_text())
    values = {s.targets[0].id: s.value for s in module.body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}
    def entry(name):
        node = values[name]
        return next(v for k,v in zip(node.keys,node.values) if ast.literal_eval(k)=='palm_base_pitch')
    assert ast.literal_eval(entry('PALM_BASE_DOFS').elts[2]) == (-.5,.5)
    assert ast.literal_eval(entry('INITIAL_ANGLES')) == 0.

@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='Physical/GPU validation requires remote Slurm')
def test_compiled_range_mapping_preparation_and_return(tmp_path):
    import numpy as np
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
    from tsensei_v3_2_pianist.mjlab_env.ending import generate
    torch.set_num_threads(1)
    recipe=make_recipe(world_x=.16,right_world_y=.15,left_world_y=-.15,world_z=.09,
        palm_base_yaw=.2617993877991494,control_hz=20,physics_hz=1000,
        midi=str(ROOT/'training_mjlab/inputs/turkish_march_advanced/a32_repeat_static_guarded_20261025/turkish-march-advanced_fingered.proto'),
        initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',
        task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
        preparation_mode='first_note_v1',preparation_time=2,preparation_lift=0,ending_time=2)
    scene,task,_=build(recipe)
    scene,task=without_arms(scene,task,scene.parent/'no_arms')
    cfg=effective_config(parser().parse_args(['--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
        '--action-profile','direct_joint_v1','--finger-coupling','--no-coupling-reduce-observation',
        '--finger-coupling-angle-ranges',str(ROOT/'configs/finger_coupling/angle_ranges_10_80.json'),
        '--preparation-mode','first_note_v1','--preparation-time','2','--preparation-lift','0','--ending-time','2',
        '--reward-config',str(ROOT/'configs/rewards/default_20261006.json'),
        '--hard-joint-limits','--wrist-tx-max-velocity','1.99','--wrist-ty-max-velocity','1.99',
        '--wrist-tx-max-acceleration','1.99','--wrist-ty-max-acceleration','1.99',
        '--wrist-pitch-max-velocity','3.1416','--wrist-pitch-max-acceleration','15.708']))
    env=PianoTaskEnv(scene,task,num_envs=2,device='cuda:0',physics_graph=False,
        disable_wrist_tz=True,disable_wrist_yaw=True,
        **{k:cfg[k] for k in ('preparation','finger_coupling','control_limits','reward_config','action_profile')})
    b=env.scene.bundle
    env.reset()
    assert b.preparation['report']['valid']
    assert env.preparation_steps==0
    np.testing.assert_array_equal(b.reset_qpos,b.preparation_trace['qpos'][-1])
    names=b.manifest['action_names']
    ids=[names.index(f'{s}_Palm_Base_Pitch') for s in ('R','L')]
    for side,i in zip(('right','left'),ids):
        joint=b.model.joint(f'{side}_/palm_base_pitch')
        np.testing.assert_allclose(joint.range,[-.5,.5],atol=1e-12)
        actuator=np.flatnonzero(b.model.actuator_trnid[:,0]==joint.id)
        assert len(actuator)==1
        np.testing.assert_allclose(b.model.actuator_ctrlrange[actuator[0]],[-.5,.5],atol=1e-12)
        assert b.origin_ctrl[actuator[0]]==pytest.approx(0.)
    for value,policy_value in ((-.5,0.),(0.,1.),(.5,2.)):
        physical=env.scene.control_reset_action.clone()
        physical[ids]=value
        policy=env.scene.policy_action(physical)
        active_ids=env.scene.policy_ids.tolist()
        for i in ids: assert policy[active_ids.index(i)].item()==pytest.approx(policy_value)
        torch.testing.assert_close(env.scene.physical_action(policy)[ids],physical[ids])
    physical=env.scene.control_reset_action.clone(); physical[ids]=-.25
    action=env.scene.policy_action(physical)[None].repeat(2,1)
    previous=env.scene.target_motion_limiter.previous.clone()
    result=env.step(action)
    limiter=env.scene.target_motion_limiter
    speed=(limiter.previous-previous)/limiter.dt
    assert (speed.abs()<=limiter.max_velocity+1e-7).all()
    assert (speed.abs()/limiter.dt<=limiter.max_acceleration+1e-6).all()
    assert torch.isfinite(result.observation).all()
    initial=env.scene.cpu_snapshot()
    initial.qacc_warmstart[:]=env.scene.data.qacc_warmstart[0].cpu().numpy()
    contract,trace=generate(b,initial,b.preparation['control_context'],2.)
    assert contract['report']['valid']
    np.testing.assert_array_equal(trace['qpos'][0],initial.qpos)
    np.testing.assert_allclose(trace['ctrl'][-1],b.origin_ctrl,atol=1e-12)
    assert not contract['included_in_replay']
    out=ROOT/'results/pitch_range_validation';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(preparation=b.preparation['report'],ending=contract['report']),indent=2)+'\n')
    np.savez_compressed(out/'ending.npz',**trace)
