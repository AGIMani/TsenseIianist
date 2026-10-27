"""Run physical model and Warp tests only inside remote Slurm."""
import importlib.util
import os
from types import SimpleNamespace
import numpy as np
import pytest


def test_collector_available():
    assert importlib.util.find_spec('tsensei_v3_2_pianist.mjlab_env.key_hold_lateral') is not None


@pytest.fixture
def sample_fixture():
    if not os.environ.get('SLURM_JOB_ID'):
        pytest.skip('Model/GPU tests require remote Slurm')
    import mujoco
    import torch
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.key_hold_lateral import KeyHoldLateralCollector
    # Ancestor rotation and a nonzero palm origin offset exercise spatial velocity.
    xml='''<mujoco><worldbody>
      <body name="rp" euler="0 0 35"><joint type="slide" axis="0 1 0"/>
        <geom size=".02" mass="1"/><body name="rh" pos=".1 0 0">
        <joint type="hinge" axis="0 0 1"/><geom size=".02" mass="1"/>
        <body name="right_/base_link" pos=".15 .03 0">
          <geom name="right_/base_link_collision_0" size=".02" mass="1"/>
          <geom name="right_/visual" size=".01" contype="0" conaffinity="0"/>
        </body></body></body>
      <body name="left_/base_link"><joint type="slide" axis="0 1 0"/>
        <geom name="left_/base_link_collision_0" size=".02" mass="1"/>
      </body><body name="key"><joint type="hinge" axis="0 1 0"/>
        <geom name="keygeom" size=".02" mass="1"/>
      </body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    d=mujoco.MjData(model); d.qpos[:]=[.03,.2,-.04,0.]; d.qvel[:]=[.15,.6,-.3,0.]
    mujoco.mj_forward(model,d)
    device='cuda:0'; keys=[model.geom('keygeom').id]
    sampler=KeyHoldLateralCollector(model,keys,num_envs=2,decimation=10,device=device)
    right=model.geom('right_/base_link_collision_0').id
    left=model.geom('left_/base_link_collision_0').id
    visual=model.geom('right_/visual').id; key=keys[0]
    pairs=[[right,key],[key,right],[left,key],[visual,key],[-1,key],[right,key]]
    data=SimpleNamespace(qvel=wp.array(np.tile(d.qvel,(2,1)),dtype=wp.float32,device=device),
        xpos=wp.array(np.tile(d.xpos,(2,1,1)),dtype=wp.vec3,device=device),
        xaxis=wp.array(np.tile(d.xaxis,(2,1,1)),dtype=wp.vec3,device=device),
        xanchor=wp.array(np.tile(d.xanchor,(2,1,1)),dtype=wp.vec3,device=device),
        naconmax=len(pairs),nacon=wp.array([len(pairs)],dtype=wp.int32,device=device),
        contact=SimpleNamespace(geom=wp.array(pairs,dtype=wp.vec2i,device=device),
            dist=wp.array([0.,-.01,.001,0.,0.,0.],dtype=wp.float32,device=device),
            worldid=wp.array([0,0,0,1,0,1],dtype=wp.int32,device=device)))
    active=torch.zeros((2,10,1),device=device,dtype=torch.bool);active[0,:,0]=True
    return sampler,data,active,model,d


def test_attribution_duplicate_contacts_and_activation(sample_fixture):
    import torch
    sampler,data,active,_,_=sample_fixture
    sampler.sample(data,active,0)
    assert sampler.contact_history[:,0].tolist()==[[1,0],[0,0]]
    active[1,1,0]=True
    sampler.sample(data,active,1)
    assert sampler.contact_history[:,1].tolist()==[[1,0],[1,0]]
    # Separation does not count; zero distance does. A reversed geom pair counts once.
    import warp as wp
    wp.to_torch(data.contact.dist)[2]=0.
    sampler.sample(data,active,2)
    assert sampler.contact_history[:,2].tolist()==[[1,1],[0,0]]
    sampler.clear(torch.tensor([0],device='cuda:0'))
    assert sampler.contact_history[0].count_nonzero().item()==0
    assert sampler.contact_history[1,1,0].item()==1
    assert sampler.velocity_history[0].count_nonzero().item()==0


def test_actual_origin_velocity_matches_mujoco_and_capture(sample_fixture):
    import mujoco
    import torch
    import warp as wp
    sampler,data,active,model,d=sample_fixture
    sampler.sample(data,active,0)
    expected=[]
    for name in ('right_/base_link','left_/base_link'):
        jac=np.zeros((3,model.nv));rot=np.zeros_like(jac)
        mujoco.mj_jacBody(model,d,jac,rot,model.body(name).id)
        expected.append((jac@d.qvel)[1])
    np.testing.assert_allclose(sampler.velocity_history[0,0].cpu(),expected,rtol=1e-5,atol=1e-6)
    sampler.clear()
    with wp.ScopedCapture() as cap:
        sampler.sample(data,active,0)
    wp.capture_launch(cap.graph); wp.synchronize()
    assert sampler.contact_history[:,0].tolist()==[[1,0],[0,0]]
    np.testing.assert_allclose(sampler.velocity_history[0,0].cpu(),expected,rtol=1e-5,atol=1e-6)
    # No previous velocity state survives a reset.
    sampler.clear(); wp.to_torch(data.qvel).zero_(); sampler.sample(data,active,0)
    assert sampler.velocity_history.count_nonzero().item()==0


def test_real_solver_contacts_gate_only_held_hand():
    if not os.environ.get('SLURM_JOB_ID'):
        pytest.skip('Physical model test requires remote Slurm')
    import mujoco
    import torch
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.substep_rewards import SubstepRewardCollector
    xml='''<mujoco><worldbody>
      <body name="right_/base_link" pos="0 0 .035"><joint type="slide" axis="0 1 0"/>
        <geom name="right_/base_link_collision_0" type="sphere" size=".02" mass="1"/>
      </body><body name="left_/base_link" pos="0 .3 .2"><joint type="slide" axis="0 1 0"/>
        <geom name="left_/base_link_collision_0" size=".02" mass="1"/>
      </body><body name="key"><joint type="slide" axis="0 0 -1" range="0 .01"/>
        <geom name="keygeom" type="box" size=".05 .05 .03" mass="1"/>
      </body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(model)
    d.qpos[2]=.01; d.qvel[:]=[.055,.3,0.];mujoco.mj_forward(model,d)
    assert d.ncon>0
    device='cuda:0'
    data=SimpleNamespace(qpos=wp.array(d.qpos[None],dtype=wp.float32,device=device),
        qvel=wp.array(d.qvel[None],dtype=wp.float32,device=device),
        xpos=wp.array(d.xpos[None],dtype=wp.vec3,device=device),
        xaxis=wp.array(d.xaxis[None],dtype=wp.vec3,device=device),
        xanchor=wp.array(d.xanchor[None],dtype=wp.vec3,device=device),
        actuator_force=wp.zeros((1,1),dtype=wp.float32,device=device),
        actuator_velocity=wp.zeros((1,1),dtype=wp.float32,device=device),
        naconmax=d.ncon,nacon=wp.array([d.ncon],dtype=wp.int32,device=device),
        contact=SimpleNamespace(geom=wp.array(d.contact.geom[:d.ncon],dtype=wp.vec2i,device=device),
            dist=wp.array(d.contact.dist[:d.ncon],dtype=wp.float32,device=device),
            worldid=wp.zeros(d.ncon,dtype=wp.int32,device=device)))
    collector=SubstepRewardCollector(num_envs=1,decimation=10,device=device,key_qadr=[2],
        key_limits=[[0,.01]],action_ids=[],geom_pair_mask=np.zeros((model.ngeom,model.ngeom),dtype=np.int32),
        key_hold_model=model,key_hold_geom_ids=[model.geom('keygeom').id])
    sim=SimpleNamespace(wp_data=data)
    collector.sample(sim,0)
    assert collector.key_hold_lateral.contact_history[0,0].tolist()==[1,0]
    np.testing.assert_allclose(collector.key_hold_lateral.velocity_history[0,0].cpu(),[.055,.3])
    # Just below activation, real contact remains but the penalty gate is off.
    wp.to_torch(data.qpos)[0,2]=.00999
    collector.sample(sim,1)
    assert collector.key_hold_lateral.contact_history[0,1].tolist()==[0,0]
    before=collector.key_hold_lateral.contact_history.clone()
    collector.sample_contacts(sim)
    torch.testing.assert_close(collector.key_hold_lateral.contact_history,before)


def test_no_hardlimit_poststep_contact_enter_exit_and_capture():
    if not os.environ.get('SLURM_JOB_ID'):
        pytest.skip('Model/GPU test requires remote Slurm')
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    from tsensei_v3_2_pianist.mjlab_env.substep_rewards import SubstepRewardCollector
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="right_/base_link" pos="0 0 .035"><joint type="slide" axis="0 1 0"/>
        <geom name="right_/base_link_collision_0" size=".02" mass="1"/>
      </body><body name="left_/base_link" pos="0 .3 .2"><joint type="slide" axis="0 1 0"/>
        <geom name="left_/base_link_collision_0" size=".02" mass="1"/>
      </body><body name="key"><joint type="slide" axis="0 0 -1" range="0 .01"/>
        <geom name="keygeom" type="box" size=".05 .05 .03" mass="1"/>
      </body></worldbody></mujoco>''')
    with wp.ScopedDevice('cuda:0'):
        m=mw.put_model(model);d=mw.make_data(model,nworld=1,nconmax=64,njmax=128)
        q=wp.to_torch(d.qpos);q[0,2]=.01
        wp.to_torch(d.qvel)[0,0]=.055
        mw.forward(m,d)
        collector=SubstepRewardCollector(num_envs=1,decimation=10,device='cuda:0',key_qadr=[2],
            key_limits=[[0,.01]],action_ids=[],geom_pair_mask=np.zeros((model.ngeom,model.ngeom),dtype=np.int32),
            key_hold_model=model,key_hold_geom_ids=[model.geom('keygeom').id])
        scene=SimpleNamespace(reward_sampling=collector,joint_limiter=None,arm_follower=None,
            sim=SimpleNamespace(wp_model=m,wp_data=d),hand_collision_detection_only=False)
        PianoScene._sample_reward_substep(scene,0)
        assert collector.key_hold_lateral.contact_history[0,0].tolist()==[1,0]
        # Leave the old contact buffer untouched while moving out of contact.
        q[0,0]=.3
        PianoScene._sample_reward_substep(scene,1)
        assert collector.key_hold_lateral.contact_history[0,1].tolist()==[0,0]
        q[0,0]=0.
        PianoScene._sample_reward_substep(scene,2)
        assert collector.key_hold_lateral.contact_history[0,2].tolist()==[1,0]
        with wp.ScopedCapture() as capture:
            PianoScene._sample_reward_substep(scene,3)
        wp.capture_launch(capture.graph);wp.synchronize()
        assert collector.key_hold_lateral.contact_history[0,3].tolist()==[1,0]
        q[0,0]=.3
        wp.capture_launch(capture.graph);wp.synchronize()
        assert collector.key_hold_lateral.contact_history[0,3].tolist()==[0,0]
