"""Native scene backend parity, batched atomicity and transform/reset checks."""
import argparse
from types import SimpleNamespace
import numpy as np
import pytest
import torch
pytest.importorskip('numba')
from tsensei_v3_2_pianist.mjlab_env.arm_options import add_arguments,from_args
from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
from tsensei_v3_2_pianist.mjlab_env.arm_native_scene import NativeSceneArmFollower


def fixture_bundle():
    p=argparse.ArgumentParser();add_arguments(p)
    angles=['40','-30','20','-50','40','10','-20']
    o=from_args(p.parse_args(['--arms','--arm-ik-backend','numba_cpu',
        '--arm-initial-right-deg',*angles,'--arm-initial-left-deg',*angles]))
    poses={s:(np.array([.2,y,.15]),np.eye(3)) for s,y in [('right',.1),('left',-.1)]}
    wrists={s:dict(indices=list(range(j*3,j*3+3)),axis=np.eye(3),kind=[2]*3,
        rest_pos=poses[s][0],rest_rot=poses[s][1],body_name=s,body_id=j,visual_ids=list(range(j*8,j*8+8)))
        for j,s in enumerate(poses)}
    return SimpleNamespace(arm_follow_spec=dict(options=o,initial_poses=poses,wrists=wrists,
        piano_pose=(np.zeros(3),np.eye(3)),parent_collision_filter={}),preparation_options=None,
        manifest={'physics_dt':.005})


@pytest.mark.parametrize('worlds',[1,8])
def test_native_scene_matches_torch_and_partial_reset(worlds):
    torch.set_num_threads(1)
    b=fixture_bundle();native=NativeSceneArmFollower(b,num_envs=worlds,device='cpu')
    ref=SceneArmFollower(b,num_envs=worlds,device='cpu')
    physical=torch.zeros((worlds,6),dtype=torch.float64)
    for i in range(4):
        physical[:,0]=.01 if i<2 else -.01
        targets=native.requests(physical,torch.zeros(worlds,dtype=torch.float64))
        native.step(targets);ref.step(targets)
        for s in native.sides:
            for k in ('q','v','acceleration','reference'):
                torch.testing.assert_close(native.sides[s][k],ref.sides[s][k],atol=1e-7,rtol=0)
            pn,qn=native.transforms()[s];pr,qr=ref.transforms()[s]
            torch.testing.assert_close(pn,pr,atol=1e-10,rtol=0)
            assert torch.all(torch.minimum((qn-qr).abs().amax(-1),(qn+qr).abs().amax(-1))<1e-10)
    before=native._native_state.copy();native.reset(torch.tensor([0]));ref.reset(torch.tensor([0]))
    if worlds>1:np.testing.assert_array_equal(native._native_state[1:],before[1:])
    native.step(targets);ref.step(targets)
    torch.testing.assert_close(native.observation(),ref.observation(),atol=1e-6,rtol=0)


@pytest.mark.parametrize('worlds',[1,8])
def test_native_scene_transforms_match_torch_across_joint_poses(worlds):
    torch.set_num_threads(1)
    b=fixture_bundle();native=NativeSceneArmFollower(b,num_envs=worlds,device='cpu')
    ref=SceneArmFollower(b,num_envs=worlds,device='cpu')
    rng=np.random.default_rng(20261018)
    coupling=native.options['coupling']
    j6,j7=coupling['j6_index'],coupling['j7_index']
    for _ in range(4):
        q=np.empty((worlds,2,7))
        for j,(side,a) in enumerate(native.sides.items()):
            low,high=a['low'].numpy(),a['high'].numpy()
            q[:,j]=rng.uniform(low,high,size=(worlds,7))
            total=coupling['coefficient']*np.abs(q[:,j,j6])+np.abs(q[:,j,j7])
            scale=np.minimum(1.,.9*coupling['sum_limit_rad']/np.maximum(total,1e-12))
            q[:,j,j6]*=scale;q[:,j,j7]*=scale
            ref.sides[side]['q'].copy_(torch.from_numpy(q[:,j]))
        # Updating the recording also invalidates the cached scene poses.
        native.load_recorded_q(q)
        native_transforms=native.transforms();reference_transforms=ref.transforms()
        for side in native.sides:
            pn,qn=native_transforms[side];pr,qr=reference_transforms[side]
            torch.testing.assert_close(pn,pr,atol=1e-10,rtol=0)
            assert torch.all(torch.minimum((qn-qr).abs().amax(-1),(qn+qr).abs().amax(-1))<1e-10)
        if worlds==1:
            pos=np.empty((16,3));quat=np.empty((16,4))
            native.publish_numpy(pos,quat)
            for side,(pr,qr) in reference_transforms.items():
                ids=native.wrists[side]['visual_ids']
                np.testing.assert_allclose(pos[ids],pr[0].numpy(),atol=1e-10,rtol=0)
                error=np.minimum(np.max(np.abs(quat[ids]-qr[0].numpy()),axis=-1),
                                 np.max(np.abs(quat[ids]+qr[0].numpy()),axis=-1))
                assert np.all(error<1e-10)


def test_native_scene_batch_failure_is_atomic():
    f=NativeSceneArmFollower(fixture_bundle(),num_envs=8,device='cpu')
    targets=f.requests(torch.zeros((8,6)),torch.zeros(8))
    old=f._native_state.copy()
    targets['left'][0][7,0]=float('nan')
    with pytest.raises(ValueError):f.step(targets)
    np.testing.assert_array_equal(f._native_state,old)
    # Internal corrupt acceleration feasibility in one world also rejects all.
    targets=f.requests(torch.zeros((8,6)),torch.zeros(8))
    f._native_state[7,1,1,0]=100.
    old=f._native_state.copy()
    with pytest.raises(RuntimeError,match='entire batch'):f.step(targets)
    np.testing.assert_array_equal(f._native_state,old)


def test_cartesian_tz_request_survives_removed_physical_slide():
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import active_indices
    ids=active_indices(31,'wuji_cartesian_v1')
    fake=SimpleNamespace(action_size=31,action_low=torch.full((31,),-1.),action_high=torch.ones(31),
        physical_low=torch.full((51,),-1.),physical_high=torch.ones(51),policy_ids=torch.tensor(ids),
        bundle=SimpleNamespace(policy_action_indices=ids,fixed_joint_values={
            'right_/forearm_tz':.111,'left_/forearm_tz':.222,
            'right_/forearm_pitch':.3,'left_/forearm_pitch':.4}))
    action=torch.zeros((1,31));action[0,list(ids).index(22)]=.25;action[0,list(ids).index(47)]=-.4
    physical=PianoScene.physical_action(fake,action)
    torch.testing.assert_close(physical[0,[22,47]],torch.tensor([.25,-.4]))
    torch.testing.assert_close(physical[0,[23,48]],torch.tensor([.3,.4]))
