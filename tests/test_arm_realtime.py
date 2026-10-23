"""Safety/equivalence checks for the optional single-scene native CPU backend."""
import argparse
import numpy as np
import pytest
import torch

pytest.importorskip('numba')
from tsensei_v3_2_pianist.mjlab_env.arm_options import add_arguments, from_args
from tsensei_v3_2_pianist.mjlab_env.arm_follow import ArmFollower
from tsensei_v3_2_pianist.mjlab_env.arm_realtime import RealtimeArmFollower, _segment, _fk


@pytest.fixture
def followers():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();add_arguments(parser)
    angles=['40','-30','20','-50','40','10','-20']
    options=from_args(parser.parse_args(['--arms','--arm-initial-right-deg',*angles,
                                      '--arm-initial-left-deg',*angles]))
    poses={s:(np.array([.2,y,.15]),np.eye(3)) for s,y in [('right',.1),('left',-.1)]}
    fast=RealtimeArmFollower(options,poses);fast.warmup()
    return fast,ArmFollower(options,poses,num_envs=1,dt=.005),poses


def test_same_targets_same_bounded_trajectory(followers):
    fast,reference,poses=followers
    for i in range(12):
        fast.begin_control();reference.begin_control()
        targets={s:(p+np.array([.015 if i<6 else -.01,.007,.01]),r) for s,(p,r) in poses.items()}
        fast.step(targets)
        reference.step({s:tuple(torch.tensor(x[None]) for x in pair) for s,pair in targets.items()})
        for s in poses:
            for name,tol in [('q',1e-8),('v',1e-6),('acceleration',2e-4),('reference',1e-7)]:
                np.testing.assert_allclose(fast.sides[s][name],reference.sides[s][name][0].numpy(),atol=tol,rtol=0)
        np.testing.assert_allclose(fast.observation(),reference.observation()[0].numpy(),atol=1e-6,rtol=0)
        assert abs(fast.penalty()-reference.penalty()[0].item())<1e-5


def _target_from_native_q(a,q):
    return _fk(q,*[a[k] for k in ('joint_pos','joint_rot','axis','base_pos','base_rot',
                                  'tool_pos','tool_rot')],False)[:2]


def test_converged_reference_still_executes_bounded_fixed_target_trajectory(followers):
    fast,reference,_=followers
    targets={};starts={}
    for side,a in fast.sides.items():
        starts[side]=a['q'].copy()
        goal=a['q'].copy();goal[0]+=.02
        a['reference'][:]=goal
        reference.sides[side]['reference'].copy_(torch.tensor(goal[None]))
        targets[side]=_target_from_native_q(a,goal)
    saw_braking=False
    for i in range(32):
        before={s:(a['q'].copy(),a['v'].copy()) for s,a in fast.sides.items()}
        fast.step(targets)
        reference.step({s:tuple(torch.tensor(x[None]) for x in pair) for s,pair in targets.items()})
        for side,a in fast.sides.items():
            for name,tol in [('q',1e-8),('v',1e-6),('acceleration',2e-4),('reference',1e-7)]:
                np.testing.assert_allclose(a[name],reference.sides[side][name][0].numpy(),atol=tol,rtol=0)
            np.testing.assert_allclose((a['q']-before[side][0])/fast.dt,a['v'],atol=1e-10,rtol=0)
            np.testing.assert_allclose((a['v']-before[side][1])/fast.dt,a['acceleration'],atol=1e-8,rtol=0)
            assert np.all(np.abs(a['v'])<=a['vmax']+1e-8)
            assert np.all(np.abs(a['acceleration'])<=a['amax']+1e-7)
            if i==0:
                assert starts[side][0]<a['q'][0]<a['reference'][0]-.01
            saw_braking |= a['acceleration'][0]<-1e-3
    assert saw_braking
    for a in fast.sides.values():
        np.testing.assert_allclose(a['q'],a['reference'],atol=1e-10,rtol=0)


@pytest.mark.parametrize('direction',[-1,1])
def test_converged_reference_still_brakes_near_joint_position_limit(followers,direction):
    fast,reference,_=followers
    targets={}
    for side,a in fast.sides.items():
        goal=a['q'].copy();goal[0]=a['high' if direction>0 else 'low'][0]
        a['q'][0]=goal[0]-direction*.001
        a['v'][0]=direction*.1
        a['reference'][:]=goal
        for name in ('q','v','reference'):
            reference.sides[side][name].copy_(torch.tensor(a[name][None]))
        targets[side]=_target_from_native_q(a,goal)
    saw_braking=False
    for _ in range(8):
        fast.step(targets)
        reference.step({s:tuple(torch.tensor(x[None]) for x in pair) for s,pair in targets.items()})
        for side,a in fast.sides.items():
            for name,tol in [('q',1e-8),('v',1e-6),('acceleration',2e-4),('reference',1e-7)]:
                np.testing.assert_allclose(a[name],reference.sides[side][name][0].numpy(),atol=tol,rtol=0)
            assert np.all(a['q']>=a['low']-1e-12) and np.all(a['q']<=a['high']+1e-12)
            assert np.all(np.abs(a['v'])<=a['vmax']+1e-8)
            assert np.all(np.abs(a['acceleration'])<=a['amax']+1e-7)
            saw_braking |= direction*a['acceleration'][0]<-1e-3
    assert saw_braking


def test_converged_reference_does_not_skip_final_coupling_check(followers):
    fast,_,_=followers
    a=fast.sides['left'];c=fast.options['coupling']
    # A corrupted state can have valid independent joint positions while
    # violating the coupled boundary; an exact pose match must still reject it.
    a['q'][[5,6]]=.9*a['high'][[5,6]]
    a['reference'][:]=a['q']
    assert c['coefficient']*abs(a['q'][5])+abs(a['q'][6])>c['sum_limit_rad']
    targets={s:_target_from_native_q(a,a['reference']) for s,a in fast.sides.items()}
    before={s:{k:a[k].copy() for k in ('q','v','acceleration','reference')} for s,a in fast.sides.items()}
    with pytest.raises(RuntimeError,match='CCS-696 coupling verification'):
        fast.step(targets)
    for side in before:
        for name,value in before[side].items():np.testing.assert_array_equal(fast.sides[side][name],value)
    assert fast.last=={}


def test_second_arm_failure_is_atomic(followers):
    fast,_,poses=followers
    targets={s:(p.copy(),r.copy()) for s,(p,r) in poses.items()}
    targets['right'][0][0]+=.02;targets['left'][0][0]=np.nan
    before={s:{k:a[k].copy() for k in ('q','v','acceleration','reference')} for s,a in fast.sides.items()}
    with pytest.raises(ValueError,match='Invalid palm'):fast.step(targets)
    for s in before:
        for k,x in before[s].items():np.testing.assert_array_equal(fast.sides[s][k],x)
    assert fast.last=={}


def test_coupled_projection_matches_reference(followers):
    fast,reference,_=followers
    c=fast.options['coupling'];rng=np.random.default_rng(18)
    for i in range(100):
        start=rng.uniform(-.5,.5,(1,7));end=rng.uniform(-2,2,(1,7))
        expected=reference._coupled_segment(torch.tensor(start),torch.tensor(end))[0].numpy()
        got=_segment(start[0],end[0],5,6,c['coefficient'],c['sum_limit_rad'])
        np.testing.assert_allclose(got,expected,atol=1e-12,rtol=0)
        assert c['coefficient']*abs(got[5])+abs(got[6])<=c['sum_limit_rad']+1.01e-12
