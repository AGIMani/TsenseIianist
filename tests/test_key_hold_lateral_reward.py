"""Analytic contracts: no model construction or physics execution."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from tests.test_paper_rewards import scene_fixture
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager

TERM = 'key_hold_lateral_motion_penalty'
PARAMS = dict(velocity_tolerance_m_s=.01, full_penalty_velocity_m_s=.10,
              maximum_penalty_per_step=.5)


def config():
    c = template()
    assert TERM in c['terms'], 'Missing approved key-hold lateral penalty'
    for name, term in c['terms'].items():
        term['enabled'] = name == TERM
    return c


def score(speed, held, c=None):
    speed = torch.tensor(speed, dtype=torch.float32)
    held = torch.tensor(held, dtype=torch.bool)
    scene = scene_fixture()
    manager = RewardManager(config() if c is None else c, scene, {}, num_envs=len(speed), device='cpu')
    action = torch.tensor(scene['neutral_action']).repeat(len(speed), 1)
    return manager({}, action, dict(key_hold_lateral_velocity_history=speed,
        key_hold_contact_history=held), None, None)[TERM]


def test_registered_defaults_and_enabled_release_json():
    c = config()
    assert c['terms'][TERM]['params'] == PARAMS
    d = json.loads((Path(__file__).resolve().parents[1]/'configs/rewards/default_20261006.json').read_text())
    assert d['terms'][TERM] == dict(enabled=True, weight=1., params=PARAMS)
    validate(d)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_boundaries_and_actual_velocity_sign(sign):
    speeds = [[[sign*v, 0.]] for v in [0., .009, .01, .055, .10, 2.]]
    got = score(speeds, [[[True, False]]]*6)
    torch.testing.assert_close(got, torch.tensor([0., 0., 0., -.125, -.5, -.5]))


def test_per_hand_gate_and_maximum_not_sum():
    got = score([[[.1,.1]], [[0.,.1]], [[.1,.1]], [[.1,0.]]],
                [[[False,False]], [[True,False]], [[True,True]], [[True,False]]])
    torch.testing.assert_close(got, torch.tensor([0.,0.,-.5,-.5]))


@pytest.mark.parametrize('substeps',[10,50])
def test_mean_physical_samples_and_no_dt_rescaling(substeps):
    c=config(); c['scale_by_dt']=True; c['reference_dt']=.1
    got = score([[[.1,0.]]*substeps], [[[True,False]]*(substeps//2)+[[False,False]]*(substeps//2)], c)
    torch.testing.assert_close(got, torch.tensor([-.25]))


@pytest.mark.parametrize('key,value',[("velocity_tolerance_m_s",-1.),("velocity_tolerance_m_s",float('nan')),
    ("full_penalty_velocity_m_s",.01),("full_penalty_velocity_m_s",0.),
    ("maximum_penalty_per_step",-1.),("maximum_penalty_per_step",float('inf'))])
def test_invalid_parameters_rejected(key,value):
    c=config(); c['terms'][TERM]['params'][key]=value
    with pytest.raises(ValueError): validate(c)


def test_missing_new_term_or_samples_rejected():
    c=config(); del c['terms'][TERM]
    with pytest.raises(ValueError): validate(c)
    c=config(); scene=scene_fixture()
    manager=RewardManager(c,scene,{},num_envs=1,device='cpu')
    with pytest.raises((KeyError,ValueError)):
        manager({},torch.tensor(scene['neutral_action'])[None],{},None,None)


def test_weight_applied_once_final_cap_and_cli_disable():
    c=config(); c['terms'][TERM]['weight']=.5
    torch.testing.assert_close(score([[[.055,0.]]],[[[True,False]]],c),torch.tensor([-.0625]))
    c['terms'][TERM]['weight']=2.
    torch.testing.assert_close(score([[[.1,0.]]],[[[True,False]]],c),torch.tensor([-.5]))
    from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
    path=Path(__file__).resolve().parents[1]/'configs/rewards/default_20261006.json'
    result=effective_config(parser().parse_args(['--reward-config',str(path),'--disable-reward',TERM]))
    assert not result['reward_config']['terms'][TERM]['enabled']
    assert result['reward_config']['terms'][TERM]['params']==PARAMS
