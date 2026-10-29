"""Analytic reward formulas and explicit-mapping gates, independent of physics."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Requires training_mjlab/.venv')
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate, canonical, FINGER_NAMES
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
from tests.test_mjlab_task import fixture, snapshot
from tests.test_torch_sac import agent


def scene_fixture():
    names=[]
    for side in ('R','L'):
        names += [f'{side}_Thumb_{j}' for j in ('Rot','Rot2','MCP','PIP')]
        names += [f'{side}_{f}_{j}' for f in ('Index','Mid','Ring','Pinky') for j in ('MCP','SW','PIP','DIP')]
        names += [f'{side}_Palm_Base_{j}' for j in ('Tx','Ty','Tz','Pitch','Yaw','Roll')]
    return dict(action_names=names+['sustain'], action_min=[-3.]*52+[0.],
                action_max=[3.]*52+[1.], neutral_action=[0.]*53, control_dt=.05)


def only(*names, stage='current'):
    c=template(stage)
    for name,t in c['terms'].items():t['enabled']=name in names
    return c


def explicit_test_mappings(c):
    """Synthetic arithmetic fixtures, NOT TsenseI physical calibration/defaults."""
    c['action_units']='physical'
    c['regularization_shaping']='identity'
    c['terms']['wrist_pitch_reward']['params']['directions']={'R_Palm_Base_Pitch':1,'L_Palm_Base_Pitch':-1}
    c['terms']['forward_press_reward']['params']['angle_map']={
        f:dict(joints={f+'_MCP':1.},offset=0.) for f in FINGER_NAMES}
    return c


class PaperRewardsTest(unittest.TestCase):
    def inputs(self,c,device='cpu'):
        bundle=fixture();logic=MusicTaskLogic(bundle,num_envs=2,device=device)
        s={k:v.to(device) for k,v in snapshot().items()}
        s.update(signed_power=torch.zeros(2,52,device=device),
                 self_collision_pairs=torch.tensor([0,3],device=device),
                 self_collision_pair_substeps=torch.tensor([0,3],device=device),
                 undesired_collision_pairs=torch.tensor([0,3],device=device))
        base,_=logic.reward_terms(s,logic.index)
        r=RewardManager(c,scene_fixture(),bundle.manifest,num_envs=2,device=device)
        action=torch.zeros(2,53,device=device)
        return r,action,s,base,logic.goal[logic.index],logic.fingers[logic.index],logic

    def call(self,data):
        r,a,s,b,t,f,_=data
        return r(b,a,s,t,f)

    def test_missing_user_mappings_and_conflicts_fail(self):
        for name in ('forearm_velocity_reward','wrist_acceleration_reward','finger_speed_reward',
                     'wrist_pitch_reward','forward_press_reward'):
            c=only(name)
            with self.subTest(name=name),self.assertRaises(ValueError):validate(c)
        c=only('energy_reward','finger_power_reward')
        with self.assertRaises(ValueError):validate(c)
        for name in ('collision_reward','natural_posture_reward'):
            c=only();c['terms'][name]=dict(enabled=False,weight=1.,params={})
            with self.assertRaises(ValueError):validate(c)
        c=only('finger_power_reward',stage='stage2_other')
        with self.assertRaises(ValueError):validate(c)
        c['regularization_shaping']='identity';validate(c)
        c=only('key_press_reward',stage='stage1');c['terms']['key_press_reward']['params']['chord_bonus']=True
        with self.assertRaises(ValueError):validate(c)
        for value in (float('nan'),-1,True):
            c=only('self_collision_reward');c['terms']['self_collision_reward']['weight']=value
            with self.assertRaises(ValueError):validate(c)
        c=template();c['typo']=True
        with self.assertRaises(ValueError):validate(c)

    def test_templates_and_stage_weights(self):
        root=Path(__file__).resolve().parents[1]/'configs/rewards'
        validate(json.loads((root/'default_20261006.json').read_text()))
        # Old contracts are rejected without depending on archived runtime files.
        old=template();old['contract']='physical_key_travel_exact_v1'
        with self.assertRaises(ValueError):validate(old)
        self.assertEqual(len(template()['terms']),24)
        s1=template('stage1');s2=template('stage2_canon');other=template('stage2_other')
        self.assertEqual(s1['terms']['wrist_velocity_reward']['weight'],.176)
        self.assertEqual(s2['terms']['forearm_acceleration_reward']['weight'],6.4)
        self.assertEqual(other['terms']['forearm_acceleration_reward']['weight'],5.5)
        self.assertEqual(s2['terms']['finger_speed_reward']['weight'],.4)
        self.assertEqual(other['terms']['finger_speed_reward']['weight'],.3)
        self.assertFalse(s1['terms']['finger_speed_reward']['enabled'])
        self.assertIsNone(s1['terms']['forward_press_reward']['params']['angle_map'])
        self.assertIsNone(s1['terms']['wrist_pitch_reward']['params']['directions'])

    def test_key_wrong_key_chord_silence_and_dt(self):
        c=only('key_press_reward',stage='stage1')
        data=self.inputs(c);data[2]['key_activation'][1,70]=True
        terms=self.call(data)
        torch.testing.assert_close(terms['key_press_reward'],torch.tensor([.05,(.6+.4*math.exp(-1.25))*.05]))
        for scope in ('on','whole_key'):
            c['terms']['key_press_reward']['params'].update(chord_bonus=True,chord_scope=scope)
            d=self.inputs(c);d[2]['key_activation'][1,70]=True
            expected=(.6*1.08+.4*math.exp(-1.25)) if scope=='on' else (.6+.4*math.exp(-1.25))*1.08
            self.call(d)
            self.assertAlmostEqual(d[0].raw_terms['key_press_reward'][1].item(),expected,places=6)
        d=self.inputs(only('key_press_reward',stage='stage1'));d[4].zero_();d[2]['key_activation'].zero_()
        self.assertAlmostEqual(self.call(d)['key_press_reward'][0].item(),.4*.05,places=6)

    def test_fingering_averages_active_fingers(self):
        c=only('fingering_reward',stage='stage1');d=self.inputs(c)
        self.call(d);torch.testing.assert_close(d[0].raw_terms['fingering_reward'],torch.ones(2))
        # One target finger is .11 m away: tolerance .1. Other finger remains 1.
        d=self.inputs(c);d[2]['fingertip_history'][:,:,1,0]=.11
        self.call(d);torch.testing.assert_close(d[0].raw_terms['fingering_reward'],torch.full((2,),.55))
        d[2]['exact_finger_key_history'].fill_(-1);self.call(d);torch.testing.assert_close(d[0].raw_terms['fingering_reward'],torch.zeros(2))

    def test_power_grouping_positive_only_and_pedal_excluded(self):
        c=only('forearm_power_reward','wrist_power_reward','finger_power_reward',stage='stage1')
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            d=self.inputs(c,device);d[2]['signed_power'].fill_(-4.)
            r=d[0]
            for group in ('forearm','wrist','finger'):
                d[2]['signed_power'][:,r.groups[group][0]]=2.
            self.call(d)
            for group,k,w in [('forearm',-.001,.2),('wrist',-.01,.176),('finger',-.005,.2)]:
                self.assertAlmostEqual(r.raw_terms[f'{group}_power_reward'][0].item(),k*2,places=6)
                self.assertAlmostEqual(r.weighted_terms[f'{group}_power_reward'][0].item(),k*2*w,places=6)

    def test_derivatives_first_action_second_difference_and_partial_reset(self):
        names=[f'{g}_{k}_reward' for g in ('forearm','wrist','finger') for k in ('velocity','acceleration')]
        c=only(*names,'finger_speed_reward');c['action_units']='physical'
        d=self.inputs(c);r,a=d[:2]
        for group in ('forearm','wrist','finger'):a[:,r.groups[group][0]]=1.
        a[:,-1]=1.  # Pedal must not contribute.
        self.call(d)
        for group,k in [('forearm',-.02),('wrist',-.005),('finger',-.0002)]:
            self.assertAlmostEqual(r.raw_terms[f'{group}_velocity_reward'][0].item(),k*20,places=6)
        for group,k in [('forearm',-.002),('wrist',-.0003),('finger',-.00001)]:
            self.assertAlmostEqual(r.raw_terms[f'{group}_acceleration_reward'][0].item(),k*400,places=6)
        self.assertAlmostEqual(r.raw_terms['finger_speed_reward'][0].item(),-.05*(20-6.5),places=6)
        self.call(d);self.assertEqual(r.raw_terms['finger_velocity_reward'][0].item(),0.)
        self.assertAlmostEqual(r.raw_terms['finger_acceleration_reward'][0].item(),-.004,places=6)
        self.call(d);self.assertEqual(r.raw_terms['finger_acceleration_reward'][0].item(),0.)
        r.reset(torch.tensor([0]));self.call(d)
        torch.testing.assert_close(r.raw_terms['finger_velocity_reward'],torch.tensor([-.004,0.]))
        torch.testing.assert_close(r.previous[1],a[1,:52])

    def test_posture_pitch_and_explicit_press_angle(self):
        c=explicit_test_mappings(only('wrist_pitch_reward','forward_press_reward'))
        d=self.inputs(c);r,a=d[:2]
        a[:,r.groups['forearm']]=2.
        a[:,23]=.1;a[:,48]=-.1
        a[1,2]=1.7 # Synthetic map uses R_Thumb_MCP only.
        self.call(d)
        torch.testing.assert_close(r.raw_terms['wrist_pitch_reward'],torch.tensor([.1,.1]))
        torch.testing.assert_close(r.raw_terms['forward_press_reward'],torch.tensor([0.,-.1]))
        c['terms']['forward_press_reward']['params']['angle_map']['R_Thumb']['joints']={'R_Thumb_typo':1}
        with self.assertRaises(ValueError):self.inputs(c)

    def test_self_collision_pair_penalty(self):
        d=self.inputs(only('self_collision_reward'))
        torch.testing.assert_close(self.call(d)['self_collision_reward'],torch.tensor([0.,-.6]))

    def test_reward_enters_observation_returns_and_all_off(self):
        c=only('self_collision_reward');c['scale_by_dt']=True
        d=self.inputs(c);r,a,s,_,_,_,logic=d
        logic.reset(torch.arange(2),s)
        result=logic.advance(a,s,reward_transform=r)
        torch.testing.assert_close(result.reward,torch.tensor([0.,-.6]))
        torch.testing.assert_close(logic.last_reward,result.reward)
        torch.testing.assert_close(result.info['episode_return'],result.reward)
        observation=result.observation.clone()
        logic.reset(torch.tensor([1]),s);r.reset(torch.tensor([1]))
        torch.testing.assert_close(result.observation,observation)
        self.assertEqual(logic.returns[1],0.)
        d=self.inputs(only());r,a,s,_,_,_,logic=d
        logic.reset(torch.arange(2),s)
        result=logic.advance(a,s,reward_transform=r)
        self.assertEqual(result.info['reward_terms'],{})
        torch.testing.assert_close(result.reward,torch.zeros(2))

    def test_checkpoint_restores_full_contract_and_blocks_silent_changes(self):
        c=template('stage1');meta=dict(scene_sha256='s',task_sha256='t',stage=1,reward_config=c)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            checkpoint.save(path,agent(),meta,torch.Generator().manual_seed(7))
            self.assertEqual(checkpoint.reward_options(path),{'reward_config':c})
            checkpoint.load_agent(path,scene_sha256='s',task_sha256='t',obs_size=7,action_size=3,
                                  device='cpu',reward_config=c)
            with self.assertRaises(ValueError):
                checkpoint.load_agent(path,scene_sha256='s',task_sha256='t',obs_size=7,action_size=3,device='cpu')
            changed=deepcopy(meta);changed['reward_config']['scale_by_dt']=False
            with self.assertRaises(ValueError):checkpoint.load(path,agent(),changed)
            with self.assertRaises(ValueError):checkpoint.load(path,agent(),changed,allow_reward_change=True)
            second=agent(stage=2)
            checkpoint.load(path,second,changed,weights_only=True,allow_reward_change=True)
            self.assertEqual(second.updates,0)
            self.assertEqual(len(second.actor_optimizer.state),0)
            self.assertAlmostEqual(second.temperature.item(),.2)

    def test_cli_switches_and_invalid_enabled_mapping(self):
        args=parser().parse_args(['--inspect-only','--enable-reward','self_collision_reward'])
        c=effective_config(args)['reward_config']
        self.assertTrue(c['terms']['self_collision_reward']['enabled'])
        with self.assertRaises(ValueError):
            effective_config(parser().parse_args(['--inspect-only','--enable-reward','forward_press_reward']))
        with self.assertRaises(ValueError):
            effective_config(parser().parse_args(['--inspect-only','--enable-reward','sustain_reward',
                                                 '--disable-reward','sustain_reward']))

if __name__=='__main__':unittest.main()
