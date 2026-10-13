"""Geometry controls require explicit migration and exact replay contracts."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.finger_clearance import canonical
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser

OPTIONS = dict(schema=1, mode='full_finger_clearance_v1')
CONTRACT = dict(schema=1, mode='full_finger_clearance_v1', geometry='test fixture only',
                key_top_z=.0225, finger_geoms=['fixture_capsule'])


class ClearanceCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.path=self.root/'best.pt';self.metadata=dict(scene_sha256='s', task_sha256='t', stage=1)
        self.config=SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        self.agent=SAC(7,3,self.config,device='cpu',seed=3)

    def tearDown(self):self.temporary.cleanup()
    def save(self, metadata):checkpoint.save(self.path,self.agent,metadata,torch.Generator().manual_seed(9))
    def fresh(self):return SAC(7,3,self.config,device='cpu',seed=11)
    def replay(self, **kwargs):
        return checkpoint.load_agent(self.path,scene_sha256='s',task_sha256='t',obs_size=7,action_size=3,device='cpu',**kwargs)

    def test_legacy_none_and_enabled_exact_replay(self):
        self.save(self.metadata)
        self.assertNotIn('finger_clearance',checkpoint.reward_options(self.path))
        checkpoint.load(self.path,self.fresh(),dict(self.metadata,finger_clearance=None,finger_clearance_contract=None))
        self.replay(finger_clearance=None,finger_clearance_contract=None)
        self.save(dict(self.metadata,finger_clearance=OPTIONS,finger_clearance_contract=CONTRACT))
        options=checkpoint.reward_options(self.path)
        self.assertEqual(options['finger_clearance'],canonical(OPTIONS))
        loaded=self.replay(**options,finger_clearance_contract=deepcopy(CONTRACT))
        x=torch.ones(1,7)
        torch.testing.assert_close(loaded.act(x,deterministic=True),self.agent.act(x,deterministic=True),rtol=0,atol=0)
        for changed in (None,dict(CONTRACT,key_top_z=.03)):
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'clearance|Clearance'):
                self.replay(**options,finger_clearance_contract=changed)
        with self.assertRaisesRegex(ValueError,'clearance'):self.replay()

    def test_changed_guard_requires_explicit_finetune_and_other_guards_remain(self):
        self.save(self.metadata)
        target=dict(self.metadata,finger_clearance=OPTIONS,finger_clearance_contract=CONTRACT)
        for kwargs in ({},{'weights_only':True},{'weights_only':True,'transfer_mode':'finetune'},
                       {'allow_control_change':True},{'weights_only':True,'allow_control_change':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                checkpoint.load(self.path,self.fresh(),target,**kwargs)
        fresh=self.fresh();rng=fresh.update_rng.get_state().clone()
        checkpoint.load(self.path,fresh,target,weights_only=True,transfer_mode='finetune',allow_control_change=True)
        self.assertEqual(fresh.updates,0);self.assertFalse(fresh.actor_optimizer.state)
        self.assertTrue(torch.equal(rng,fresh.update_rng.get_state()))
        for changed in (dict(target,scene_sha256='other'),dict(target,task_sha256='other'),
                        dict(target,preparation={'duration':1}),dict(target,self_collision_weight=.2)):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                checkpoint.load(self.path,self.fresh(),changed,weights_only=True,
                                transfer_mode='finetune',allow_control_change=True)

    def test_incomplete_geometry_contract_cannot_be_bypassed(self):
        self.save(self.metadata)
        for target in (dict(self.metadata,finger_clearance=OPTIONS),dict(self.metadata,finger_clearance_contract=CONTRACT)):
            with self.subTest(target=target),self.assertRaisesRegex(ValueError,'both options'):
                checkpoint.load(self.path,self.fresh(),target,weights_only=True,
                                transfer_mode='finetune',allow_control_change=True)

    def test_cli_and_source_config_strictly_include_geometry(self):
        path=self.root/'guard.json';path.write_text(json.dumps(OPTIONS))
        args=parser().parse_args(['--finger-clearance-config',str(path)])
        self.assertEqual(effective_config(args)['finger_clearance'],canonical(OPTIONS))
        self.assertIsNone(effective_config(parser().parse_args([]))['finger_clearance'])
        validate=runpy.run_path(str(Path(__file__).parents[1]/'scripts/eval_from_source.py'))['validate_config']
        config=dict(self.metadata,source_runtime={'fixture':True},finger_clearance=canonical(OPTIONS),finger_clearance_contract=CONTRACT)
        saved=dict(config,vector_step=3,updates=2,last_evaluation={})
        validate(config,saved)
        for changed in (dict(config,finger_clearance=None),dict(config,finger_clearance_contract=None),
                        {k:v for k,v in config.items() if k!='finger_clearance_contract'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):validate(changed,saved)


if __name__=='__main__':unittest.main()
