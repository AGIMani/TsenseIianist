"""Preparation contracts survive learner snapshots without changing old resets."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser


OPTIONS = {'duration': 1., 'lift': .02, 'trajectory': 'quintic_v1'}
CONTRACT = dict(OPTIONS, schema=1, method='kinematic_mocap_preparation_v1',
                origin={'joint_positions': {'wrist': 0.}},
                initial={'hands': {'right': {'position': [0., 0., .12]}}},
                terminal={'joint_positions': {'wrist': 0.}})


class PreparationCheckpointChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'checkpoint.pt'
        self.config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)

    def tearDown(self):
        self.tmp.cleanup()

    def create(self, prepared=False):
        profile = 'direct_joint_v1' if prepared else 'legacy'
        count = 47 if prepared else 2
        agent = SAC(7, count, self.config, device='cpu', seed=5, action_profile=profile)
        metadata = dict(scene_sha256='scene', task_sha256='task', stage=1,
                        action_profile=profile)
        if prepared:
            metadata.update(preparation=deepcopy(OPTIONS), preparation_contract=deepcopy(CONTRACT),
                            disable_wrist_tz=True, disable_wrist_yaw=True,
                            wrist_tz_physics='joint_removed_v1')
        checkpoint.save(self.path, agent, metadata, torch.Generator().manual_seed(13))
        return agent, metadata

    def test_old_missing_preparation_matches_explicit_none(self):
        agent, metadata = self.create()
        expected = dict(metadata, preparation=None, preparation_contract=None)
        clone = SAC(7,2,self.config,device='cpu',seed=17)
        checkpoint.load(self.path,clone,expected)
        loaded_options = checkpoint.reward_options(self.path)
        self.assertNotIn('preparation',loaded_options)
        restored = checkpoint.load_agent(self.path,scene_sha256='scene',task_sha256='task',
                                         obs_size=7,action_size=2,device='cpu',**loaded_options)
        obs = torch.arange(7,dtype=torch.float32)[None,:]/10
        torch.testing.assert_close(restored.act(obs,deterministic=True),agent.act(obs,deterministic=True),rtol=0,atol=0)

    def test_changed_action_target_range_rejects_checkpoint_transfer(self):
        agent, metadata = self.create(prepared=True)
        metadata['action_contract'] = {'name': 'direct_joint_v1', 'mcp_max': 1.2}
        checkpoint.save(self.path, agent, metadata, torch.Generator().manual_seed(13))
        destination = SAC(7, 47, self.config, device='cpu', seed=17,
                          action_profile='direct_joint_v1')
        expected = deepcopy(metadata)
        expected['action_contract']['mcp_max'] = 1.3
        with self.assertRaisesRegex(ValueError, 'action mapping range mismatch'):
            checkpoint.load(self.path, destination, expected,
                            weights_only=True, transfer_mode='finetune')

    def test_new_options_reconstruct_identical_policy_and_controls(self):
        agent, metadata = self.create(prepared=True)
        options = checkpoint.reward_options(self.path)
        self.assertEqual(options['preparation'], OPTIONS)
        self.assertTrue(options['disable_wrist_tz'])
        self.assertTrue(options['disable_wrist_yaw'])
        restored = checkpoint.load_agent(self.path,scene_sha256='scene',task_sha256='task',
                                         obs_size=7,action_size=47,device='cpu',
                                         preparation_contract=deepcopy(CONTRACT),**options)
        obs = torch.arange(7,dtype=torch.float32)[None,:]/10
        torch.testing.assert_close(restored.act(obs,deterministic=True),agent.act(obs,deterministic=True),rtol=0,atol=0)
        checkpoint.load(self.path,restored,metadata)
        options['preparation'] = dict(OPTIONS,lift=.03)
        with self.assertRaisesRegex(ValueError,'preparation contract'):
            checkpoint.load_agent(self.path,scene_sha256='scene',task_sha256='task',
                                  obs_size=7,action_size=47,device='cpu',
                                  preparation_contract=CONTRACT,**options)

    def test_playback_rejects_changed_or_missing_full_contract(self):
        self.create(prepared=True)
        options = checkpoint.reward_options(self.path)
        changed = deepcopy(CONTRACT)
        changed['origin']['joint_positions']['wrist'] = .1
        for contract in (None,changed):
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError,'preparation contract'):
                checkpoint.load_agent(self.path,scene_sha256='scene',task_sha256='task',
                                      obs_size=7,action_size=47,device='cpu',
                                      preparation_contract=contract,**options)

    def test_warm_restart_rejects_changed_origin_and_missing_preparation(self):
        agent, metadata = self.create(prepared=True)
        for changed in ('preparation','preparation_contract'):
            expected = deepcopy(metadata)
            if changed == 'preparation':
                expected[changed] = None
            else:
                expected[changed]['origin']['joint_positions']['wrist'] = .1
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError,'preparation contract'):
                checkpoint.load(self.path,agent,expected)
        _, old = self.create()
        expected = dict(old,preparation=OPTIONS,preparation_contract=CONTRACT)
        old_agent = SAC(7,2,self.config,device='cpu')
        with self.assertRaisesRegex(ValueError,'preparation contract'):
            checkpoint.load(self.path,old_agent,expected)

    def test_learner_cli_requires_supported_structure_and_finite_values(self):
        required = ['--no-arms','--disable-wrist-tz','--disable-wrist-yaw']
        config = effective_config(parser().parse_args(['--preparation-time','1',*required]))
        self.assertEqual(config['preparation'], OPTIONS)
        for omitted in required:
            flags = [flag for flag in required if flag != omitted]
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                effective_config(parser().parse_args(['--preparation-time','1',*flags]))
        for flag in ('--preparation-time','--preparation-lift'):
            for value in ('-1','nan','inf'):
                with self.subTest(flag=flag,value=value), self.assertRaises(ValueError):
                    effective_config(parser().parse_args([f'{flag}={value}',*required]))
        self.assertIsNone(effective_config(parser().parse_args([]))['preparation'])

    def test_source_recipe_rejects_fractional_control_frame_and_stores_options(self):
        with patch('tsensei_v3_2_pianist.mjlab_env.source_runtime.source_files',return_value={}), \
             patch('tsensei_v3_2_pianist.mjlab_env.source_runtime.dependency_versions',return_value={}):
            recipe = make_recipe(preparation_time=1.,preparation_lift=.02,initial_buffer_time=0.)
            self.assertEqual(recipe['preparation'],OPTIONS)
            self.assertEqual(recipe['environment']['initial_buffer_time'],0.)
            self.assertIsNone(make_recipe()['preparation'])
            with self.assertRaisesRegex(ValueError,'control frames'):
                make_recipe(preparation_time=.025)


if __name__ == '__main__':
    unittest.main()
