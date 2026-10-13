"""Software controls stay explicit in transfer, evaluation and saved best files."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical


OPTIONS = dict(schema=1, mode='target_clip_slew_v1', channels={'R_Wrist_TX': {'min': -.1, 'max': .1, 'max_velocity': .2}})
CONTRACT = dict(schema=1, mode='target_clip_slew_v1', control_dt=.05,
                channels={'R_Wrist_TX': dict(index=23, min=-.1, max=.1, max_velocity=.2,
                           position_unit='m', velocity_unit='m/s', original_min=-1., original_max=1., reset_target=0.)})


class ControlCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root/'best.pt'
        self.config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        self.agent = SAC(7, 3, self.config, device='cpu', seed=19)
        self.metadata = dict(scene_sha256='scene', task_sha256='task', stage=1)
    def tearDown(self):
        self.temporary.cleanup()
    def save(self, metadata=None):
        checkpoint.save(self.path, self.agent, metadata or self.metadata, torch.Generator().manual_seed(27))
    def fresh(self):
        return SAC(7, 3, replace(self.config, init_temperature=.0019), device='cpu', seed=31)

    def test_old_absent_contract_equals_none_and_best_replay_remains_valid(self):
        self.save()
        expected = dict(self.metadata, control_limits=None, control_limits_contract=None)
        checkpoint.load(self.path, self.agent, expected)
        options = checkpoint.reward_options(self.path)
        self.assertNotIn('control_limits', options)
        restored = checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
            obs_size=7, action_size=3, device='cpu', control_limits=None, control_limits_contract=None, **options)
        obs = torch.arange(7, dtype=torch.float32).reshape(1, 7)/10
        torch.testing.assert_close(restored.act(obs, deterministic=True), self.agent.act(obs, deterministic=True), rtol=0, atol=0)

    def test_new_best_replays_only_with_identical_resolved_controls(self):
        metadata = dict(self.metadata, control_limits=OPTIONS, control_limits_contract=CONTRACT,
                        evaluation_selection_state={'best': {'vector_step': 4000, 'score': .97}})
        self.save(metadata)
        options = checkpoint.reward_options(self.path)
        self.assertEqual(options['control_limits'], canonical(OPTIONS))
        restored = checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
            obs_size=7, action_size=3, device='cpu', control_limits_contract=deepcopy(CONTRACT), **options)
        obs = torch.ones(1, 7)
        torch.testing.assert_close(restored.act(obs, deterministic=True), self.agent.act(obs, deterministic=True), rtol=0, atol=0)
        changed = deepcopy(CONTRACT);changed['channels']['R_Wrist_TX']['max_velocity'] = .3
        for resolved in (None, changed):
            with self.subTest(resolved=resolved), self.assertRaisesRegex(ValueError, 'control|Control'):
                checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
                    obs_size=7, action_size=3, device='cpu', control_limits_contract=resolved, **options)
        with self.assertRaisesRegex(ValueError, 'control limits contract'):
            checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
                                  obs_size=7, action_size=3, device='cpu')

    def test_control_change_requires_explicit_finetune_and_keeps_other_guards(self):
        self.save()
        next_metadata = dict(self.metadata, control_limits=OPTIONS, control_limits_contract=CONTRACT)
        for kwargs in ({}, {'weights_only':True}, {'weights_only':True, 'transfer_mode':'finetune'},
                       {'allow_control_change':True}, {'weights_only':True, 'allow_control_change':True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'Control|control'):
                checkpoint.load(self.path, self.fresh(), next_metadata, **kwargs)
        fresh = self.fresh();rng = fresh.update_rng.get_state().clone()
        checkpoint.load(self.path, fresh, next_metadata, weights_only=True,
                        transfer_mode='finetune', allow_control_change=True)
        self.assertEqual(fresh.updates, 0)
        self.assertFalse(fresh.actor_optimizer.state)
        self.assertAlmostEqual(fresh.temperature.item(), .0019, places=8)
        self.assertTrue(torch.equal(fresh.update_rng.get_state(), rng))
        for changed in (dict(next_metadata, scene_sha256='wrong'), dict(next_metadata, preparation={'duration':1}),
                        dict(next_metadata, self_collision_weight=.2)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                checkpoint.load(self.path, self.fresh(), changed, weights_only=True,
                                transfer_mode='finetune', allow_control_change=True)

    def test_cli_records_full_json_and_rejects_implicit_change(self):
        path = self.root/'controls.json';path.write_text(json.dumps(OPTIONS))
        args = ['--control-limits', str(path), '--finetune-from', str(self.path), '--allow-control-change', '--actor-lr', '.00003']
        config = effective_config(parser().parse_args(args))
        self.assertEqual(config['control_limits'], canonical(OPTIONS))
        self.assertEqual(config['sac']['actor_lr'], 3e-5)
        self.assertIsNone(effective_config(parser().parse_args([]))['control_limits'])
        for flags in (['--allow-control-change'], ['--warm-start', str(self.path), '--allow-control-change'],
                      ['--stage', '2', '--stage2-from', str(self.path), '--allow-control-change']):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, 'finetune-from'):
                effective_config(parser().parse_args(flags))


if __name__ == '__main__':
    unittest.main()
