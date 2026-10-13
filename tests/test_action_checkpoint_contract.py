"""Evaluation and learner restarts reject changed targets with the same profile."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.action_mapping import resolved_contract
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig


ROOT = Path(__file__).resolve().parents[1]


class ActionCheckpointContractChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'checkpoint.pt'
        self.config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        self.agent = SAC(7, 47, self.config, device='cpu', seed=5,
                         action_profile='direct_joint_v1')
        rows = json.loads((ROOT/'configs/actions/direct_joint_v1.json').read_text())['channels']
        self.contract = resolved_contract('direct_joint_v1', 47,
            [row['name'] for row in rows], [row['physical_min'] for row in rows],
            [row['physical_max'] for row in rows])
        self.metadata = dict(scene_sha256='scene', task_sha256='task', stage=1,
                             action_profile='direct_joint_v1', action_contract=self.contract)
        checkpoint.save(self.path, self.agent, self.metadata, torch.Generator().manual_seed(13))

    def replay(self, contract):
        return checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
            obs_size=7, action_size=47, device='cpu', action_profile='direct_joint_v1',
            action_contract=contract)

    def test_matching_full_contract_reconstructs_identical_actions(self):
        restored = self.replay(deepcopy(self.contract))
        obs = torch.arange(7, dtype=torch.float32)[None]/10
        torch.testing.assert_close(restored.act(obs, deterministic=True),
                                   self.agent.act(obs, deterministic=True), rtol=0, atol=0)

    def test_replay_rejects_same_profile_with_changed_physical_or_policy_bounds(self):
        # Scene hashes alone cannot protect JSON bounds external to an explicit frozen scene.
        for field in ('physical_max', 'policy_max', 'name'):
            changed = deepcopy(self.contract)
            channel = next(row for row in changed['channels'] if row['name']=='R_Index_MCP')
            channel[field] = 'R_Mid_MCP' if field=='name' else channel[field]-.1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'action mapping range mismatch'):
                self.replay(changed)

    def test_replay_requires_the_reconstructed_contract(self):
        with self.assertRaisesRegex(ValueError, 'action mapping range mismatch'):
            self.replay(None)

    def test_all_resume_modes_reject_changed_physical_bounds_before_loading(self):
        changed = deepcopy(self.metadata)
        changed['action_contract']['channels'][4]['physical_max'] -= .1
        for kwargs in ({}, {'weights_only': True},
                       {'weights_only': True, 'transfer_mode': 'finetune', 'allow_control_change': True}):
            fresh = SAC(7, 47, self.config, device='cpu', seed=31,
                        action_profile='direct_joint_v1')
            before = deepcopy(fresh.actor.state_dict())
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'action mapping range mismatch'):
                checkpoint.load(self.path, fresh, changed, **kwargs)
            for name, value in fresh.actor.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__=='__main__':
    unittest.main()
