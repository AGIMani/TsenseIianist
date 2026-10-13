"""Daxian wrong-key travel costs, task timing and opt-in checkpoint contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, false_press_penalty
from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config
from tests.test_mjlab_task import fixture, snapshot
from tests.test_paper_rewards import scene_fixture
from tests.test_torch_sac import agent


def enabled_config():
    c = template()
    c['terms']['false_press_penalty']['enabled'] = True
    return c


class FalsePressPenaltyTests(unittest.TestCase):
    def test_travel_boundaries_clipping_and_gpu(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            travel = torch.tensor([-1., 0., .0009, .001, .001001, .5, 1., 1.5], device=device)
            keys = torch.zeros(8, 88, device=device)
            keys[:, 0] = travel
            target = torch.zeros(8, 89, device=device)
            expected = torch.tensor([0., 0., 0., 0., -.01014014, -.08, -.15, -.15], device=device)
            torch.testing.assert_close(false_press_penalty(keys, target), expected)
            target[:, 0] = 1
            torch.testing.assert_close(false_press_penalty(keys, target), torch.zeros(8, device=device))
            # Pedal state does not make a depressed non-target piano key valid.
            target[:, 0] = 0; target[:, 88] = 1
            torch.testing.assert_close(false_press_penalty(keys, target), expected)

    def test_additive_cost_current_goal_silence_and_reset(self):
        c = enabled_config()
        logic = MusicTaskLogic(fixture(), num_envs=2, device='cpu')
        manager = RewardManager(c, scene_fixture(), logic.manifest, num_envs=2, device='cpu')
        state = snapshot()
        state['key_state'][:, 70] = .5
        # This wrong key has not reached activation, but must already cost 0.08.
        self.assertFalse(state['key_activation'][:, 70].any())
        ids = torch.arange(2)
        logic.reset(ids, state)
        base, _ = logic.reward_terms(state, logic.index)
        result = logic.advance(torch.zeros(2, 51), state, reward_transform=manager)
        torch.testing.assert_close(result.info['reward_terms']['false_press_penalty'], torch.full((2,), -.08))
        torch.testing.assert_close(result.reward, sum(base.values()) - .08)
        torch.testing.assert_close(logic.last_reward, result.reward)
        torch.testing.assert_close(result.info['episode_return'], result.reward)
        self.assertFalse(result.done.any())
        # Next score frame is silent: the two formerly correct keys now cost .15 each.
        result = logic.advance(torch.zeros(2, 51), state, reward_transform=manager)
        torch.testing.assert_close(result.info['reward_terms']['false_press_penalty'], torch.full((2,), -.38))
        logic.reset(torch.tensor([0]), state); manager.reset(torch.tensor([0]))
        target = logic.goal[logic.index]
        torch.testing.assert_close(false_press_penalty(state['key_state'], target), torch.tensor([-.08, -.38]))

    def test_weight_and_global_dt_scaling(self):
        c = enabled_config()
        c['terms']['false_press_penalty']['weight'] = 2.
        c['scale_by_dt'] = True
        for name, term in c['terms'].items():
            term['enabled'] = name == 'false_press_penalty'
        manager = RewardManager(c, scene_fixture(), fixture().manifest, num_envs=2, device='cpu')
        state = snapshot(); state['key_state'].zero_(); state['key_state'][:, 70] = .5
        terms = manager({}, torch.zeros(2, 51), state, torch.zeros(2, 89), torch.zeros(2, 10, 88))
        torch.testing.assert_close(manager.raw_terms['false_press_penalty'], torch.full((2,), -.08))
        torch.testing.assert_close(manager.weighted_terms['false_press_penalty'], torch.full((2,), -.16))
        torch.testing.assert_close(terms['false_press_penalty'], torch.full((2,), -.008))

    def test_current_json_requires_the_complete_reward_contract(self):
        for stage in ('current', 'stage1', 'stage2_canon', 'stage2_other'):
            self.assertFalse(template(stage)['terms']['false_press_penalty']['enabled'])
        old = template(); old['terms'].pop('false_press_penalty')
        saved = deepcopy(old)
        with self.assertRaises(ValueError): validate(old)
        self.assertEqual(old, saved)
        bad = deepcopy(old); bad['terms'].pop('key_press_reward')
        with self.assertRaises(ValueError): validate(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.json'; path.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                effective_config(parser().parse_args(['--reward-config', str(path),
                                      '--enable-reward', 'false_press_penalty']))
        c = effective_config(parser().parse_args(['--disable-reward', 'false_press_penalty']))
        self.assertFalse(c['reward_config']['terms']['false_press_penalty']['enabled'])

    def test_checkpoint_restores_switch_and_rejects_reward_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            for enabled in (False, True):
                c = template(); c['terms']['false_press_penalty']['enabled'] = enabled
                saved = deepcopy(c)
                if not enabled: saved['terms'].pop('false_press_penalty')
                meta = dict(scene_sha256='s', task_sha256='t', stage=1, reward_config=saved)
                checkpoint.save(path, agent(), meta, torch.Generator().manual_seed(3))
                self.assertEqual(checkpoint.reward_options(path)['reward_config'], c)
                checkpoint.load(path, agent(), {**meta, 'reward_config': c})
                changed = deepcopy(c); changed['terms']['false_press_penalty']['enabled'] = not enabled
                with self.assertRaises(ValueError):
                    checkpoint.load(path, agent(), {**meta, 'reward_config': changed})


if __name__ == '__main__':
    unittest.main()
