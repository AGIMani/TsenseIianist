"""Reward-rate invariance and compatibility; CPU only."""
from copy import deepcopy
import unittest
import torch

from tests import test_paper_rewards as fixtures
from tests.test_mjlab_task import fixture
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager


class RewardTimeUnitsTest(unittest.TestCase):
    def contribution(self, dt, config):
        data = fixtures.PaperRewardsTest().inputs(config)
        _, action, snapshot, base, target, fingers, logic = data
        manager = RewardManager(config, dict(fixtures.scene_fixture(), control_dt=dt),
                               fixture().manifest, num_envs=2, device='cpu')
        result = manager(base, action, snapshot, target, fingers)
        return result, manager

    def test_same_constant_reward_per_second_and_legacy_units(self):
        cfg = fixtures.only('key_press_reward', 'energy_reward', 'forearm_reward')
        cfg.update(scale_by_dt=True, reference_dt=.05)
        old, old_manager = self.contribution(.05, cfg)
        new, new_manager = self.contribution(.025, cfg)
        for name in old:
            torch.testing.assert_close(old[name]*20, new[name]*40)
            torch.testing.assert_close(old_manager.raw_terms[name], new_manager.raw_terms[name])
            torch.testing.assert_close(new[name], new_manager.weighted_terms[name]*.5)
        legacy = deepcopy(cfg); legacy.pop('reference_dt')
        historical, _ = self.contribution(.025, legacy)
        unscaled = deepcopy(legacy); unscaled['scale_by_dt'] = False
        raw, _ = self.contribution(.025, unscaled)
        for name in raw:
            torch.testing.assert_close(historical[name], raw[name]*.025)
            torch.testing.assert_close(old[name], raw[name])

    def test_invalid_reference_units_rejected(self):
        for value in (0, -1, True, float('nan'), float('inf'), '0.05', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(dict(fixtures.only('key_press_reward'), scale_by_dt=True, reference_dt=value))
        with self.assertRaisesRegex(ValueError, 'requires scale_by_dt'):
            validate(dict(fixtures.only('key_press_reward'), reference_dt=.05))
