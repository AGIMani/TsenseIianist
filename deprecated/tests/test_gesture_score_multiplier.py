"""Explicit gesture score gain preserves the strict external weight bound."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import torch

from tests import test_geometry_clearance_reward as fixtures
from tests.test_paper_rewards import scene_fixture
from tests.test_torch_sac import agent
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, geometry_clearance_penalty
from tsensei_v3_2_pianist.rl_torch import checkpoint

ROOT = Path(__file__).resolve().parents[1]
TERM = 'humanoid_finger_gesture_reward'


class GestureScoreMultiplierTest(unittest.TestCase):
    def test_actual_contributions_tenfold_after_chord_budget_and_dt(self):
        _, snapshot, target, fingers, baseline = fixtures.GeometryClearanceRewardTest().fixture(5)
        for row, count in enumerate((0, 1, 2, 10, 88)):
            target[row].zero_(); target[row, :count] = 1
        snapshot['finger_clearance_deficit_m'][:, 5] = .005
        snapshot['left_qpos'][:] = .25
        scene = scene_fixture(); names = scene['action_names'][:-1]
        scene.update(control_dt=.025, joint_names=names, qpos=[0.]*50)
        task = {'hand_joints': [names[:25], names[25:]]}
        baseline.update(scale_by_dt=True, reference_dt=.05)
        amplified = deepcopy(baseline); amplified['terms'][TERM]['params']['score_multiplier'] = 10.
        outputs = []
        for config in (baseline, amplified):
            manager = RewardManager(config, scene, task, num_envs=5, device='cpu')
            result = manager({'key_press_reward': torch.ones(5)}, torch.zeros(5, 51), snapshot, target, fingers)
            outputs.append((result, manager.raw_terms))
        torch.testing.assert_close(outputs[1][0][TERM], 10*outputs[0][0][TERM])
        torch.testing.assert_close(outputs[1][1][TERM], 10*outputs[0][1][TERM])
        torch.testing.assert_close(outputs[1][0][TERM], outputs[1][1][TERM]*.5)
        torch.testing.assert_close(outputs[1][0]['key_press_reward'], outputs[0][0]['key_press_reward'])
        self.assertTrue((outputs[1][0][TERM] < 0).all())

    def test_weight_above_one_and_invalid_gain_still_rejected(self):
        config = fixtures.GeometryClearanceRewardTest().fixture()[4]
        config['terms'][TERM]['params']['score_multiplier'] = 10.
        for weight in (1.000001, 10., True, float('inf')):
            bad = deepcopy(config); bad['terms'][TERM]['weight'] = weight
            with self.subTest(weight=weight), self.assertRaises(ValueError): validate(bad)
        for gain in (0, -1, True, float('nan'), float('inf'), '10', None):
            bad = deepcopy(config); bad['terms'][TERM]['params']['score_multiplier'] = gain
            with self.subTest(gain=gain), self.assertRaises(ValueError): validate(bad)
        validate(config)

    def test_legacy_contract_unchanged_and_only_one_config_parameter_differs(self):
        baseline = json.loads((ROOT/'configs/rewards/coupled_40hz_20260930.json').read_text())
        amplified = json.loads((ROOT/'configs/rewards/coupled_40hz_gesture10_20260930.json').read_text())
        self.assertEqual(validate(baseline), baseline)
        validate(amplified)
        self.assertEqual(amplified['terms'][TERM]['weight'], 1)
        self.assertEqual(amplified['terms'][TERM]['params'].pop('score_multiplier'), 10)
        self.assertEqual(amplified, baseline)

    def test_gain_scales_gradients_and_preserves_zero(self):
        params = deepcopy(fixtures.GEOMETRY_CLEARANCE_GESTURE)
        deficit = torch.full((1, 10), .005, requires_grad=True)
        zeros = torch.zeros_like(deficit); mask = torch.ones_like(deficit, dtype=torch.bool)
        target = torch.zeros(1, 89)
        old = geometry_clearance_penalty(deficit, zeros, mask, mask, target, params, 1.)
        old_grad, = torch.autograd.grad(old.sum(), deficit)
        params['score_multiplier'] = 10.
        new = geometry_clearance_penalty(deficit, zeros, mask, mask, target, params, 1.)
        new_grad, = torch.autograd.grad(new.sum(), deficit)
        torch.testing.assert_close(new, 10*old)
        torch.testing.assert_close(new_grad, 10*old_grad)
        torch.testing.assert_close(geometry_clearance_penalty(zeros, zeros, mask, mask, target, params, 1.), torch.zeros(1))

    def test_checkpoint_requires_the_saved_multiplier(self):
        config = fixtures.GeometryClearanceRewardTest().fixture()[4]
        config['terms'][TERM]['params']['score_multiplier'] = 10.
        metadata = dict(scene_sha256='s', task_sha256='t', stage=1, reward_config=config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            checkpoint.save(path, agent(), metadata, torch.Generator().manual_seed(7))
            self.assertEqual(checkpoint.reward_options(path)['reward_config'], config)
            checkpoint.load(path, agent(), metadata)
            changed = deepcopy(metadata)
            changed['reward_config']['terms'][TERM]['params'].pop('score_multiplier')
            with self.assertRaises(ValueError): checkpoint.load(path, agent(), changed)
