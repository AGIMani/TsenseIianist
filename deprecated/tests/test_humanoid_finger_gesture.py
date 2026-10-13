"""Idle-finger gesture reward uses physical joints and the scene reset pose."""
import math
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager
from tests.test_paper_rewards import scene_fixture


class GestureRewardTest(unittest.TestCase):
    def test_idle_fingers_only_and_joint_angle_similarity(self):
        scene = scene_fixture()
        names = scene['action_names'][:-1]
        scene['joint_names'] = names
        scene['qpos'] = [0.] * 50
        task = {'hand_joints': [names[:25], names[25:]]}
        config = template()
        for term in config['terms'].values():
            term['enabled'] = False
        config['terms']['humanoid_finger_gesture_reward']['enabled'] = True
        config['terms']['humanoid_finger_gesture_reward']['params'] = dict(maximum_per_finger=.2, sigma=.25)
        manager = RewardManager(config, scene, task, num_envs=2, device='cpu')
        snapshot = {'right_qpos': torch.zeros(2, 25), 'left_qpos': torch.zeros(2, 25)}
        fingers = torch.zeros(2, 10, 88)
        fingers[:, 1, 10] = 1  # Right index is assigned, so its pose is irrelevant.
        fingers[:, 6, 20] = 1  # Left index is assigned.
        snapshot['right_qpos'][:, 4] = 1.0
        snapshot['right_qpos'][1, 8] = .5  # Right middle MCP: one idle joint differs.
        result = manager({}, torch.zeros(2, 51), snapshot, torch.zeros(2, 89), fingers)
        expected = 1.4 + .2 * math.exp(-(.5 ** 2 / 4) / (2 * .25 ** 2))
        torch.testing.assert_close(result['humanoid_finger_gesture_reward'],
                                   torch.tensor([1.6, expected]), atol=1e-6, rtol=0)
        self.assertEqual(len(manager.gesture_pose), 40)

    def test_invalid_width_and_sigma_rejected(self):
        config = template()
        config['terms']['humanoid_finger_gesture_reward']['params'] = dict(maximum_per_finger=.2, sigma=0.)
        with self.assertRaises(ValueError):
            validate(config)
        scene = scene_fixture()
        names = scene['action_names'][:-1]
        scene.update(joint_names=names, qpos=[0.] * 50)
        config = template()
        config['terms']['humanoid_finger_gesture_reward']['enabled'] = True
        config['terms']['humanoid_finger_gesture_reward']['params'] = dict(maximum_per_finger=.2, sigma=.25)
        with self.assertRaisesRegex(ValueError, 'four reset joints'):
            RewardManager(config, scene, {'hand_joints': [names[:19] + names[20:25], names[25:]]},
                          num_envs=1, device='cpu')


if __name__ == '__main__':
    unittest.main()
