"""Training comparison contracts and short events missed by endpoint sampling."""
from copy import deepcopy
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env import performance_options
from tsensei_v3_2_pianist.mjlab_env.fingering_experiments import fingering_reward, staged_scores
from tsensei_v3_2_pianist.mjlab_env.reward_config import template, validate, legacy_weight, GEOMETRY_CLEARANCE_GESTURE
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, false_press_penalty
from tests.test_paper_rewards import scene_fixture


def snapshot(count=1):
    return dict(fingertips=torch.zeros(count, 10, 3), key_targets=torch.zeros(count, 88, 3),
                key_surface_height=torch.zeros(count, 10, 88), key_state=torch.zeros(count, 88),
                key_activation=torch.zeros(count, 88, dtype=torch.bool))


def only(*names, **performance):
    config = template()
    for name, term in config['terms'].items():
        term['enabled'] = name in names
    config['performance'] = performance_options.canonical(performance)
    return config


class PerformanceOptionsTest(unittest.TestCase):
    def test_historical_config_unchanged_and_three_switches_independent(self):
        config = template()
        self.assertEqual(validate(config), config)
        self.assertNotIn('performance', validate(config))
        self.assertEqual(legacy_weight(config), 0.)
        for name, value in (('anticipatory_fingering_time', .15),
                            ('staged_fingering_reward', True), ('exact_note_timing', True)):
            candidate = deepcopy(config);candidate['performance'] = {name: value}
            result = performance_options.get(validate(candidate))
            self.assertEqual(result[name], value)
            self.assertEqual(sum(v != performance_options.DEFAULTS[k] for k, v in result.items()), 1)
            self.assertIsNone(legacy_weight(validate(candidate)))

    def test_invalid_options_and_paper_fingering_are_rejected(self):
        for options in ({'bad': True}, {'substep_rewards': 1}, {'staged_fingering_reward': 'true'},
                        {'anticipatory_fingering_time': True}, {'anticipatory_fingering_time': -.1},
                        {'anticipatory_fingering_time': .51}, {'anticipatory_fingering_time': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                performance_options.canonical(options)
        config = template('stage1');config['performance'] = {'staged_fingering_reward': True}
        with self.assertRaisesRegex(ValueError, 'enabled legacy fingering_reward'):validate(config)
        config = only('key_press_reward', anticipatory_fingering_time=.15)
        with self.assertRaisesRegex(ValueError, 'enabled legacy fingering_reward'):validate(config)


class FingeringExperimentsTest(unittest.TestCase):
    def test_hover_gets_at_most_approach_credit_and_press_requires_travel(self):
        s = snapshot(3);s['key_state'][1, 20] = .5;s['key_state'][2, 20] = 1.
        fingers = torch.zeros(3, 10, 88);fingers[:, 0, 20] = 1.
        torch.testing.assert_close(fingering_reward(s, fingers, staged=True), torch.tensor([.3, .65, 1.]))
        torch.testing.assert_close(fingering_reward(s, fingers), torch.ones(3))

    def test_actual_top_plane_blocks_below_key_press_credit(self):
        s = snapshot(4);s['key_state'].fill_(1.)
        s['fingertips'][1, :, 2] = -.005;s['fingertips'][2, :, 2] = .007
        # Same key target but physically shifted top plane: use the actual plane.
        s['key_surface_height'][3] = .006
        scores = staged_scores(s)
        self.assertAlmostEqual(scores[0, 0, 20].item(), 1., places=6)
        self.assertTrue((scores[1:, 0, 20] <= .300001).all())
        del s['key_surface_height']
        with self.assertRaisesRegex(ValueError, 'key_surface_height'):staged_scores(s)

    def test_far_other_finger_cannot_take_credit_for_pressed_key(self):
        s = snapshot();s['key_state'][0, 20] = 1.;s['fingertips'][0, 0, 0] = .3
        fingers = torch.zeros(1, 10, 88);fingers[:, 0, 20] = 1.
        self.assertLess(fingering_reward(s, fingers, staged=True).item(), .001)

    def test_future_only_rest_has_signal_and_bounded_budget(self):
        s = snapshot(2);s['anticipatory_targets'] = torch.zeros(2, 10, 3)
        s['anticipatory_mask'] = torch.zeros(2, 10, dtype=torch.bool);s['anticipatory_mask'][:, 0] = True
        s['fingertips'][1, 0, 0] = .1
        fingers = torch.zeros(2, 10, 88)
        value = fingering_reward(s, fingers, anticipatory=True)
        self.assertAlmostEqual(value[0].item(), .5)
        self.assertLess(value[1].item(), value[0].item())
        fingers[:, 1, 20] = 1.
        self.assertAlmostEqual(fingering_reward(s, fingers, anticipatory=True)[0].item(), .75)
        self.assertTrue((fingering_reward(s, fingers, anticipatory=True) <= 1.).all())

    def test_current_finger_not_double_counted_as_future(self):
        s = snapshot();s['anticipatory_targets'] = torch.ones(1, 10, 3)
        s['anticipatory_mask'] = torch.zeros(1, 10, dtype=torch.bool);s['anticipatory_mask'][:, 0] = True
        fingers = torch.zeros(1, 10, 88);fingers[:, 0, 20] = 1.
        self.assertEqual(fingering_reward(s, fingers, anticipatory=True).item(), 1.)

    def test_opt_in_anticipation_exempts_posture_but_never_geometry(self):
        scene = scene_fixture();names = scene['action_names'][:-1]
        scene.update(joint_names=names, qpos=[0.]*50)
        config = only('key_press_reward', 'fingering_reward', 'humanoid_finger_gesture_reward',
                      anticipatory_fingering_time=.15)
        config['terms']['humanoid_finger_gesture_reward']['params'] = deepcopy(GEOMETRY_CLEARANCE_GESTURE)
        task = dict(hand_joints=[names[:25], names[25:]], finger_tolerance=.01)
        manager = RewardManager(config, scene, task, num_envs=2, device='cpu')
        s = snapshot(2)
        s.update(right_qpos=torch.zeros(2, 25), left_qpos=torch.zeros(2, 25),
                 finger_clearance_deficit_m=torch.zeros(2, 10),
                 finger_pose_reference_feasible=torch.ones(2, 10, dtype=torch.bool),
                 anticipatory_targets=torch.zeros(2, 10, 3),
                 anticipatory_mask=torch.zeros(2, 10, dtype=torch.bool))
        s['right_qpos'][:, manager.gesture_indices[0][0]] = .25
        s['anticipatory_mask'][0, 0] = True
        fingers = torch.zeros(2, 10, 88);target = torch.zeros(2, 89)
        action = torch.zeros(2, 51);base = {'key_press_reward': torch.full((2,), .5)}
        result = manager(base, action, s, target, fingers)['humanoid_finger_gesture_reward']
        torch.testing.assert_close(result, torch.tensor([0., -.001]))
        s['finger_clearance_deficit_m'][:, 0] = .005
        result = manager(base, action, s, target, fingers)['humanoid_finger_gesture_reward']
        torch.testing.assert_close(result, torch.tensor([-.04, -.041]))


if __name__ == '__main__':unittest.main()
