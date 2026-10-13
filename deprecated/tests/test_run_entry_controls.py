"""Exercise the actual Bash entrypoint without building a scene or using CUDA."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser

ROOT = Path(__file__).resolve().parents[1]


def run_arguments(*extra):
    with tempfile.TemporaryDirectory() as folder:
        replacement = Path(folder) / 'python3'
        replacement.write_text('#!/bin/sh\nprintf "%s\\0" "$@"\n')
        replacement.chmod(0o755)
        output = subprocess.check_output(['bash', str(ROOT / 'run.sh'), *extra],
            cwd=ROOT, env={**os.environ, 'PATH': folder + os.pathsep + os.environ['PATH']})
    return output.decode().rstrip('\0').split('\0')[1:]


def run_config(*extra):
    args, _source_args = parser().parse_known_args(run_arguments(*extra))
    for name in ('finger_clearance_config', 'reward_config', 'control_limits'):
        if (value := getattr(args, name)) is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)
    return effective_config(args)


class RunEntryControlChecks(unittest.TestCase):
    def test_rate_defaults_and_group_joint_override_precedence(self):
        from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical
        import json
        original = json.loads((ROOT/'configs/control_limits/generic_wrist_rate_20260927.json').read_text())
        expected = canonical(original)
        for name in ('R_Wrist_TX', 'L_Wrist_TX', 'R_Wrist_TY', 'L_Wrist_TY'):
            expected['channels'][name]['max_velocity'] = 1.
        self.assertEqual(run_config()['control_limits'], expected)
        result = run_config('--joint-max-velocity', 'R_Wrist_TX=0.4',
                            '--wrist-tx-max-velocity', '0.6', '--wrist-ty-max-velocity', '0.8',
                            '--wrist-pitch-max-velocity', '2', '--finger-max-velocity', '8',
                            '--joint-max-velocity', 'R_Pinky_PIP=6',
                            '--joint-max-velocity', 'R_Wrist_TX=0.3')['control_limits']['channels']
        self.assertEqual(result['R_Wrist_TX']['max_velocity'], .3)
        self.assertEqual(result['L_Wrist_TX']['max_velocity'], .6)
        self.assertEqual(result['R_Wrist_TY']['max_velocity'], .8)
        self.assertEqual(result['L_Wrist_Pitch']['max_velocity'], 2)
        self.assertEqual(result['R_Pinky_PIP']['max_velocity'], 6)
        self.assertEqual(result['L_Pinky_PIP']['max_velocity'], 8)
        self.assertEqual(len(result), 46)

    def test_zero_rates_remove_rate_without_removing_position_bounds(self):
        self.assertIsNone(run_config('--wrist-tx-max-velocity', '0',
                                    '--wrist-ty-max-velocity', '0',
                                    '--wrist-pitch-max-velocity', '0')['control_limits'])
        from tsensei_v3_2_pianist.mjlab_env.control_limits import rate_overrides
        original = dict(schema=1, mode='target_clip_slew_v1', channels={
            'R_Wrist_TX': dict(min=-.1, max=.1, max_velocity=1.2)})
        result = rate_overrides(original, parser().parse_args(['--wrist-tx-max-velocity', '0']))
        self.assertEqual(result['channels'], {'R_Wrist_TX': dict(min=-.1, max=.1)})
        self.assertEqual(original['channels']['R_Wrist_TX']['max_velocity'], 1.2)

    def test_invalid_rates_and_removed_axes_fail_before_gpu_allocation(self):
        for argv in [('--wrist-tx-max-velocity', '-1'), ('--finger-max-velocity', 'nan'),
                     ('--wrist-pitch-max-velocity', 'inf'), ('--wrist-tz-max-velocity', '1'),
                     ('--wrist-yaw-max-velocity', '1'), ('--joint-max-velocity', 'ty=0'),
                     ('--joint-max-velocity', 'sustain=1'), ('--joint-max-velocity', 'R_Wrist_TX'),
                     ('--joint-max-velocity', 'R_Wrist_TX=bad')]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                run_config(*argv)

    def test_rate_cli_works_without_json_and_roundtrips_as_control_contract(self):
        import json
        from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical
        config = effective_config(parser().parse_args(['--wrist-tx-max-velocity', '.5']))
        expected = {'L_Wrist_TX': {'max_velocity': .5}, 'R_Wrist_TX': {'max_velocity': .5}}
        self.assertEqual(config['control_limits']['channels'], expected)
        self.assertEqual(canonical(json.loads(json.dumps(config['control_limits']))), config['control_limits'])

    def test_entrypoint_disables_both_controls_but_keeps_selected_rewards(self):
        config = run_config()
        self.assertFalse(config['finger_clearance']['idle_finger_protection'])
        self.assertFalse(config['finger_clearance']['active_press_permission'])
        terms = config['reward_config']['terms']
        self.assertEqual(terms['self_collision_reward'], dict(enabled=True, weight=.2, params={}))
        self.assertEqual(terms['false_press_penalty'], dict(enabled=True, weight=1., params={}))
        self.assertTrue(terms['humanoid_finger_gesture_reward']['enabled'])
        self.assertEqual(terms['humanoid_finger_gesture_reward']['params']['mode'], 'geometry_clearance_v1')
        self.assertEqual(config['evaluation_selection']['max_key_underside_substeps'], 0)
        self.assertEqual(config['preparation']['lift'], .025)
        self.assertFalse('--self-collision-weight' in run_arguments())

    def test_entrypoint_overrides_are_independent_and_last_value_wins(self):
        for idle, active in ((False, True), (True, False), (True, True)):
            with self.subTest(idle=idle, active=active):
                flags = ['--idle-finger-protection' if idle else '--no-idle-finger-protection',
                         '--active-press-permission' if active else '--no-active-press-permission']
                config = run_config(*flags)
                self.assertEqual(config['finger_clearance']['idle_finger_protection'], idle)
                self.assertEqual(config['finger_clearance']['active_press_permission'], active)
        self.assertFalse(run_config('--idle-finger-protection', '--no-idle-finger-protection')
                         ['finger_clearance']['idle_finger_protection'])

    def test_standalone_switch_does_not_enable_its_companion(self):
        self.assertIsNone(effective_config(parser().parse_args([]))['finger_clearance'])
        config = effective_config(parser().parse_args(['--active-press-permission']))
        self.assertFalse(config['finger_clearance']['idle_finger_protection'])
        self.assertTrue(config['finger_clearance']['active_press_permission'])
        self.assertEqual(config['finger_clearance']['active_command_mode'], 'active_contact_v1')

    def test_historical_geometry_file_retains_missing_control_fields(self):
        path = ROOT / 'configs/finger_clearance/generic_20260927.json'
        config = effective_config(parser().parse_args(['--finger-clearance-config', str(path)]))
        self.assertNotIn('idle_finger_protection', config['finger_clearance'])
        self.assertNotIn('active_press_permission', config['finger_clearance'])

    def test_geometry_reward_requires_computation_module_before_gpu_allocation(self):
        path = ROOT / 'configs/rewards/generic_finger_clearance_20260927.json'
        with self.assertRaisesRegex(ValueError, 'geometry settings'):
            effective_config(parser().parse_args(['--reward-config', str(path)]))
        config = run_config('--disable-reward', 'humanoid_finger_gesture_reward')
        self.assertFalse(config['reward_config']['terms']['humanoid_finger_gesture_reward']['enabled'])

    def test_plot_label_reads_actual_json_weight_and_legacy_fallback(self):
        spec = importlib.util.spec_from_file_location('plot_f1_options', ROOT / 'scripts/plot_training_f1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = run_config()
        self.assertEqual(config['self_collision_weight'], 0.)  # legacy shortcut only
        self.assertEqual(module.effective_self_collision_weight(config), .2)
        config['reward_config']['terms']['self_collision_reward']['enabled'] = False
        self.assertEqual(module.effective_self_collision_weight(config), 0.)
        self.assertEqual(module.effective_self_collision_weight({'self_collision_weight': .3}), .3)


if __name__ == '__main__':
    unittest.main()
