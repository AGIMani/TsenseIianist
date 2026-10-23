"""Launch contract and serialization checks without initializing CUDA."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from tsensei_v3_2_pianist.mjlab_env import performance_options, source_runtime
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

ROOT=Path(__file__).resolve().parents[1]


class ExactTimingEntryTests(unittest.TestCase):
    def setUp(self):
        from tsensei_v3_2_pianist.mjlab_env.reward_config import template
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reward_path = Path(self.directory.name)/'reward.json'
        reward = template()
        reward['terms']['humanoid_finger_gesture_reward'].update(
            enabled=True, params=dict(coefficient=-.1, reference='scene_reset', lookahead_steps=20))
        reward['performance'] = dict(exact_note_timing=True, anticipatory_fingering_time=.15)
        self.reward_path.write_text(json.dumps(reward))

    def config(self,*args):
        parsed=parser().parse_args(['--reward-config', str(self.reward_path),
            '--finger-clearance-config', str(ROOT/'configs/finger_clearance/control_disabled_20260927.json'),
            '--finger-coupling', *args])
        return effective_config(parsed)

    def test_default_and_single_factor_overrides(self):
        common=('--finger-coupling-angle-ranges',str(ROOT/'configs/finger_coupling/angle_ranges_60_80.json'),'--reward-config',str(self.reward_path))
        base=self.config(*common,'--exact-note-timing')
        self.assertTrue(base['reward_config']['performance']['exact_note_timing'])
        self.assertTrue(all(v == [60,80] for v in base['finger_coupling']['angle_ranges_deg'].values()))
        for extra,key,value in ((( '--upcoming-press-permission-time','.05'),'upcoming_press_permission_time',.05),):
            expected=deepcopy(base)
            expected['reward_config']['performance'][key]=value
            self.assertEqual(self.config(*common,'--exact-note-timing',*extra),expected)
        with self.assertRaisesRegex(ValueError, 'Current rewards require'):
            self.config(*common,'--no-exact-note-timing')

    def test_old_sampling_dictionary_requires_archived_source(self):
        with self.assertRaisesRegex(ValueError, 'matching archived source'):
            performance_options.canonical(dict(substep_rewards=True))

    def test_permission_rejects_incomplete_or_unsafe_contracts(self):
        for extra in (('--no-exact-note-timing',),('--anticipatory-fingering-time','.025'),
                      ('--idle-finger-protection',),('--active-press-permission',),
                      ('--disable-reward','humanoid_finger_gesture_reward')):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                self.config('--reward-config',str(self.reward_path),
                            '--upcoming-press-permission-time','.05',*extra)
        for value in (-.01,.151,float('inf'),float('nan'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                performance_options.canonical(dict(exact_note_timing=True,
                    anticipatory_fingering_time=.15,upcoming_press_permission_time=value))

    def test_recipe_defaults_to_exact_and_matches_explicit_request(self):
        with patch.object(source_runtime,'source_files',return_value={}),patch.object(source_runtime,'dependency_versions',return_value={}):
            old=source_runtime.make_recipe()
            exact=source_runtime.make_recipe(exact_note_timing=True)
        self.assertTrue(old['environment']['exact_note_timing'])
        self.assertTrue(exact['environment']['exact_note_timing'])
        self.assertEqual(old,exact)

    def test_source_entry_honors_json_and_explicit_cli_override(self):
        spec=importlib.util.spec_from_file_location('source_timing_entry',ROOT/'scripts/train_from_source.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reward.json'
            path.write_text(json.dumps({'performance':{'exact_note_timing':True}}))
            self.assertTrue(module.resolve_exact_note_timing(None,['--reward-config',str(path)]))
            self.assertFalse(module.resolve_exact_note_timing(False,['--reward-config='+str(path)]))
            self.assertTrue(module.resolve_exact_note_timing(None,[]))
