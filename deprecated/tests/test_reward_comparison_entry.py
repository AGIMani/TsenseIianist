"""One-factor CLI, full configuration persistence, and honest score summaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

import torch

from tsensei_v3_2_pianist.mjlab_env import reward_config, performance_options
from tsensei_v3_2_pianist.rl_torch.train import SubstepKeyMetrics, effective_config, parser

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reward_comparison_runner', ROOT/'scripts/run_reward_comparison.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PerformanceCLI(unittest.TestCase):
    def test_absence_keeps_old_contract_and_switches_are_independent(self):
        self.assertNotIn('performance', effective_config(parser().parse_args([]))['reward_config'])
        for argv, expected in (
            (['--anticipatory-fingering-time', '.15'], dict(anticipatory_fingering_time=.15)),
            (['--staged-fingering-reward'], dict(staged_fingering_reward=True)),
        ):
            with self.subTest(argv=argv):
                config = effective_config(parser().parse_args(argv))['reward_config']
                defaults = performance_options.DEFAULTS
                self.assertEqual(config['performance'], defaults | expected)

    def test_cli_overrides_json_and_later_negative_switch_wins(self):
        config = reward_config.template()
        config['performance'] = dict(anticipatory_fingering_time=.2,
                                     staged_fingering_reward=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rewards.json'
            path.write_text(json.dumps(config))
            args = parser().parse_args(['--reward-config', str(path), '--no-staged-fingering-reward',
                                       '--staged-fingering-reward', '--no-staged-fingering-reward'])
            self.assertEqual(effective_config(args)['reward_config']['performance'],
                             performance_options.DEFAULTS | dict(anticipatory_fingering_time=.2, staged_fingering_reward=False))

    def test_invalid_horizon_rejected(self):
        for value in ('nan', 'inf', '-.01', '.51'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                effective_config(parser().parse_args(['--anticipatory-fingering-time', value]))


class SubstepEvaluation(unittest.TestCase):
    def info(self, index, sampled=10, wrong=0, tp=0, fn=0):
        return dict(score_index=torch.tensor([index]), substep_count=torch.tensor([sampled]),
                    wrong_key_substeps=torch.tensor([wrong]), wrong_key_key_substeps=torch.tensor([wrong]),
                    substep_key_counts=torch.tensor([[tp, wrong, fn]]))

    def test_transient_counts_and_preparation_are_separate(self):
        metric = SubstepKeyMetrics('cpu', preparation_steps=1, music_start_steps=2, decimation=10)
        metric.sample(self.info(0, wrong=2))
        metric.sample(self.info(1))
        metric.sample(self.info(2, wrong=1, tp=8, fn=2))
        result = metric.result(3)
        self.assertEqual(result['wrong_key_substeps'], 3)
        self.assertEqual(result['preparation_wrong_key_substeps'], 2)
        self.assertEqual(result['music_only_wrong_key_substeps'], 1)
        self.assertAlmostEqual(result['music_only_substep_key_micro_f1'], 16/19)
        self.assertEqual(result['substep_key_coverage'], 1)
        self.assertTrue(result['substep_key_diagnostics_valid'])

    def test_missing_and_partial_sampling_never_report_zero_error(self):
        for missing in (True, False):
            metric = SubstepKeyMetrics('cpu', 0, 0, 10)
            metric.sample(self.info(0))
            metric.sample({} if missing else self.info(1, sampled=9))
            result = metric.result(2)
            self.assertIsNone(result['wrong_key_substeps'])
            self.assertIsNone(result['substep_key_micro_f1'])
            self.assertFalse(result['substep_key_diagnostics_valid'])
            self.assertLess(result['substep_key_coverage'], 1)


class ComparisonRunner(unittest.TestCase):
    def test_source_archive_is_replayable_and_configuration_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tsensei_v3_2_pianist').mkdir()
            (root/'tsensei_v3_2_pianist/model.json').write_text('{"model": "fixture"}')
            (root/'configs').mkdir()
            (root/'configs/reward.json').write_text('{}')
            (root/'run.sh').write_text('# fixture\n')
            (root/'README.md').write_text('documentation')
            (root/'outputs').mkdir()
            (root/'outputs/old.pt').write_bytes(b'not source')
            output = root/'archive'
            output.mkdir()
            archive = RUNNER.create_source_archive(output, root)
            with tarfile.open(archive['path']) as bundle:
                self.assertEqual(set(bundle.getnames()),
                    {'tsensei_v3_2_pianist/model.json', 'configs/reward.json', 'run.sh', 'README.md'})
            manifest = dict(source_archive=archive)
            RUNNER.verify_sources(manifest, root)
            (root/'README.md').write_text('Result documentation can update during training')
            RUNNER.verify_sources(manifest, root)
            (root/'configs/reward.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(RuntimeError, 'configs/reward.json'):
                RUNNER.verify_sources(manifest, root)

    def test_summary_distinguishes_selected_best_from_unsafe_peak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = RUNNER.parser().parse_args(['--arms', 'baseline'])
            plan = RUNNER.make_plan(args, root)
            output = root/'baseline'/'run-20260928-test'
            output.mkdir(parents=True)
            evaluation = dict(music_only_f1_min=.99, completed_episodes=3,
                              evaluation_episodes=3, truncated_episodes=0)
            (output/'metrics.jsonl').write_text(json.dumps(dict(vector_step=100, eval=evaluation,
                checkpoint_selection=dict(best_eligible=False)))+'\n{')
            (output/'summary.json').write_text(json.dumps(dict(transitions=1_000_448, elapsed_s=100,
                stop_reason='requested_budget_exhausted', evaluation_selection_state=dict(best=None),
                final_evaluation=evaluation)))
            (output/'config.json').write_text('{}')
            row = RUNNER.summarize(plan)['runs'][0]
            self.assertIsNone(row['best'])
            self.assertEqual(row['max_observed']['score'], .99)
            self.assertFalse(row['max_observed']['best_eligible'])
            self.assertEqual(plan['runs'][0]['config_path'], str(output/'config.json'))
            self.assertTrue((root/'comparison.md').is_file())


if __name__ == '__main__':
    unittest.main()
