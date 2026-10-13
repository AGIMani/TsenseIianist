"""Protect inferred timing from evaluation/save overhead and baseline credit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('quality_summary',
    Path(__file__).resolve().parents[1]/'scripts/summarize_quality_trials.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class QualitySummaryTests(unittest.TestCase):
    def test_timing_excludes_evaluation_save_and_no_updates(self):
        rows = []
        for step, elapsed, updates in ((170, 1, 0), (180, 10, 1024), (190, 11, 6144),
                                        (200, 22, 11264), (210, 23, 16384), (220, 24, 21504)):
            rows.append(dict(vector_step=step, transitions=step*512, updates=updates,
                             loop_transitions_per_second=step*512/elapsed))
        rows.append(dict(vector_step=195, eval={}))
        actual = summary.steady_intervals(rows, {'cli': {'save_interval': 200}})
        self.assertEqual([(r['from_vector_step'], r['to_vector_step']) for r in actual['intervals']],
                         [(180, 190), (210, 220)])
        self.assertEqual(actual['median_transitions_per_second'], 5120)
        self.assertEqual(actual['median_updates_per_second'], 5120)
        # Restored global Adam/update counters must not make the prefill look
        # like learning or admit the first graph capture into steady timing.
        for row in rows:
            if 'updates' in row:
                row['updates'] += 4_000_000
        restored = summary.steady_intervals(rows, {'cli': {'save_interval': 200}})
        self.assertEqual(restored['initial_updates'], 4_000_000)
        self.assertEqual(restored['intervals'], actual['intervals'])

    def test_initial_best_does_not_count_as_learning_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(batch_size=1024, steady_updates_per_transition=.25, warmup_transitions=0,
                          learning_start_transition_1_based=91137, reward_config={'terms': {}}, sac={}, cli={})
            (root/'config.json').write_text(json.dumps(config))
            rows = []
            for step, updates, score in ((0, 0, .97), (256, 9984, .88), (512, 42752, .93)):
                rows.append(dict(vector_step=step, transitions=step*512, updates=updates,
                    baseline=step == 0, eval=dict(music_only_f1_min=score,
                    completed_episodes=3, truncated_episodes=0, evaluation_episodes=3)))
            (root/'metrics.jsonl').write_text('\n'.join(map(json.dumps, rows))+'\n')
            result = summary.summarize_run(root, 'B')
            self.assertEqual(result['status'], 'in_progress_or_interrupted')
            self.assertEqual(result['best_after_learning']['vector_step'], 512)
            self.assertAlmostEqual(result['best_learned_music_delta_from_initial'], -.04)
            self.assertFalse(result['learned_improved_over_initial'])

    def test_only_an_unfinished_final_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metrics.jsonl'
            path.write_text('{"vector_step":1}\n{"vector_')
            rows, warnings = summary.read_rows(path)
            self.assertEqual(rows, [{'vector_step': 1}])
            self.assertEqual(len(warnings), 1)
            path.write_text('{"vector_step":1}\n{"vector_\n')
            with self.assertRaises(ValueError):
                summary.read_rows(path)


if __name__ == '__main__':
    unittest.main()
