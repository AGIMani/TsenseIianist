"""Learned peaks cannot receive credit for initial/prefill/partial evaluations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('range_summary',
    Path(__file__).resolve().parents[1]/'scripts/summarize_range_trials.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RangeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root/'test-run'; self.run.mkdir()
        self.config = dict(num_envs=512, batch_size=1024, requested_vector_steps=512,
            requested_transitions=262144, learning_start_transition_1_based=91137,
            steady_updates_per_transition=.25, sac={'actor_lr':3e-5, 'critic_lr':3e-4, 'temp_lr':3e-4},
            origin={'checkpoint_sha256':'parent', 'fresh_optimizer_alpha_rng':False},
            control_limits=None, cli={})
        (self.run/'config.json').write_text(json.dumps(self.config))
        rows = []
        for step, updates, score, truncated in ((0,500000,.97,0),(128,500000,1.,0),
                (256,510000,.9,0),(384,520000,.999,1),(512,530000,.92,0)):
            rows.append(dict(vector_step=step, transitions=step*512, updates=updates,
                eval=dict(music_only_f1_min=score,music_only_micro_f1_min=score,
                          completed_episodes=3-truncated,truncated_episodes=truncated,evaluation_episodes=3)))
        self.rows = rows
        (self.run/'metrics.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
        (self.run/'best.json').write_text(json.dumps({'vector_step':0,'score':.97}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_running_warm_start_excludes_prefill_and_incomplete_peak(self):
        with (self.run/'metrics.jsonl').open('a') as stream:
            stream.write('{"vector_step":')
        result = module.summarize_run(self.run)
        self.assertEqual(result['status'], 'in_progress_or_interrupted')
        self.assertIsNone(result['elapsed_s'])
        self.assertIsNone(result['final_evaluation'])
        self.assertEqual(result['updates_this_run'],30000)
        self.assertEqual(result['evaluations'][1]['phase'],'pre_learning')
        self.assertEqual(result['best_after_learning']['vector_step'],512)
        self.assertAlmostEqual(result['best_learned_music_delta_from_initial'],-.05)
        self.assertFalse(result['learned_improved_over_initial'])
        self.assertTrue(result['saved_best_is_initial'])
        self.assertTrue(result['warnings'])

    def test_completed_csv_and_full_controller_preserved(self):
        controls = {'schema':1,'mode':'target_clip_slew_v1','channels':{'R_Wrist_TX':{'max_velocity':.4}}}
        self.config['control_limits'] = controls
        self.config['control_limits_contract'] = {'channels':{'R_Wrist_TX':{'index':20,'max_velocity':.4}}}
        (self.run/'config.json').write_text(json.dumps(self.config))
        (self.run/'summary.json').write_text(json.dumps(dict(status='completed',vector_steps=512,
            transitions=262144,updates_this_run=30000,elapsed_s=120.)))
        output = self.root/'report'
        module.main(['--runs',str(self.run),'--output',str(output)])
        result = json.loads((output/'summary.json').read_text())['runs'][0]
        self.assertEqual(result['configuration']['control_limits'], controls)
        self.assertEqual(result['final_evaluation']['vector_step'],512)
        self.assertEqual(result['elapsed_s'],120.)
        self.assertTrue((output/'runs.csv').exists())
        self.assertIn('control_clipped_fraction_max',(output/'evaluations.csv').read_text())

    def test_missing_config_has_explicit_state_and_source_output_is_rejected(self):
        missing = self.root/'not-yet-created'
        result = module.summarize([missing])
        self.assertEqual(result['runs'][0]['status'],'awaiting_config_or_invalid_config')
        self.assertFalse(result['comparison_checks']['num_envs']['equal'])
        module.flat_runs(result)
        with self.assertRaises(SystemExit):
            module.main(['--runs',str(self.run),'--output',str(self.run)])


if __name__ == '__main__':
    unittest.main()
