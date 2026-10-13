import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('timing_queue',ROOT/'scripts/hpc/queue_note_timing_tests.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class TimingQueueTests(unittest.TestCase):
    def test_comparison_has_only_isolated_factor_changes(self):
        plan=queue.make_plan(Path('/tmp/timing-plan'))
        self.assertEqual(plan['common']['coupling'],'range_60_80_deg')
        jobs=plan['jobs']
        self.assertEqual([j['mode'] for j in jobs],['validate','strike-diagnostic',*['train']*5])
        self.assertTrue(all(j['transitions']==3000320 for j in jobs[3:]))
        a,b,c,d=[j['arguments'][:-2] for j in jobs[3:]]
        self.assertEqual(a[:2],['--finger-coupling-angle-ranges','configs/finger_coupling/angle_ranges_60_80.json'])
        self.assertEqual([v for v,w in zip(a,b) if v!=w],['--no-exact-note-timing'])
        self.assertEqual([v for v,w in zip(b,c) if v!=w],['--no-staged-fingering-reward'])
        self.assertEqual([v for v,w in zip(b,d) if v!=w],['0.0'])

    def test_submit_prefix_keeps_dependency_and_arguments_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            (base/'sbatch').write_text('#!/bin/sh\nprintf "%s\\0" "$@"\n')
            (base/'sbatch').chmod(0o755)
            archive=base/'source bundle.tar.gz';archive.write_bytes(b'test')
            value='literal $(do-not-execute) value'
            result=subprocess.check_output(['bash',str(ROOT/'scripts/hpc/submit.sh'),
                '--dependency','afterok:123','--source-archive',str(archive),
                'train','--output-dir',value],env={**os.environ,'PATH':directory+os.pathsep+os.environ['PATH']})
            words=result.decode().rstrip('\0').split('\0')
            self.assertIn('--dependency=afterok:123',words)
            self.assertEqual(words[-3:],['train','--output-dir',value])
