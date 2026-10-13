"""CPU-only scheduler stubs for two-GPU allocation and single-GPU replay."""
import json
import hashlib
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AllocationEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='allocation entry ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root/'scripts/hpc/submit.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/'scripts/hpc/submit.sh', self.script)
        self.bin = self.root/'bin'
        self.bin.mkdir()
        for command in ('sbatch', 'srun'):
            shim = self.bin/command
            shim.write_text(f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'print(json.dumps(dict(command=Path(sys.argv[0]).name, argv=sys.argv[1:], '
                'allocation_step=os.environ.get("TSENSEI_ALLOCATION_STEP"), '
                'expected_gpus=os.environ.get("TSENSEI_EXPECTED_GPUS"))))\n')
            shim.chmod(0o755)
        queue = self.bin/'squeue'
        queue.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_ALLOCATION_STATE"\n')
        queue.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin)+os.pathsep+os.environ['PATH'],
            TEST_ALLOCATION_STATE=pwd.getpwuid(os.getuid()).pw_name+' RUNNING TsenseI-reserve2')

    def run_entry(self, *args, success=True):
        result = subprocess.run(['bash', str(self.script), *args], env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        return result.stderr

    def test_reserve_has_explicit_48_hours_and_no_training_command(self):
        result = self.run_entry('reserve')
        self.assertEqual(result['command'], 'sbatch')
        self.assertIn('--gres=gpu:2', result['argv'])
        self.assertEqual(result['expected_gpus'], '2')
        self.assertIn('--time=2-00:00:00', result['argv'])
        self.assertIn('--job-name=TsenseI-reserve2', result['argv'])
        self.assertEqual(result['argv'][-3:], [str(self.script.parent/'reserve.sbatch'), str(self.root), 'reserve'])
        self.run_entry('reserve', '--vector-steps', '1', success=False)

    def test_running_allocation_uses_srun_preserving_arguments_and_step_logs(self):
        result = self.run_entry('--allocation', '1364', 'train', '--output-dir', '/path with spaces/run')
        self.assertEqual(result['command'], 'srun')
        self.assertEqual(result['allocation_step'], '1')
        self.assertEqual(result['expected_gpus'], '2')
        for value in ('--jobid=1364', '--overlap', '--exact', '--gres=gpu:2', '--cpus-per-task=16'):
            self.assertIn(value, result['argv'])
        self.assertIn('--output='+str(self.root/'logs/hpc/1364/steps/%s.out'), result['argv'])
        self.assertEqual(result['argv'][-6:], ['bash', str(self.script.parent/'job.sbatch'),
                                             str(self.root), 'train', '--output-dir', '/path with spaces/run'])

    def test_pending_foreign_or_non_keeper_allocation_rejected(self):
        username = pwd.getpwuid(os.getuid()).pw_name
        for state in (username+' PENDING TsenseI-reserve2', 'other RUNNING TsenseI-reserve2',
                      username+' RUNNING TsenseI-A32', ''):
            self.env['TEST_ALLOCATION_STATE'] = state
            with self.subTest(state=state):
                self.assertIn('RUNNING', self.run_entry('--allocation', '1364', 'train', success=False))

    def test_invalid_allocation_combinations_rejected(self):
        for args in (('--allocation', 'bad', 'train'), ('--allocation', '1364', 'reserve'),
                     ('--allocation', '1364', '--dependency', 'afterok:12', 'train')):
            with self.subTest(args=args):
                self.run_entry(*args, success=False)

    def test_normal_submission_retains_dependency_and_source_archive(self):
        archive = self.root/'source file.tar.gz'
        archive.write_bytes(b'archive fixture')
        result = self.run_entry('--dependency', 'afterok:12', '--source-archive', str(archive),
                                'validate', 'tests/test_finger_coupling.py')
        self.assertEqual(result['command'], 'sbatch')
        self.assertIn('--dependency=afterok:12', result['argv'])
        self.assertNotIn('--time=2-00:00:00', result['argv'])
        self.assertEqual(result['argv'][-4:], [str(self.script.parent/'job.sbatch'), str(self.root),
                                             'validate', 'tests/test_finger_coupling.py'])

    def test_default_and_explicit_two_gpu_submission_override_inherited_count(self):
        self.env['TSENSEI_EXPECTED_GPUS'] = '1'
        for args in (('train',), ('replay',), ('--gpus', '2', 'train')):
            with self.subTest(args=args):
                result = self.run_entry(*args)
                self.assertIn('--gres=gpu:2', result['argv'])
                self.assertEqual(result['expected_gpus'], '2')

    def test_single_gpu_replay_preserves_archive_dependency_and_arguments(self):
        archive = self.root/'source file.tar.gz'
        archive.write_bytes(b'archive fixture')
        result = self.run_entry('--gpus', '1', '--source-archive', str(archive),
                                '--dependency', 'afterok:12', 'replay',
                                '--checkpoint', '/path with spaces/best.pt')
        self.assertEqual(result['command'], 'sbatch')
        self.assertEqual(result['expected_gpus'], '1')
        self.assertIn('--gres=gpu:1', result['argv'])
        self.assertNotIn('--gres=gpu:2', result['argv'])
        self.assertIn('--dependency=afterok:12', result['argv'])
        self.assertEqual(result['argv'][-5:], [str(self.script.parent/'job.sbatch'),
            str(self.root), 'replay', '--checkpoint', '/path with spaces/best.pt'])

    def test_single_gpu_training_validation_and_reward_diagnostic(self):
        for mode in ('train', 'validate', 'reward-diagnostic'):
            with self.subTest(mode=mode):
                result = self.run_entry('--gpus', '1', mode)
                self.assertIn('--gres=gpu:1', result['argv'])
                self.assertEqual(result['expected_gpus'], '1')
                self.assertEqual(result['argv'][-1], mode)

    def test_single_gpu_export_preserves_plan(self):
        result = self.run_entry('--gpus', '1', 'export', '--plan', '/path with spaces/plan.json')
        self.assertEqual(result['expected_gpus'], '1')
        self.assertIn('--gres=gpu:1', result['argv'])
        self.assertEqual(result['argv'][-3:], ['export', '--plan', '/path with spaces/plan.json'])

    def test_single_gpu_rejected_for_unsupported_modes_and_reused_allocation(self):
        for mode in ('reserve', 'probe', 'benchmark', 'performance',
                     'preview', 'arm-replay', 'online-export', 'strike-diagnostic'):
            with self.subTest(mode=mode):
                self.assertIn('supports only', self.run_entry('--gpus', '1', mode, success=False))
        self.assertIn('cannot reuse --allocation', self.run_entry(
            '--gpus', '1', '--allocation', '1364', 'replay', success=False))

    def test_invalid_or_missing_gpu_count_rejected_before_scheduler(self):
        for args in (('--gpus',), ('--gpus', '0'), ('--gpus', '3'), ('--gpus', '-1'),
                     ('--gpus', '1.0'), ('--gpus', 'gpu:1'), ('--gpus', 'replay')):
            with self.subTest(args=args):
                self.assertIn('requires 1 or 2', self.run_entry(*args, success=False))


class JobProbeEntryTests(unittest.TestCase):
    def test_expected_gpu_count_reaches_probe_before_any_workload(self):
        # The fake interpreter stops at the probe. No GPU package is imported,
        # no archive is unpacked, and no Slurm command is executed by this test.
        with tempfile.TemporaryDirectory(prefix='job probe entry ') as directory:
            root = Path(directory)
            python = root/'training_mjlab/.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.write_text(f'#!{sys.executable}\nimport json, sys\n'
                              'print(json.dumps(sys.argv[1:]))\nsys.exit(73)\n')
            python.chmod(0o755)
            env = {name: value for name, value in os.environ.items()
                   if not name.startswith(('TSENSEI_', 'SLURM_'))}
            env['SLURM_JOB_ID'] = '123'
            for expected, mode, valid in ((None, 'replay', True), ('1', 'replay', True),
                                          ('2', 'train', True), ('1', 'train', True),
                                          ('1', 'validate', True), ('1', 'reward-diagnostic', True),
                                          ('1', 'export', True),
                                          ('1', 'benchmark', False),
                                          ('0', 'replay', False), ('3', 'replay', False)):
                with self.subTest(expected=expected, mode=mode):
                    job_env = dict(env)
                    if expected is not None:
                        job_env['TSENSEI_EXPECTED_GPUS'] = expected
                    result = subprocess.run(['bash', str(ROOT/'scripts/hpc/job.sbatch'),
                        str(root), mode], env=job_env, text=True, capture_output=True, timeout=10)
                    if valid:
                        self.assertEqual(result.returncode, 73, result.stderr)
                        self.assertEqual(json.loads(result.stdout), [str(root/'scripts/hpc/probe_gpus.py'),
                            '--expected-gpus', expected or '2', '--output',
                            str(root/'logs/hpc/123/gpu_probe.json')])
                    else:
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual(result.stdout, '')


class ArchivedTrainingEntryTests(unittest.TestCase):
    def run_archive(self, *, new_entry, new_exit=0):
        """Execute the real job shell with fake probe/telemetry and a real tar."""
        with tempfile.TemporaryDirectory(prefix='archived_training_entry_') as directory:
            root = Path(directory)
            bin_dir = root/'training_mjlab/.venv/bin'
            bin_dir.mkdir(parents=True)
            (root/'training/.venv').mkdir(parents=True)
            interpreter = (f'#!{sys.executable}\nimport sys\n'
                'if sys.argv[1] == "-c": print("GPU-fixture")\n'
                'elif sys.argv[1].endswith("probe_gpus.py"): pass\n'
                'elif sys.argv[1].endswith("package_a32.py"): print("{\\"audit\\":\\"fixture\\"}")\n'
                'else: raise SystemExit("Unexpected fake interpreter call")\n')
            for name in ('python', 'python3'):
                script = bin_dir/name
                script.write_text(interpreter)
                script.chmod(0o755)
            for name, content in (
                    ('nvcc', '#!/bin/sh\nexit 0\n'),
                    ('nvidia-smi', f'#!{sys.executable}\nimport time\ntime.sleep(60)\n')):
                script = bin_dir/name
                script.write_text(content)
                script.chmod(0o755)
            source = root/'archive_fixture/scripts/hpc'
            source.mkdir(parents=True)
            def entry(label, exit_code):
                return ('#!/bin/sh\nprintf "'+label+'\\0" > "$TEST_ENTRY_RESULT"\n'
                        'printf "%s\\0" "$@" >> "$TEST_ENTRY_RESULT"\n'
                        'exit '+str(exit_code)+'\n')
            (source/'train_coupled_40hz.sh').write_text(entry('40', 0))
            if new_entry:
                (source/'train_coupled_20hz.sh').write_text(entry('20', new_exit))
            archive = root/'fixture.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                stream.add(root/'archive_fixture/scripts', arcname='scripts')
            result_path = root/'selected_entry.bin'
            env = {name: value for name, value in os.environ.items()
                   if not name.startswith(('TSENSEI_', 'SLURM_'))}
            env.update(SLURM_JOB_ID='123', TSENSEI_EXPECTED_GPUS='1',
                TSENSEI_JOB_SOURCE_ARCHIVE=str(archive),
                TSENSEI_JOB_SOURCE_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                TEST_ENTRY_RESULT=str(result_path))
            result = subprocess.run(['bash', str(ROOT/'scripts/hpc/job.sbatch'),
                str(root), 'train', '--seed', '43', '--output-dir', '/path with spaces/run'],
                env=env, text=True, capture_output=True, timeout=15)
            self.assertTrue(result_path.exists(), result.stderr)
            selected = result_path.read_bytes().decode().rstrip('\0').split('\0')
            self.assertIn('/path with spaces/run', selected)
            self.assertEqual(selected[-5:], ['--device', 'cuda:0', '--learner-device',
                                             'cuda:0', '--no-collector-pipeline'])
            return result, selected[0]

    def test_new_archive_prefers_20hz_even_with_historical_entry_present(self):
        result, selected = self.run_archive(new_entry=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(selected, '20')

    def test_old_archive_uses_its_own_historical_40hz_entry(self):
        result, selected = self.run_archive(new_entry=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(selected, '40')

    def test_failed_new_entry_does_not_retry_the_historical_entry(self):
        result, selected = self.run_archive(new_entry=True, new_exit=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(selected, '20')


if __name__ == '__main__':
    unittest.main()
