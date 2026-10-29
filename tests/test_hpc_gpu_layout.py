"""CPU-only checks for the four-GPU Slurm resource layout.

These tests stub sbatch and inspect the generated command. They do not submit
jobs or import CUDA libraries.
"""
from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / 'scripts/hpc/submit.sh'


def fake_sbatch(tmp_path):
    sbatch = tmp_path / 'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    sbatch.chmod(0o755)
    return sbatch


def run_submit(tmp_path, *args):
    fake_sbatch(tmp_path)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    return subprocess.run(['bash', str(SUBMIT), *args], cwd=ROOT, env=env,
                          text=True, capture_output=True, timeout=10)


def test_four_gpu_batch_requests_four_gpus_and_host_threads(tmp_path):
    result = run_submit(tmp_path, '--gpus', '4', 'probe')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert '--gres=gpu:4' in args
    assert '--cpus-per-task=32' in args
    assert str(ROOT / 'scripts/hpc/job.sbatch') in args


def test_two_gpu_default_keeps_historical_resource_request(tmp_path):
    result = run_submit(tmp_path, '--gpus', '2', 'probe')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert '--gres=gpu:2' in args
    assert '--cpus-per-task=32' not in args


def test_four_gpu_reserve_is_named_and_printable_without_submitting(tmp_path):
    result = run_submit(tmp_path, '--gpus', '4', 'reserve')
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert '--gres=gpu:4' in args
    assert '--cpus-per-task=32' in args
    assert '--job-name=TsenseI-reserve4' in args


def test_invalid_three_gpu_request_is_rejected_before_sbatch(tmp_path):
    result = run_submit(tmp_path, '--gpus', '3', 'probe')
    assert result.returncode == 2
    assert 'requires 1, 2 or 4' in result.stderr
    assert result.stdout == ''


def test_entry_declares_two_gpus_and_cpu_replay():
    text = (ROOT / 'all_use_this_train.sh').read_text()
    assert '--gpus 2' in text
    assert '--collector-devices cuda:0 --learner-device cuda:1' in text
    assert '--replay-device cpu' in text

