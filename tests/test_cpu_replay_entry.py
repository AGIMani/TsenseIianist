"""CPU-only budget, memory admission and real shell submission checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tsensei_v3_2_pianist.rl_torch import host_memory
from tsensei_v3_2_pianist.rl_torch.train import (
    effective_config, parser, resolve_replay_prefill, updates_this_step,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_score_cpu_budget_does_not_drop_learning_updates():
    args = parser().parse_args(['--num-envs', '4096', '--batch-size', '1024',
        '--vector-steps', '13381', '--replay-capacity', '48000000', '--replay-device', 'cpu',
        '--learner-device', 'cuda:1', '--updates-per-transition', '.25',
        '--learning-start-transitions', '10000', '--replay-prefill-episodes', '1',
        '--eval-interval', '1000'])
    config = effective_config(args)
    resolve_replay_prefill(args, config, 11137)
    assert config['replay_device'] == 'cpu'
    assert config['replay_prefill']['required_transitions'] == 45617152
    assert config['learning_start_transition_1_based'] == 45617153
    assert updates_this_step(args, 11137) == 0
    assert updates_this_step(args, 11138) == 1024
    assert sum(updates_this_step(args, step) for step in range(1, 13382)) == 2297856
    assert config['requested_transitions'] == 54808576
    from tsensei_v3_2_pianist.score_validation import release_tail_steps
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe
    recipe = make_recipe(control_hz=50, physics_hz=500, release_tail_time=.26,
                         lookahead=10, exact_note_timing=True)
    assert recipe['environment']['physics_hz'] / recipe['environment']['control_hz'] == 10
    assert release_tail_steps(recipe['environment']['release_tail_time'], 1 / 50) == 13
    with pytest.raises(ValueError, match='whole control frames'):
        make_recipe(control_hz=50, physics_hz=500, release_tail_time=.25)
    config['replay_capacity'] = 24000000
    with pytest.raises(ValueError, match='at least one full score'):
        resolve_replay_prefill(args, config, 11137)
    assert effective_config(parser().parse_args(['--learner-device', 'cuda:1']))['replay_device'] == 'cuda:1'


def test_host_capacity_rejects_old_128g_job_without_allocating(monkeypatch):
    assert host_memory.replay_bytes(1406, 31, 24000000) == 273168000000
    monkeypatch.setattr(host_memory, 'host_memory_limits', lambda: {
        'host_available_bytes': 1024**4, 'slurm_requested_bytes': 128 * 1024**3})
    with pytest.raises(MemoryError, match='--mem-gb'):
        host_memory.check_host_replay_memory(1406, 31, 24000000)
    monkeypatch.setattr(host_memory, 'host_memory_limits', lambda: {
        'host_available_bytes': 1024**4, 'slurm_requested_bytes': 384 * 1024**3})
    proof = host_memory.check_host_replay_memory(1406, 31, 24000000)
    assert proof['required_bytes'] < 384 * 1024**3
    monkeypatch.setattr(host_memory, 'host_memory_limits', lambda: {
        'host_available_bytes': 1024**4, 'cgroup_available_bytes': 100 * 1024**3})
    with pytest.raises(MemoryError):
        host_memory.check_host_replay_memory(1406, 31, 24000000)


def test_50hz_pool_requires_larger_host_request_without_allocating(monkeypatch):
    assert host_memory.replay_bytes(1406, 31, 48000000) == 546336000000
    monkeypatch.setattr(host_memory, 'host_memory_limits', lambda: {
        'slurm_requested_bytes': 384 * 1024**3})
    with pytest.raises(MemoryError):
        host_memory.check_host_replay_memory(1406, 31, 48000000)
    monkeypatch.setattr(host_memory, 'host_memory_limits', lambda: {
        'slurm_requested_bytes': 640 * 1024**3})
    assert host_memory.check_host_replay_memory(1406, 31, 48000000)['required_bytes'] < 640 * 1024**3


def test_submit_requests_host_memory_not_extra_gpus(tmp_path):
    stub = tmp_path/'sbatch'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']}
    result = subprocess.run(['/bin/bash', str(ROOT/'scripts/hpc/submit.sh'),
        '--gpus', '2', '--mem-gb', '384', 'validate', 'tests/test_cpu_replay.py'],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--mem=384G' in result.stdout.splitlines()
    assert '--gres=gpu:2' in result.stdout.splitlines()
    result = subprocess.run(['/bin/bash', str(ROOT/'scripts/hpc/submit.sh'),
        '--mem-gb', '0', 'validate'], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert result.stdout == ''


@pytest.mark.skipif(not (ROOT/'all_use_this_train.sh').exists(), reason='Outer entry only')
def test_real_default_entry_has_cpu_replay_and_overrideable_memory(tmp_path):
    capture = tmp_path/'capture.json'
    stub = tmp_path/'bash'
    stub.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                    'open(os.environ["ENTRY_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH'],
           'ENTRY_CAPTURE': str(capture)}
    result = subprocess.run(['/bin/bash', str(ROOT/'all_use_this_train.sh'),
        '--mem-gb', '400', '--eval-interval', '1200'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(capture.read_text())
    for key, value in (('--gpus','2'), ('--mem-gb','400'), ('--num-envs','4096'),
                       ('--vector-steps','13381'), ('--replay-capacity','48000000'),
                       ('--control-hz','50'), ('--physics-hz','500'), ('--score-validation','legacy'), ('--release-tail-time','0.26'),
                       ('--replay-device','cpu'), ('--batch-size','1024'),
                       ('--updates-per-transition','0.25'), ('--eval-interval','1000')):
        assert command[command.index(key)+1] == value
    assert command[-2:] == ['--eval-interval','1200']
