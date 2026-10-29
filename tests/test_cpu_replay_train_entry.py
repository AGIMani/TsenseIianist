"""Opt-in real train.run acceptance with the current source recipe, under Slurm.

PIANIST_CPU_REPLAY_SCALE_DIR supplies config.json and source_archive_sha256.txt.
The default is a one-GPU sequential diagnostic, not the production two-GPU
layout. Set PIANIST_CPU_REPLAY_TRAIN_PIPELINE=1 in a two-GPU allocation to also
verify the real entrypoint's final pending-update flush. Neither mode changes
the default launcher or claims full-score training/convergence acceptance.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _mode():
    value = os.environ.get('PIANIST_CPU_REPLAY_TRAIN_PIPELINE', '0')
    assert value in ('0', '1'), 'PIANIST_CPU_REPLAY_TRAIN_PIPELINE must be 0 or 1'
    return ('pipeline' if value == '1' else 'single_gpu'), value == '1'


def _arguments(config, work, pipeline):
    from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

    argument_parser = parser()
    args = argument_parser.parse_args([])
    # Reconstruct all supported original CLI settings, then replace file paths
    # with copies of the fully resolved contracts. Never read the parent's
    # original project directory or its obsolete relative JSON paths.
    for action in argument_parser._actions:
        if action.dest in config['cli']:
            value = deepcopy(config['cli'][action.dest])
            if action.type is Path and value is not None:
                value = Path(value)
            setattr(args, action.dest, value)
    for argument, key in (('reward_config', 'reward_config'),
                          ('control_limits', 'control_limits'),
                          ('finger_clearance_config', 'finger_clearance')):
        value = config.get(key)
        path = work / (key + '.json')
        if value is not None:
            _save(path, value)
        setattr(args, argument, path if value is not None else None)
    coupling = config.get('finger_coupling')
    if coupling is not None:
        assert coupling['mode'] == 'distal_angle_range_v1'
        ranges = work / 'finger_coupling_ranges.json'
        _save(ranges, coupling['angle_ranges_deg'])
        args.finger_coupling_angle_ranges = ranges
        args.finger_coupling_angle = None

    overrides = dict(device='cuda:0', collector_devices='cuda:0',
        learner_device='cuda:1' if pipeline else 'cuda:0', collector_pipeline=pipeline,
        replay_device='cpu', num_envs=4, vector_steps=3, replay_capacity=32,
        batch_size=4, replay_prefill_episodes=0, warmup_transitions=0,
        learning_start_transitions=1, updates_per_transition=.25,
        eval_interval=0, eval_at_start=False, save_interval=0, save_forward=False,
        log_interval=1, progress=False, inspect_only=False, max_episode_steps=None,
        warm_start=None, stage2_from=None, finetune_from=None,
        allow_reward_change=False, allow_control_change=False,
        output_dir=work / 'run')
    for key, value in overrides.items():
        setattr(args, key, value)
    effective = effective_config(args)
    unchanged = ('sac', 'stage', 'preparation', 'action_profile', 'no_arms',
        'disable_wrist_tz', 'disable_wrist_yaw', 'hand_collision_detection_only',
        'self_collision_detection_only', 'wrist_tz_physics', 'acceleration',
        'self_collision_weight', 'collision_diagnostics', 'control_limits',
        'finger_coupling', 'arm_follow', 'finger_clearance', 'reward_config')
    for key in unchanged:
        assert json.dumps(effective[key], sort_keys=True) == json.dumps(config[key], sort_keys=True), \
            f'Diagnostic changed physical/learner contract: {key}'
    assert effective['gradient_steps_per_vector_step'] == 1
    assert effective['steady_updates_per_transition'] == .25
    assert effective['acceleration']['learner_graph'] is True
    assert effective['acceleration']['learner_matmul_precision'] == 'tf32'
    assert effective['arm_follow'] is None
    assert effective['preparation']['mode'] == 'first_note_v1'
    return args, {key: str(value) if isinstance(value, Path) else value
                  for key, value in overrides.items()}, unchanged


def _worker():
    assert os.environ.get('SLURM_JOB_ID'), 'Never initialize this test outside Slurm'
    name, pipeline = _mode()
    acceptance = Path(os.environ['PIANIST_CPU_REPLAY_SCALE_DIR']).resolve()
    report_path = acceptance / f'train_entry_{name}.json'
    work = acceptance / f'train_entry_{name}'
    assert not report_path.exists() and not work.exists(), 'Preserve earlier acceptance evidence'
    work.mkdir()
    report = dict(status='running', passed=False, kind=f'train_entry_{name}',
        job_id=os.environ['SLURM_JOB_ID'], host=os.uname().nodename, pid=os.getpid(),
        scope='Real train.run with 4 worlds / 3 steps / batch 4 / replay 32; '
              'diagnostic prefill=0, warmup=0, learning-start=1, eval disabled; '
              'current physical/reward/preparation/learner contracts unchanged',
        production_pipeline_exercised=pipeline, formal_training=False,
        script_sha256=_sha(__file__), output_directory=str(work))
    started = time.perf_counter()
    try:
        expected_sha = (acceptance / 'source_archive_sha256.txt').read_text().split()[0]
        actual_sha = os.environ.get('TSENSEI_JOB_SOURCE_SHA256')
        assert len(expected_sha) == 64 and actual_sha == expected_sha, 'Frozen source SHA mismatch'
        config = json.loads((acceptance / 'config.json').read_text())
        report.update(source_archive_sha256=actual_sha,
                      input_config_sha256=_sha(acceptance / 'config.json'))
        _save(report_path, report)
        import torch
        assert torch.cuda.is_available(), 'CUDA allocation required'
        assert torch.cuda.device_count() >= (2 if pipeline else 1)
        torch.set_num_threads(4)
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_source_from_config
        from tsensei_v3_2_pianist.rl_torch.train import run

        args, overrides, unchanged = _arguments(config, work, pipeline)
        report.update(diagnostic_overrides=overrides,
                      preserved_contracts=list(unchanged))
        _save(report_path, report)
        # This verifies exact source fingerprints and dependency versions before
        # building the real scene. No alternate or cached source is substituted.
        args.scene, args.task, args.source_config = build_source_from_config(config)
        assert json.loads(args.source_config.read_text()) == config['source_runtime']
        task_manifest = json.loads((args.task / 'task.json').read_text())
        assert task_manifest['length'] == config['replay_prefill']['episode_steps']
        report.update(episode_steps=task_manifest['length'],
                      control_dt=task_manifest['control_dt'],
                      physics_dt=task_manifest['physics_dt'])
        _save(report_path, report)
        summary = run(args)
        saved_config = json.loads((args.output_dir / 'config.json').read_text())
        saved_summary = json.loads((args.output_dir / 'summary.json').read_text())
        final_path = args.output_dir / 'final.pt'
        checkpoint = torch.load(final_path, map_location='cpu', weights_only=True)
        metadata = checkpoint['metadata']
        validate_config = runpy.run_path(str(ROOT / 'scripts/eval_from_source.py'))['validate_config']
        validate_config(saved_config, metadata)
        assert saved_summary == summary
        assert summary['status'] == 'completed'
        assert summary['stop_reason'] == 'requested_budget_exhausted'
        assert summary['vector_steps'] == 3 and summary['transitions'] == 12
        assert summary['replay_insertions'] == 12 and summary['replay_size'] == 12
        assert summary['updates_this_run'] == summary['updates_total'] == 3
        assert metadata['vector_step'] == 3 and metadata['updates'] == 3
        assert checkpoint['agent']['updates'] == 3
        assert summary['random_action_transitions'] == 0
        assert summary['replay_prefill']['episodes'] == 0
        assert summary['evaluation_timing']['count'] == 0
        assert summary['final_evaluation'] is None
        assert saved_config['source_runtime'] == config['source_runtime']
        for key in unchanged:
            assert saved_config[key] == config[key], key
        storage = saved_config['replay_storage']
        assert storage == summary['replay_storage']
        assert storage['device'] == 'cpu' and storage['pool_pinned'] is False
        assert storage['sampling_rng_backend'] == 'pytorch_cpu'
        per_row = 4 * (2 * saved_config['observation_size'] + saved_config['action_size'] + 2) + 2
        assert saved_config['replay_bytes'] == 32 * per_row
        assert saved_config['replay_host_memory_check']['pool_bytes'] == 32 * per_row
        staging = summary['replay_staging']
        assert staging['host_buffer_count'] == 2
        assert staging['host_pinned_bytes'] == 2 * 4 * per_row
        assert staging['device_batch_bytes'] == 4 * per_row
        for optimizer in ('actor_optimizer', 'critic_optimizer', 'temp_optimizer'):
            states = checkpoint['agent'][optimizer]['state']
            assert states and all(value['step'].item() == 3 for value in states.values()), optimizer
            for state in states.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert bool(torch.isfinite(value).all())
        metrics = [json.loads(line) for line in (args.output_dir / 'metrics.jsonl').read_text().splitlines()]
        assert [row['vector_step'] for row in metrics] == [1, 2, 3]
        assert metrics[-1]['updates'] == 3
        assert all(row['reward_terms'] and row['reward_raw_terms'] and row['reward_weighted_terms']
                   for row in metrics)
        if pipeline:
            state = summary['collector_pipeline_state']
            assert state['pending_vector_step'] is None
            assert state['learner_updated_through_vector_step'] == 3
            assert state['collected_vector_steps'] == 3
            assert metadata['collector_pipeline_state'] == state
        else:
            assert summary['collector_pipeline_state'] is None
            assert summary['execution_layout']['mode'] == 'single_device_v1'
        report.update(status='passed', passed=True, config_checkpoint_validated=True,
            updates_total=3, replay_insertions=12, replay_storage=storage,
            replay_staging=staging, final_checkpoint_sha256=_sha(final_path),
            final_config_sha256=_sha(args.output_dir / 'config.json'),
            final_summary_sha256=_sha(args.output_dir / 'summary.json'),
            collector_pipeline_state=summary['collector_pipeline_state'])
    except BaseException as error:
        report.update(status='failed', passed=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        _save(report_path, report)


def test_cpu_replay_real_train_entry():
    import pytest
    if 'PIANIST_CPU_REPLAY_SCALE_DIR' not in os.environ:
        pytest.skip('Opt-in Slurm real train.run CPU replay acceptance')
    assert os.environ.get('SLURM_JOB_ID'), 'Never run physical acceptance outside Slurm'
    name, _ = _mode()
    root = Path(os.environ['PIANIST_CPU_REPLAY_SCALE_DIR']).resolve()
    report_path = root / f'train_entry_{name}.json'
    assert not report_path.exists(), 'Use a fresh acceptance directory; preserve evidence'
    timeout = int(os.environ.get('PIANIST_CPU_REPLAY_TRAIN_TIMEOUT', '1800'))
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                cwd=ROOT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status='failed', passed=False, error_type='TimeoutExpired',
                      error=f'Train entry worker exceeded {timeout} seconds')
        _save(report_path, report)
        raise
    if result.returncode:
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status='failed', passed=False, worker_returncode=result.returncode)
        _save(report_path, report)
    assert result.returncode == 0, f'Real train entry failed; inspect {report_path}'
    assert json.loads(report_path.read_text())['passed'] is True


if __name__ == '__main__':
    assert sys.argv[1:] == ['--worker']
    sys.path.insert(0, str(ROOT))
    _worker()
