"""Slurm-only 3M CUDA replay learner baseline for the CPU replay comparison.

Use PIANIST_CPU_REPLAY_SCALE_DIR with the current config.json and frozen-source
SHA. This measures the previous 3M GPU pool against the separate 24M CPU-pool
component test. It is not a full training-speed comparison: there is no physics,
collection, prefill, evaluation or checkpoint export in either timed block.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
KIND = 'learner_baseline_gpu_3m'


def _helpers():
    path = Path(__file__).with_name('test_cpu_replay_scale.py')
    spec = importlib.util.spec_from_file_location('_cpu_replay_scale_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _benchmark(config, torch, report, helpers, report_path):
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition

    assert torch.cuda.device_count() == 1, 'Allocate exactly one GPU for this learner baseline'
    learner_device = torch.device('cuda:0')
    torch.cuda.set_device(learner_device)
    capacity = 3_000_000
    obs, action = config['observation_size'], config['action_size']
    assert obs == 1406 and action == 31
    assert list(config['sac']['actor_hidden']) == [256, 256, 256]
    assert list(config['sac']['critic_hidden']) == [256, 256, 256]
    assert config['batch_size'] == 1024
    expected_bytes = capacity * ((2 * obs + action + 2) * 4 + 2)
    free, total = torch.cuda.mem_get_info(learner_device)
    assert free >= expected_bytes + 8 * GIB, \
        'GPU baseline requires the 3M FP32 pool plus 8 GiB free overhead'
    report.update(scope='Old 3M GPU replay capacity learner component baseline versus '
        'the separate 24M CPU replay test; not whole-training throughput',
        diagnostic_overrides=dict(replay_device='cuda:0', replay_capacity=capacity),
        input_replay_device=config['replay_device'], input_replay_capacity=config['replay_capacity'],
        observation_size=obs, action_size=action, batch_size=1024,
        learner_device=str(learner_device), expected_replay_bytes=expected_bytes,
        expected_replay_gib=expected_bytes / GIB,
        device_free_before_bytes=free, device_total_bytes=total,
        replay_rng_comparison='CPU and CUDA uniform-with-replacement generators differ; '
                              'this comparison is not bitwise-equivalent training',
        timed_scope='replay sampling + CUDA gather + full sequential SAC update blocks',
        excluded_scope='physics, collection, prefill, graph setup, evaluation, checkpoint export')
    helpers._save(report_path, report)
    agent = helpers._agent(config, device=str(learner_device))
    allocated_before = torch.cuda.memory_allocated(learner_device)
    replay = Replay(obs, action, capacity, 1024, device=str(learner_device),
                    seed=20261027, action_profile=config['action_profile'])
    assert replay.bytes_allocated == expected_bytes
    assert all(value.device == learner_device for value in replay.data)
    assert all(value.dtype == torch.float32 for value in replay.data[:5])
    assert all(value.dtype == torch.bool for value in replay.data[5:])
    # Construct exactly the CPU-capacity benchmark's fixture before transfer.
    # Every field is then copied over the whole CUDA pool, including its tail.
    count = 8192
    generator = torch.Generator(device='cpu').manual_seed(20261027)
    low, high = replay.action_low.cpu(), replay.action_high.cpu()
    host_base = Transition(torch.randn(count, obs, generator=generator) * .2,
        low + torch.rand(count, action, generator=generator) * (high - low),
        torch.rand(count, generator=generator) * 4 - 5, torch.ones(count),
        torch.randn(count, obs, generator=generator) * .2,
        torch.zeros(count, dtype=torch.bool), torch.zeros(count, dtype=torch.bool))
    report['synthetic_fixture_sha256'] = hashlib.sha256(
        b''.join(value.numpy().tobytes() for value in host_base)).hexdigest()
    base = Transition(*(value.to(learner_device) for value in host_base))
    torch.cuda.synchronize(learner_device)
    started = time.perf_counter()
    touched = 0
    for name, storage, values in zip(Transition._fields, replay.data, base):
        for start in range(0, capacity, count):
            stop = min(start + count, capacity)
            storage[start:stop].copy_(values[:stop - start])
            touched += (stop - start) * storage[0].numel() * storage.element_size()
        print(f'Full GPU baseline touched {name}: {touched / GIB:.3f} GiB', flush=True)
    torch.cuda.synchronize(learner_device)
    report['touch_seconds'] = time.perf_counter() - started
    assert touched == expected_bytes
    replay.size, replay.pointer, replay.total_inserted = capacity, 0, capacity
    probes = torch.tensor([0, count - 1, count, capacity // 2, capacity - 1], device=learner_device)
    assert all(torch.equal(storage[probes], values[probes % count])
               for storage, values in zip(replay.data, base))
    assert torch.cuda.memory_allocated(learner_device) - allocated_before >= expected_bytes
    report.update(touched_bytes=touched, synthetic_fixture_rows=count,
                  gpu_after_touch=helpers._gpu_memory(torch),
                  storage_contract=replay.storage_contract)
    del host_base, base
    helpers._save(report_path, report)
    initial_updates = agent.updates
    started = time.perf_counter()
    learner = CUDALearner(agent, replay, optimizer_layout=True, matmul_precision='tf32')
    report['learner_setup_seconds'] = time.perf_counter() - started
    assert agent.updates == initial_updates
    assert not torch.backends.cuda.matmul.allow_tf32
    staging = learner.staging_contract
    assert staging['host_buffer_count'] == 0
    assert staging['host_pinned_bytes'] == 0 and staging['device_batch_bytes'] == 0
    assert staging['transfer'] == 'cuda_replay_graph_gather_v1'
    report['staging_contract'] = staging
    helpers._save(report_path, report)
    learner.update(128)
    assert agent.updates - initial_updates == 128
    blocks = []
    for block in range(8):
        torch.cuda.synchronize(learner_device)
        started = time.perf_counter()
        metrics = learner.update(128)
        torch.cuda.synchronize(learner_device)
        blocks.append(time.perf_counter() - started)
        assert all(bool(torch.isfinite(value).all()) for value in metrics.values())
        print(f'GPU 3M learner block {block + 1}/8: {blocks[-1]:.6f}s', flush=True)
    assert agent.updates - initial_updates == 1152
    assert replay.size == capacity and replay.bytes_allocated == expected_bytes
    median = statistics.median(blocks)
    report.update(warmup_updates=128, steady_updates=1024, total_updates=1152,
        steady_block_updates=128, steady_block_seconds=blocks,
        steady_block_median_seconds=median, steady_updates_per_second=128 / median,
        replay_capacity=capacity, replay_bytes=replay.bytes_allocated,
        gpu_final=helpers._gpu_memory(torch), host_final=helpers._host_memory(),
        actual_torch_allocated_bytes=torch.cuda.memory_allocated(learner_device),
        actual_torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(learner_device))


def _worker():
    assert os.environ.get('SLURM_JOB_ID'), 'Never initialize this baseline outside Slurm'
    root = Path(os.environ['PIANIST_CPU_REPLAY_SCALE_DIR']).resolve()
    path = root / f'{KIND}.json'
    assert not path.exists(), 'Preserve previous acceptance evidence'
    helpers = _helpers()
    report = dict(kind=KIND, status='running', passed=False, formal_training=False)
    started = time.perf_counter()
    torch = None
    try:
        config, torch = helpers._load(root, report)
        report['helper_script_sha256'] = report.pop('script_sha256')
        report['script_sha256'] = helpers._sha(__file__)
        helpers._save(path, report)
        _benchmark(config, torch, report, helpers, path)
        report.update(status='passed', passed=True)
    except BaseException as error:
        report.update(status='failed', passed=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        if torch is not None:
            try:
                report['gpu_exit'] = helpers._gpu_memory(torch)
            except BaseException as error:
                report['exit_memory_error'] = repr(error)
        helpers._save(path, report)


def test_gpu_replay_3m_learner_baseline():
    import pytest
    if 'PIANIST_CPU_REPLAY_SCALE_DIR' not in os.environ:
        pytest.skip('Opt-in Slurm 3M CUDA replay learner baseline')
    assert os.environ.get('SLURM_JOB_ID'), 'Never run this GPU test outside Slurm'
    root = Path(os.environ['PIANIST_CPU_REPLAY_SCALE_DIR']).resolve()
    path = root / f'{KIND}.json'
    assert not path.exists(), 'Use a fresh report path; preserve earlier evidence'
    timeout = int(os.environ.get('PIANIST_CPU_REPLAY_BASELINE_TIMEOUT', '1800'))
    helpers = _helpers()
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                cwd=ROOT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        report = json.loads(path.read_text()) if path.exists() else dict(kind=KIND)
        report.update(status='failed', passed=False, error_type='TimeoutExpired',
                      error=f'Learner baseline exceeded {timeout} seconds')
        helpers._save(path, report)
        raise
    if result.returncode:
        report = json.loads(path.read_text()) if path.exists() else dict(kind=KIND)
        report.update(status='failed', passed=False, worker_returncode=result.returncode)
        helpers._save(path, report)
    assert result.returncode == 0, f'GPU learner baseline failed; inspect {path}'
    assert json.loads(path.read_text())['passed'] is True


if __name__ == '__main__':
    assert sys.argv[1:] == ['--worker']
    sys.path.insert(0, str(ROOT))
    _worker()
