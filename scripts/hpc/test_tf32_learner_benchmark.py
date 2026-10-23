"""Opt-in isolated SAC precision benchmark, run via submit.sh validate.

TF32_BENCH_DIR must contain config.json and a frozen checkpoint.pt. This is
a learner microbenchmark on fixed synthetic replay, not a piano training run.
Imports resolve against the Slurm job's extracted, pinned source archive.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(mode, tag):
    import torch
    sys.path.insert(0, str(Path.cwd()))
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
    from tsensei_v3_2_pianist.rl_torch.replay import Replay
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner

    root = Path(os.environ["TF32_BENCH_DIR"])
    config = json.loads((root / "config.json").read_text())
    torch.set_num_threads(4)
    # v5 selects learner precision during graph capture; inference stays IEEE.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False  # No convolution change.
    saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = saved["agent"]
    agent = SAC(config["observation_size"], config["action_size"],
                SACConfig(**state["config"]), device="cuda:0", seed=42,
                compile_learner=True, fused_optimizer=True,
                action_profile=config["action_profile"])
    agent.load_state_dict(state)
    initial_updates = agent.updates
    replay = Replay(agent.obs_size, agent.action_size, config["replay_capacity"],
                    config["batch_size"], device="cuda:0", seed=20261020,
                    action_profile=agent.action_profile)
    # Identical, synthetic observations/actions/rewards in both precision modes.
    # Fill the actual 2M capacity, avoiding a misleading tiny-cache benchmark.
    generator = torch.Generator(device="cpu").manual_seed(20261020)
    n = 8192
    base = [
        torch.randn(n, agent.obs_size, generator=generator) * .2,
        torch.rand(n, agent.action_size, generator=generator),
        torch.rand(n, generator=generator) * 4 - 5,
        torch.ones(n),
        torch.randn(n, agent.obs_size, generator=generator) * .2,
        torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
    ]
    low, high = replay.action_low.cpu(), replay.action_high.cpu()
    base[1] = low + base[1] * (high - low)
    data_hash = hashlib.sha256(b"".join(x.numpy().tobytes() for x in base)).hexdigest()
    for storage, values in zip(replay.data, base):
        values = values.to("cuda")
        for start in range(0, replay.capacity, n):
            count = min(n, replay.capacity - start)
            storage[start:start+count].copy_(values[:count])
    replay.size = replay.capacity
    replay.total_inserted = replay.capacity
    probe = base[0][:1024].to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    learner = CUDALearner(agent, replay, optimizer_layout=True,
                         matmul_precision="tf32" if mode == "tf32" else "ieee")
    setup_seconds = time.perf_counter() - started
    print(f"{tag}: graph ready in {setup_seconds:.2f}s", flush=True)

    def snapshot(step, metrics):
        tensors = {}
        for name in ("actor", "critic", "target_critic"):
            module = getattr(agent, name)
            tensors[name] = torch.cat([p.detach().flatten() for p in module.parameters()]).cpu()
            grads = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
            if grads:
                tensors[name + "_gradient"] = torch.cat(grads).cpu()
        with torch.no_grad():
            action = agent.act(probe, deterministic=True)
            tensors["action"] = action.cpu()
            tensors["q"] = agent.critic(probe, action).cpu()
            tensors["temperature"] = agent.temperature.detach().cpu()
        tensors["metrics"] = torch.stack([metrics[k].detach() for k in sorted(metrics)]).cpu()
        tensors["update_rng"] = agent.update_rng.get_state().cpu()
        tensors["replay_rng"] = replay.generator.get_state().cpu()
        torch.save(tensors, root / f"{tag}_step{step}.pt")

    previous = 0
    for step in (1, 128, 1024):
        metrics = learner.update(step - previous)
        snapshot(step, metrics)
        previous = step
    # Exclude initialization/compilation and the first 1024 updates from timing.
    times = []
    for _ in range(12):
        torch.cuda.synchronize()
        started = time.perf_counter()
        learner.update(128)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    assert agent.updates - initial_updates == 2560
    result = dict(mode=mode, tag=tag, gpu=torch.cuda.get_device_name(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  learner_matmul_precision="tf32" if mode == "tf32" else "ieee",
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  setup_seconds=setup_seconds, block_seconds=times,
                  median_ms_per_update=statistics.median(times)*1000/128,
                  updates_per_second=128/statistics.median(times),
                  initial_updates=initial_updates, benchmark_updates=2560,
                  synthetic_base_sha256=data_hash, replay_capacity=replay.capacity,
                  replay_size=replay.size, replay_bytes=replay.bytes_allocated,
                  batch_size=replay.batch_size,
                  sac_config=state["config"], checkpoint_sha256=digest(root/"checkpoint.pt"),
                  config_sha256=digest(root/"config.json"),
                  source_file=str(Path(sys.modules[SAC.__module__].__file__).resolve()),
                  source_file_sha256=digest(sys.modules[SAC.__module__].__file__))
    (root / f"{tag}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def test_tf32_learner_benchmark():
    import pytest
    if "TF32_BENCH_DIR" not in os.environ:
        pytest.skip("Opt-in Slurm learner benchmark")
    import torch
    assert os.environ.get("SLURM_JOB_ID"), "GPU work requires Slurm"
    assert torch.cuda.device_count() == 1
    root = Path(os.environ["TF32_BENCH_DIR"])
    uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    protected = (root / "protected_gpu_uuids.txt").read_text()
    assert uuid not in protected, "Benchmark must not share an active training GPU"
    # Alternate ordering, separate processes and compiler caches per mode.
    for mode, tag in (("fp32", "fp32_a"), ("tf32", "tf32_a"),
                      ("tf32", "tf32_b"), ("fp32", "fp32_b")):
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(root / ("compiler_cache_" + mode))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, tag],
                       check=True, env=env)
    results = {tag: json.loads((root/f"{tag}.json").read_text())
               for tag in ("fp32_a", "tf32_a", "tf32_b", "fp32_b")}
    for key in ("synthetic_base_sha256", "checkpoint_sha256", "config_sha256",
                "source_file_sha256", "gpu_uuid"):
        assert len({r[key] for r in results.values()}) == 1, key
    comparisons = {}
    for first, second in (("fp32_a", "tf32_a"), ("fp32_a", "fp32_b"),
                          ("tf32_a", "tf32_b")):
        pair = {}
        for step in (1, 128, 1024):
            a = torch.load(root/f"{first}_step{step}.pt", weights_only=True)
            b = torch.load(root/f"{second}_step{step}.pt", weights_only=True)
            row = {}
            for key in a:
                if key.endswith("_rng"):
                    row[key] = {"equal": torch.equal(a[key], b[key])}
                    assert row[key]["equal"], "Precision comparison must preserve RNG sequence"
                    continue
                x, y = a[key].double(), b[key].double()
                delta = (x-y).abs()
                row[key] = dict(max_abs=delta.max().item(),
                                rms=delta.square().mean().sqrt().item(),
                                relative_l2=(torch.linalg.vector_norm(x-y) /
                                             torch.linalg.vector_norm(x).clamp_min(1e-30)).item(),
                                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
                assert row[key]["finite"]
            pair[str(step)] = row
        comparisons[first+"__"+second] = pair
    fp32 = statistics.median(results[k]["median_ms_per_update"] for k in ("fp32_a", "fp32_b"))
    tf32 = statistics.median(results[k]["median_ms_per_update"] for k in ("tf32_a", "tf32_b"))
    report = dict(scope="Isolated learner synthetic replay microbenchmark; no physical/F1 validation",
                  job_id=os.environ["SLURM_JOB_ID"], runs=results, comparisons=comparisons,
                  fp32_ms_per_update=fp32, tf32_ms_per_update=tf32,
                  speedup=fp32/tf32, protected_training_job=6251)
    (root/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    worker(*sys.argv[1:])
