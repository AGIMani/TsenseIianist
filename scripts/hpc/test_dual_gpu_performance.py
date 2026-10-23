"""Opt-in two-GPU physical/learner benchmark; execute via submit.sh validate.

Uses the pinned training archive and saved current configuration. No checkpoint
from this bounded benchmark is a trained result. Each ABBA worker has an
independent process, identical policy/RNG, full-size replay and real physics.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(tag, graph):
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import (
        build_source_from_config, task_kwargs)
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    from tsensei_v3_2_pianist.rl_torch.collector_pipeline import CollectorPolicy, OneBatchPipeline
    from tsensei_v3_2_pianist.rl_torch.train import evaluate

    root = Path(os.environ["DUAL_GPU_BENCH_DIR"])
    config = json.loads((root / "config.json").read_text())
    assert config["num_envs"] == 512 and config["batch_size"] == 1024
    assert config["steady_updates_per_transition"] == .25
    assert config["source_runtime"]["environment"]["control_hz"] == 20
    assert config["source_runtime"]["environment"]["physics_hz"] == 200
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(1)
    scene, task, _ = build_source_from_config(config)
    kwargs = task_kwargs(config)
    kwargs["physics_graph"] = graph
    with torch.cuda.device(0):
        env = PianoTaskEnv(scene, task, num_envs=512, device="cuda:0", **kwargs)
    saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = saved["agent"]
    # Only the numerical starting state is borrowed; current config determines
    # physical controls. This is not checkpoint continuation or a quality rank.
    assert json.loads(json.dumps(state["config"])) == config["sac"]
    agent = SAC(config["observation_size"], config["action_size"],
                SACConfig(**state["config"]), device="cuda:1", seed=42,
                compile_learner=True, fused_optimizer=True,
                action_profile=config["action_profile"])
    agent.load_state_dict(state)
    collector = CollectorPolicy(agent, "cuda:0")
    replay = Replay(config["observation_size"], config["action_size"],
                    config["replay_capacity"], config["batch_size"],
                    device="cuda:1", seed=20261020,
                    action_profile=config["action_profile"])
    print(f"{tag}: real-physics replay prefill, graph={graph}", flush=True)
    baseline_path = root / "reference_rollout.pt"
    reference = (torch.load(baseline_path, weights_only=True)
                 if tag != "A1" else None)
    trace = {key: [] for key in ("action", "qpos", "qvel", "reward", "observation_sha")}
    max_errors = dict(qpos=0., qvel=0., reward=0.)
    observation_equal = True
    durations = []
    with torch.cuda.device(0), torch.no_grad():
        obs = env.reset()
        for step in range(env.logic.length):
            action = (collector.act(obs) if reference is None
                      else reference["action"][step].to("cuda:0"))
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            result = env.step(action)
            torch.cuda.synchronize(0)
            durations.append(time.perf_counter() - started)
            values = dict(qpos=env.scene.data.qpos[:].cpu(),
                          qvel=env.scene.data.qvel[:].cpu(),
                          reward=result.reward.cpu())
            obs_sha = hashlib.sha256(result.observation.cpu().numpy().tobytes()).hexdigest()
            if reference is None:
                trace["action"].append(action.cpu())
                trace["observation_sha"].append(obs_sha)
                for key, value in values.items():
                    trace[key].append(value)
            else:
                observation_equal &= obs_sha == reference["observation_sha"][step]
                for key, value in values.items():
                    max_errors[key] = max(max_errors[key],
                                         (value - reference[key][step]).abs().max().item())
            replay.insert(Transition(obs, action, result.reward, result.discount,
                                     result.observation, result.terminated, result.truncated))
            obs = result.observation
        assert result.done.all()
        if reference is None:
            for key in trace:
                if key != "observation_sha":
                    trace[key] = torch.stack(trace[key])
            torch.save(trace, baseline_path)
    # Repeat the measured full-song real experience to fill the actual 2M
    # capacity, rather than benchmarking a tiny hot-cache replay.
    actual_rows = replay.size
    for storage in replay.data:
        for start in range(actual_rows, replay.capacity, actual_rows):
            count = min(actual_rows, replay.capacity - start)
            storage[start:start + count].copy_(storage[:count])
    replay.size = replay.capacity
    replay.pointer = 0
    replay.total_inserted = replay.capacity
    del reference, trace
    # Restore the same action RNG in every run, regardless of prefill source.
    collector.action_rng.set_state(state["action_rng"])
    torch.cuda.set_device(1)
    started = time.perf_counter()
    learner = CUDALearner(agent, replay, optimizer_layout=True, matmul_precision="tf32")
    graph_setup = time.perf_counter() - started
    assert not torch.backends.cuda.matmul.allow_tf32
    with torch.cuda.device(0):
        obs = env.reset()

    def collect(step, version):
        with torch.cuda.device(0), torch.no_grad():
            action = collector.act(obs)
            result = env.step(action)
            torch.cuda.synchronize(0)
            return obs, action, result

    def commit(batch):
        nonlocal obs
        before, action, result = batch
        replay.insert(Transition(before, action, result.reward, result.discount,
                                 result.observation, result.terminated, result.truncated))
        with torch.cuda.device(0):
            obs = result.observation
            ids = result.done.nonzero(as_tuple=False).flatten()
            if ids.numel():
                obs = env.reset(ids)
            torch.cuda.synchronize(0)

    def update(step):
        with torch.cuda.device(1):
            learner.update(128)

    print(f"{tag}: simultaneous collection/learning timing", flush=True)
    blocks = []
    initial_updates = agent.updates
    with OneBatchPipeline(snapshot=lambda: collector.sync_from(agent),
                          collect=collect, update=update, commit=commit,
                          learner_version=lambda: agent.updates) as pipeline:
        for block in range(7):
            started = time.perf_counter()
            for offset in range(32):
                pipeline.step(block * 32 + offset + 1)
            # A measurement boundary flush is included consistently.
            pipeline.flush()
            blocks.append(time.perf_counter() - started)
        pipeline_state = pipeline.state()
    assert agent.updates - initial_updates == 224 * 128
    assert pipeline_state["max_actor_lag_at_collection_end_updates"] <= 128
    assert pipeline_state["pending_vector_step"] is None
    # Evaluate the original fixed policy, not policy parameters altered during
    # the timing run. Physics and metric-graph correctness can thus be compared.
    agent.load_state_dict(state)
    collector.sync_from(agent)
    with torch.cuda.device(0):
        eval_env = PianoTaskEnv(scene, task, num_envs=3, device="cuda:0", **kwargs)
        evaluations = []
        for enabled in (False, True):
            started = time.perf_counter()
            metrics = evaluate(collector, eval_env, 3, metrics_graph=enabled)
            torch.cuda.synchronize(0)
            evaluations.append(dict(metrics_graph=enabled,
                                    seconds=time.perf_counter() - started, metrics=metrics))
    result = dict(tag=tag, physics_graph=graph, gpu=torch.cuda.get_device_name(0),
                  gpu_uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in (0, 1)],
                  graph_setup_seconds=graph_setup, original_real_replay_rows=actual_rows,
                  replay_capacity=replay.capacity, batch_size=replay.batch_size,
                  physical_step_median_seconds=statistics.median(durations[10:]),
                  physical_max_abs_error=max_errors, observations_bitwise_equal=observation_equal,
                  pipeline_block_seconds=blocks, warmup_blocks_excluded=1,
                  pipeline_step_seconds=statistics.median(blocks[1:]) / 32,
                  pipeline_transitions_per_second=512 * 32 / statistics.median(blocks[1:]),
                  pipeline_state=pipeline_state, evaluations=evaluations,
                  config_sha256=sha(root / "config.json"),
                  checkpoint_sha256=sha(root / "checkpoint.pt"),
                  fp32_inference=not torch.backends.cuda.matmul.allow_tf32)
    save_json(root / f"{tag}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}), flush=True)
    assert observation_equal and all(v == 0 for v in max_errors.values()), (
        "Graph candidate differs from the identical-action reference; do not adopt automatically")


def compare(a, b, path="", differences=None):
    """Compare complete metric trees, allowing only tiny reduction roundoff."""
    import math
    differences = [] if differences is None else differences
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            differences.append(path + ": keys")
        for key in a.keys() & b.keys():
            compare(a[key], b[key], path + "/" + key, differences)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append(path + ": length")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + f"/{i}", differences)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
            differences.append(dict(path=path, baseline=a, candidate=b))
    elif a != b:
        differences.append(dict(path=path, baseline=a, candidate=b))
    return differences


def test_dual_gpu_performance():
    import pytest
    if "DUAL_GPU_BENCH_DIR" not in os.environ:
        pytest.skip("Opt-in Slurm dual-GPU benchmark")
    import torch
    assert os.environ.get("SLURM_JOB_ID")
    assert torch.cuda.device_count() == 2
    root = Path(os.environ["DUAL_GPU_BENCH_DIR"])
    assert not (root / "A1.json").exists(), "Use a new directory for a new comparison"
    runs = []
    for tag, graph in (("A1", False), ("B1", True), ("B2", True), ("A2", False)):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), tag, str(int(graph))],
                       check=True, timeout=2400)
        runs.append(json.loads((root / f"{tag}.json").read_text()))
    differences = []
    reference = runs[0]["evaluations"][0]["metrics"]
    for run in runs:
        for evaluation in run["evaluations"]:
            differences.extend(compare(reference, evaluation["metrics"],
                                       f'{run["tag"]}/metrics_graph={evaluation["metrics_graph"]}'))
    baseline = statistics.median(r["pipeline_step_seconds"] for r in runs if not r["physics_graph"])
    candidate = statistics.median(r["pipeline_step_seconds"] for r in runs if r["physics_graph"])
    report = dict(job_id=os.environ["SLURM_JOB_ID"], hostname=os.uname().nodename,
                  scope="Bounded real-physics dual-GPU ABBA benchmark; not long-run convergence evidence",
                  runs=runs, baseline_step_seconds=baseline, candidate_step_seconds=candidate,
                  pipeline_speedup=baseline / candidate, evaluation_differences=differences,
                  eligible_for_adoption=not differences,
                  source_archive_sha256=os.environ["TSENSEI_JOB_SOURCE_SHA256"])
    save_json(root / "report.json", report)
    assert not differences, "Full evaluation metrics changed; inspect before adoption"


if __name__ == "__main__":
    worker(sys.argv[1], bool(int(sys.argv[2])))
