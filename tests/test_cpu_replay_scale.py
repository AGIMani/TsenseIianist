"""Opt-in, Slurm-only acceptance for 4096 worlds and full FP32 CPU replay.

PIANIST_CPU_REPLAY_SCALE_DIR must contain this run's config.json and
source_archive_sha256.txt. The recipe must fingerprint the current frozen
source, with current dependency versions. No checkpoint is read. Select a
test with pytest -k; each runs in an isolated child process and writes its own
audit JSON, including failures. These are bounded acceptance tests, not a full
score, training convergence, or speedup claim. ``collector_only`` and
``collector_baseline_512`` use one GPU and cannot substitute for the separate
two-GPU ``physical`` pipeline test. The 512-world override is diagnostic only;
the supplied training config and default remain at 4096 worlds.
"""

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


GIB = 1024 ** 3


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _save(path, result):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemAvailable", "MemTotal"):
            values[name] = int(value.split()[0]) * 1024
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmSwap"):
            values[name] = int(value.split()[0]) * 1024
    # Account for every enclosing cgroup v2 limit, not only the host's free RAM.
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy != "0" or controllers:
            continue
        mount = Path("/sys/fs/cgroup")
        group = mount / relative.lstrip("/")
        for directory in (group, *group.parents):
            if directory != mount and mount not in directory.parents:
                break
            maximum, current = directory / "memory.max", directory / "memory.current"
            if maximum.is_file() and current.is_file():
                limit = maximum.read_text().strip()
                if limit != "max":
                    used = int(current.read_text())
                    limits.append(dict(path=str(directory), maximum=int(limit), used=used,
                                       headroom=int(limit) - used))
    values["cgroup_limits"] = limits
    values["available_bytes"] = min([values["MemAvailable"],
                                     *(item["headroom"] for item in limits)])
    return values


def _gpu_memory(torch):
    result = {}
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        result[str(device)] = dict(name=torch.cuda.get_device_name(device),
            uuid=str(torch.cuda.get_device_properties(device).uuid), free_bytes=free,
            total_bytes=total, allocated_bytes=torch.cuda.memory_allocated(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device))
    return result


def _load(root, report):
    assert os.environ.get("SLURM_JOB_ID"), "Heavy acceptance requires a Slurm allocation"
    expected = (root / "source_archive_sha256.txt").read_text().split()[0]
    actual = os.environ.get("TSENSEI_JOB_SOURCE_SHA256")
    assert len(expected) == 64 and actual == expected, "Frozen archive SHA mismatch"
    config = json.loads((root / "config.json").read_text())
    report.update(job_id=os.environ["SLURM_JOB_ID"], host=os.uname().nodename,
                  config_sha256=_sha(root / "config.json"), source_archive_sha256=actual,
                  script_sha256=_sha(__file__), pid=os.getpid())
    assert config["num_envs"] == 4096
    assert config["replay_capacity"] == 24_000_000
    assert config["replay_device"] == "cpu"
    assert config["batch_size"] == 1024
    assert config["steady_updates_per_transition"] == .25
    assert config["gradient_steps_per_vector_step"] == 1024
    assert config["collision_diagnostics"] is True
    assert config["action_profile"] == "direct_joint_v1"
    assert config["arm_follow"] is None
    assert config["source_runtime"]["environment"]["control_hz"] == 20
    assert config["source_runtime"]["environment"]["physics_hz"] == 1000
    assert config["source_runtime"]["environment"]["exact_note_timing"] is True
    assert config["acceleration"]["learner_matmul_precision"] == "tf32"
    for flag in ("compile_learner", "fused_optimizer", "learner_graph", "optimizer_layout"):
        assert config["acceleration"][flag] is True
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import source_files
    assert source_files() == config["source_runtime"]["source_sha256"], \
        "Input recipe must fingerprint this run's frozen source, not the parent experiment"
    import torch
    assert torch.cuda.device_count() in (1, 2), "Allocate one or two GPUs for capacity acceptance"
    if report["kind"] == "physical":
        assert torch.cuda.device_count() == 2, "Physical pipeline requires collector and learner GPUs"
    if report["kind"] in ("collector_only", "collector_baseline_512"):
        assert torch.cuda.device_count() == 1, "Collector-only acceptance must use one allocated GPU"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(torch.cuda.device_count() - 1)
    report.update(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu_initial=_gpu_memory(torch), host_initial=_host_memory(),
                  reward_config=config["reward_config"])
    print("Acceptance reward config: " + json.dumps(config["reward_config"]), flush=True)
    return config, torch


def _agent(config, *, device="cuda:1", compile_learner=True, fused_optimizer=True):
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
    return SAC(config["observation_size"], config["action_size"],
               SACConfig(**config["sac"]), device=device, seed=config["cli"]["seed"],
               compile_learner=compile_learner, fused_optimizer=fused_optimizer,
               action_profile=config["action_profile"])


def _full_capacity(config, torch, report):
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    obs, action = config["observation_size"], config["action_size"]
    expected_bytes = config["replay_capacity"] * ((2 * obs + action + 2) * 4 + 2)
    before = _host_memory()
    report.update(scope="Full CPU capacity touch plus synthetic learner acceptance; no physics",
                  expected_replay_bytes=expected_bytes,
                  expected_replay_gib=expected_bytes / GIB, host_before_allocation=before)
    assert before["available_bytes"] >= expected_bytes + 8 * GIB, \
        "Insufficient host/cgroup headroom for full FP32 replay plus 8 GiB overhead"
    learner_device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    agent = _agent(config, device=str(learner_device))
    replay = Replay(obs, action, config["replay_capacity"], config["batch_size"],
                    device="cpu", seed=20261027, action_profile=config["action_profile"])
    assert replay.bytes_allocated == expected_bytes
    assert all(value.device.type == "cpu" for value in replay.data)
    assert all(value.dtype == torch.float32 for value in replay.data[:5])
    assert all(value.dtype == torch.bool for value in replay.data[5:])
    # A bounded deterministic fixture touches every backing page of every field.
    # Repeating this fixture is explicitly synthetic; it is never a training replay.
    count = 8192
    generator = torch.Generator(device="cpu").manual_seed(20261027)
    base = Transition(torch.randn(count, obs, generator=generator) * .2,
        replay.action_low + torch.rand(count, action, generator=generator) *
        (replay.action_high - replay.action_low),
        torch.rand(count, generator=generator) * 4 - 5, torch.ones(count),
        torch.randn(count, obs, generator=generator) * .2,
        torch.zeros(count, dtype=torch.bool), torch.zeros(count, dtype=torch.bool))
    report["synthetic_fixture_sha256"] = hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in base)).hexdigest()
    started = time.perf_counter()
    touched = 0
    for name, storage, values in zip(Transition._fields, replay.data, base):
        for start in range(0, replay.capacity, count):
            stop = min(start + count, replay.capacity)
            storage[start:stop].copy_(values[:stop - start])
            touched += (stop - start) * storage[0].numel() * storage.element_size()
        print(f"Full CPU replay touched {name}: {touched / GIB:.3f} GiB", flush=True)
    replay.size = replay.capacity
    replay.pointer = 0
    replay.total_inserted = replay.capacity
    assert touched == expected_bytes
    probes = torch.tensor([0, count - 1, count, replay.capacity // 2, replay.capacity - 1])
    assert all(torch.equal(storage[probes], values[probes % count])
               for storage, values in zip(replay.data, base))
    touched_memory = _host_memory()
    report.update(touch_seconds=time.perf_counter() - started, touched_bytes=touched,
                  host_after_touch=touched_memory)
    assert touched_memory["VmRSS"] >= before["VmRSS"] + expected_bytes * .98, \
        "Full replay pages are not resident; an allocation alone is not a capacity pass"
    assert touched_memory.get("VmSwap", 0) == before.get("VmSwap", 0), \
        "Capacity acceptance must not rely on newly swapped-out replay pages"
    initial_updates = agent.updates
    started = time.perf_counter()
    learner = CUDALearner(agent, replay, optimizer_layout=True, matmul_precision="tf32")
    report["learner_setup_seconds"] = time.perf_counter() - started
    staging = learner.staging_contract
    batch_bytes = expected_bytes // replay.capacity * replay.batch_size
    assert staging["host_buffer_count"] == 2
    assert staging["host_pinned_bytes"] == 2 * batch_bytes
    assert staging["device_batch_bytes"] == batch_bytes
    assert not replay.storage_contract["pool_pinned"]
    report.update(staging_contract=staging, storage_contract=replay.storage_contract,
                  learner_device=str(learner_device))
    assert agent.updates == initial_updates, "Capture must restore update accounting"
    assert not torch.backends.cuda.matmul.allow_tf32
    learner.update(128)
    assert agent.updates - initial_updates == 128
    blocks = []
    for _ in range(8):
        torch.cuda.synchronize(learner_device)
        started = time.perf_counter()
        metrics = learner.update(128)
        torch.cuda.synchronize(learner_device)
        blocks.append(time.perf_counter() - started)
        assert all(bool(torch.isfinite(value).all()) for value in metrics.values())
    assert agent.updates - initial_updates == 128 + 1024
    assert replay.size == 24_000_000 and replay.bytes_allocated == expected_bytes
    assert all(value.device.type == "cpu" for value in replay.data)
    report.update(warmup_updates=128, steady_updates=1024,
                  total_updates=agent.updates - initial_updates,
                  steady_block_updates=128, steady_block_seconds=blocks,
                  steady_updates_per_second=128 / statistics.median(blocks),
                  replay_bytes=replay.bytes_allocated, replay_capacity=replay.capacity,
                  host_final=_host_memory(), gpu_final=_gpu_memory(torch))
    # A full or substantial GPU mirror is forbidden; this generous ceiling also
    # leaves room for compilation, optimizer states, and temporary mini-batches.
    assert torch.cuda.max_memory_allocated(learner_device) < 16 * GIB, \
        "Unexpected GPU residency: CPU replay must only stage bounded mini-batches"


def _physical(config, torch, report):
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_source_from_config, task_kwargs
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    from tsensei_v3_2_pianist.rl_torch.collector_pipeline import CollectorPolicy, OneBatchPipeline
    collector_only = report["kind"] in ("collector_only", "collector_baseline_512")
    num_envs = 512 if report["kind"] == "collector_baseline_512" else config["num_envs"]
    assert num_envs == 4096 or (collector_only and report["kind"] == "collector_baseline_512")
    steps = int(os.environ.get("PIANIST_CPU_REPLAY_PHYSICAL_STEPS", "8"))
    assert 2 <= steps <= 128, "Bounded physical acceptance requires 2..128 control steps"
    scope = (f"Single-GPU {num_envs}-world collector/insertion/reset acceptance; no learner or overlap; "
             "does not validate the default two-GPU pipeline" if collector_only else
             "4096 real worlds and bounded CPU-replay/CUDA-learner pipeline; "
             "one real seed batch before capture, not full-score training prefill")
    if num_envs == 512:
        scope = "Original 512-world count baseline diagnostic; 4096 training config unchanged; " + scope
    report.update(scope=scope + "; original preparation/physics/rewards/diagnostics retained",
                  num_envs=num_envs, configured_training_num_envs=config["num_envs"],
                  diagnostic_override={"num_envs": 512} if num_envs == 512 else {},
                  training_default_modified=False,
                  pipeline_control_steps_requested=0 if collector_only else steps,
                  collection_control_steps_requested=steps + 1,
                  seed_control_steps=1, pipeline_validated=False,
                  collector_device="cuda:0", learner_device=None if collector_only else "cuda:1",
                  physics_hz=1000, control_hz=20)
    started = time.perf_counter()
    scene, task, _ = build_source_from_config(config)
    with torch.cuda.device(0):
        env = PianoTaskEnv(scene, task, num_envs=num_envs, device="cuda:0", **task_kwargs(config))
    report["environment_build_seconds"] = time.perf_counter() - started
    assert env.observation_size == config["observation_size"]
    assert env.action_size == config["action_size"]
    assert env.scene.decimation == 50
    assert env.logic.length == config["replay_prefill"]["episode_steps"]
    assert config["replay_capacity"] >= config["num_envs"] * env.logic.length
    # The default convex-hull task has no finger-clearance guard. The separate
    # key-contact classifier exists only with that guard, not merely because
    # --collision-diagnostics is enabled (PianoScene's explicit constructor rule).
    # Assert the actual contract instead of inventing a mandatory optional field.
    from tsensei_v3_2_pianist.mjlab_env.key_contact_diagnostics import KeyContactDiagnostics
    key_tracker = env.scene.key_contact_diagnostics
    key_tracker_expected = env.collision_diagnostics and env.scene.finger_clearance_guard is not None
    assert (key_tracker is not None) == key_tracker_expected
    report["key_contact_diagnostics"] = dict(enabled=key_tracker_expected,
        prerequisite="collision_diagnostics and finger_clearance_guard",
        reason="enabled_by_current_contract" if key_tracker_expected else
               "not_enabled_by_current_contract: finger_clearance_guard is absent",
        fields=list(KeyContactDiagnostics.COLUMNS) if key_tracker_expected else [],
        physical_key_activation_history_required=True)
    if collector_only:
        # Reuse the normal seeded initialization, including its actor/critic RNG
        # order, on CPU. Only the inference actor is copied to the allocated GPU;
        # no GPU critic, optimizer state, CUDA learner, or updates are created.
        from types import SimpleNamespace
        seed_agent = _agent(config, device="cpu", compile_learner=False, fused_optimizer=False)
        seed = SimpleNamespace(actor=seed_agent.actor.to("cuda:0"),
            device=torch.device("cuda:0"), updates=0,
            action_rng=torch.Generator(device="cuda:0").manual_seed(config["cli"]["seed"] + 1))
        collector = CollectorPolicy(seed, "cuda:0")
        del seed, seed_agent
        report["policy_initialization"] = "Current seeded SAC on CPU; only inference actor retained on CUDA"
    else:
        agent = _agent(config)
        collector = CollectorPolicy(agent, "cuda:0")
    replay = Replay(env.observation_size, env.action_size, num_envs * (steps + 2),
                    config["batch_size"], device="cpu", seed=20261027,
                    action_profile=config["action_profile"])
    rows = []
    with torch.no_grad(), torch.cuda.device(0):
        obs = env.reset()

    def summarize(value):
        if isinstance(value, dict):
            return {name: summarize(item) for name, item in value.items()}
        if not isinstance(value, torch.Tensor):
            return str(value)
        assert bool(torch.isfinite(value).all()), "Nonfinite diagnostic tensor"
        result = dict(shape=list(value.shape), dtype=str(value.dtype), finite=True)
        if value.numel():
            result.update(minimum=value.min().item(), maximum=value.max().item(),
                          sum=value.to(torch.float64).sum().item())
        return result

    def collect(step, actor_version):
        with torch.no_grad(), torch.cuda.device(0):
            previous = obs.clone()
            action = collector.act(obs)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            result = env.step(action)
            torch.cuda.synchronize(0)
            seconds = time.perf_counter() - started
            assert not result.done.any(), "Bounded test unexpectedly reached score end"
            assert not env.scene.data.overflow[:].any()
            info = result.info
            for key in ("collision_physics_substeps", "substep_count"):
                assert key in info and info[key].shape == (num_envs,), key
                assert bool((info[key] == 50).all()), key
            for column, key in enumerate(("self_collision_substeps", "inter_hand_collision_substeps",
                                          "hand_collision_substeps", "collision_physics_substeps")):
                assert torch.equal(info[key], env.scene.self_collision.substep_counts[:, column]), key
            if key_tracker_expected:
                for column, key in enumerate(KeyContactDiagnostics.COLUMNS):
                    assert key in info and info[key].shape == (num_envs,), key
                    assert torch.equal(info[key], key_tracker.substep_counts[:, column]), key
                    assert bool(((info[key] >= 0) & (info[key] <= 50)).all()), key
                assert bool((info["key_contact_physics_substeps"] == 50).all())
            else:
                assert all(key not in info for key in KeyContactDiagnostics.COLUMNS), \
                    "Inactive key-contact classifier unexpectedly supplied diagnostics"
            assert env.exact_score is not None
            assert info["goal_substeps"].shape == (num_envs, 50, 89)
            assert info["finger_keys_substeps"].shape == (num_envs, 50, 10)
            assert bool((info["score_index"] == step).all())
            assert torch.equal(info["goal_substeps"], env.exact_score.goal_history[info["score_index"]])
            assert torch.equal(info["finger_keys_substeps"],
                               env.exact_score.finger_key_history[info["score_index"]])
            sampling = env.scene.reward_sampling
            active = sampling.key_activation_history
            assert active.shape == (num_envs, 50, 88) and active.dtype == torch.bool
            assert sampling.fingertip_history.shape == (num_envs, 50, 10, 3)
            assert sampling.key_target_history.shape == (num_envs, 50, 88, 3)
            goal = info["goal_substeps"][:, :, :88].bool()
            actual_counts = torch.stack(((active & goal).sum((-1, -2)),
                (active & ~goal).sum((-1, -2)), (~active & goal).sum((-1, -2))), -1)
            assert torch.equal(info["substep_key_counts"], actual_counts)
            assert bool((info["substep_phase_counts"][:, :, 0].sum(-1) == 50).all())
            for key in ("reward_raw_terms", "reward_weighted_terms"):
                assert info[key] and all(bool(torch.isfinite(value).all())
                                         for value in info[key].values())
            transition = Transition(previous, action, result.reward, result.discount,
                                    result.observation, result.terminated, result.truncated)
            row = dict(step=step, seconds=seconds, actor_version=actor_version,
                reward_mean=result.reward.mean().item(),
                self_collision_substeps=info["self_collision_substeps"].sum().item(),
                inter_hand_collision_substeps=info["inter_hand_collision_substeps"].sum().item(),
                sampled_physics_substeps=info["substep_count"].sum().item(),
                diagnostics={name: summarize(value) for name, value in info.items()})
            return transition, row

    def commit(batch):
        nonlocal obs
        transition, row = batch
        first = replay.total_inserted
        replay.insert(transition)
        assert all(torch.equal(storage[first:first + num_envs], value.cpu())
                   for storage, value in zip(replay.data, transition))
        obs = transition.next_state
        rows.append(row)
        report["steps"] = rows
        print(f"{num_envs}-world physical batch {row['step']}/{steps}: {row['seconds']:.3f}s", flush=True)

    # Seed from real transitions. Capture runs before a collector worker exists.
    commit(collect(0, collector.version))
    if collector_only:
        started = time.perf_counter()
        for step in range(1, steps + 1):
            commit(collect(step, collector.version))
        collector_seconds = time.perf_counter() - started
        report.update(collector_only_seconds=collector_seconds,
                      bounded_collector_transitions_per_second=steps * num_envs / collector_seconds,
                      collector_timing_scope="Includes inference, physical step, diagnostic audit reductions, "
                                             "CPU insertion and exact-copy checks; excludes seed batch/build",
                      learner_updates=0, cuda_learner_constructed=False,
                      collector_pipeline_state=None, updates_per_vector_step=0)
    else:
        initial_updates = agent.updates
        started = time.perf_counter()
        with torch.cuda.device(1):
            learner = CUDALearner(agent, replay, optimizer_layout=True, matmul_precision="tf32")
        report["learner_setup_seconds"] = time.perf_counter() - started
        report["staging_contract"] = learner.staging_contract
        assert agent.updates == initial_updates
        assert not torch.backends.cuda.matmul.allow_tf32
        updates_per_batch = int(config["gradient_steps_per_vector_step"])

        def update(step):
            with torch.cuda.device(1):
                metrics = learner.update(updates_per_batch)
                assert all(bool(torch.isfinite(value).all()) for value in metrics.values())

        started = time.perf_counter()
        with OneBatchPipeline(snapshot=lambda: collector.sync_from(agent), collect=collect,
                              commit=commit, update=update,
                              learner_version=lambda: agent.updates) as pipeline:
            for step in range(1, steps + 1):
                pipeline.step(step)
            pipeline.flush()
            pipeline_state = pipeline.state()
        report.update(pipeline_seconds=time.perf_counter() - started,
                      collector_pipeline_state=pipeline_state,
                      updates_per_vector_step=updates_per_batch,
                      learner_updates=agent.updates - initial_updates,
                      cuda_learner_constructed=True)
        assert agent.updates - initial_updates == steps * updates_per_batch
        assert pipeline_state["pending_vector_step"] is None
        assert pipeline_state["learner_updated_through_vector_step"] == steps
        assert pipeline_state["max_actor_lag_at_collection_end_updates"] <= updates_per_batch
        report["pipeline_validated"] = True
    with torch.no_grad(), torch.cuda.device(0):
        reset_worlds = [0, num_envs // 2, num_envs - 1]
        ids = torch.tensor(reset_worlds, device="cuda:0")
        last_indices = steps * num_envs + ids.cpu()
        saved_next = replay.data.next_state[last_indices].clone()
        previous_indices = env.logic.index.clone()
        reset_obs = env.reset(ids)
        assert bool(torch.isfinite(reset_obs).all())
        assert torch.equal(saved_next, replay.data.next_state[last_indices]), \
            "Partial reset mutated previously stored next observations"
        assert bool((env.logic.index[ids] == 0).all())
        previous_indices[ids] = 0
        assert torch.equal(env.logic.index, previous_indices), "Partial reset advanced continuing worlds"
        assert not env.logic.done[ids].any()
    sample_device = "cpu" if collector_only else "cuda:1"
    batch = replay.sample(device=sample_device)
    assert all(value.device == torch.device(sample_device) for value in batch)
    assert batch.state.shape == (1024, env.observation_size)
    assert replay.total_inserted == num_envs * (steps + 1)
    median_seconds = statistics.median(row["seconds"] for row in rows[1:])
    report.update(steps=rows, replay_insertions=replay.total_inserted,
                  partial_reset_worlds=reset_worlds, partial_reset_preserved_replay=True,
                  replay_capacity=replay.capacity, replay_bytes=replay.bytes_allocated,
                  storage_contract=replay.storage_contract, sample_device=sample_device,
                  episode_steps=env.logic.length, physics_substeps_per_control=50,
                  physical_step_median_seconds=median_seconds,
                  physical_step_transitions_per_second=num_envs / median_seconds,
                  physical_timing_scope="Synchronized env.step including its original rewards/diagnostics; "
                                        "excludes policy inference, extra audit reductions and replay insertion; "
                                        "seed batch excluded, bounded window only",
                  host_final=_host_memory(), gpu_final=_gpu_memory(torch))


def _worker(kind):
    root = Path(os.environ["PIANIST_CPU_REPLAY_SCALE_DIR"]).resolve()
    path = root / f"{kind}.json"
    assert not path.exists(), "Use a fresh acceptance directory; preserve earlier evidence"
    report = dict(kind=kind, status="running", passed=False)
    torch = None
    started = time.perf_counter()
    try:
        config, torch = _load(root, report)
        _save(path, report)
        (_full_capacity if kind == "full_capacity" else _physical)(config, torch, report)
        report.update(status="passed", passed=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        try:
            report["host_exit"] = _host_memory()
            if torch is not None:
                report["gpu_exit"] = _gpu_memory(torch)
        except BaseException as memory_error:
            report["exit_memory_error"] = repr(memory_error)
        _save(path, report)


def _launch(kind):
    import pytest
    if "PIANIST_CPU_REPLAY_SCALE_DIR" not in os.environ:
        pytest.skip("Opt-in Slurm 4096-world/full-capacity CPU replay acceptance")
    assert os.environ.get("SLURM_JOB_ID"), "Never run this heavy test outside Slurm"
    root = Path(os.environ["PIANIST_CPU_REPLAY_SCALE_DIR"]).resolve()
    assert root.is_dir()
    report_path = root / f"{kind}.json"
    assert not report_path.exists(), "Use a fresh acceptance directory; preserve earlier evidence"
    timeout = int(os.environ.get("PIANIST_CPU_REPLAY_SCALE_TIMEOUT", "7200"))
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), kind],
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        report = json.loads(report_path.read_text()) if report_path.exists() else dict(kind=kind)
        report.update(status="failed", passed=False, error_type="TimeoutExpired",
                      error=f"Acceptance child exceeded {timeout} seconds")
        _save(report_path, report)
        raise
    if result.returncode != 0:
        report = json.loads(report_path.read_text()) if report_path.exists() else dict(kind=kind)
        if report.get("status") != "failed":
            report.update(status="failed", passed=False, error_type="WorkerExited",
                          error="Child did not complete; inspect Slurm stderr and memory telemetry")
        report["worker_returncode"] = result.returncode
        _save(report_path, report)
    assert result.returncode == 0, f"{kind} failed; inspect {report_path} and Slurm log"
    report = json.loads(report_path.read_text())
    assert report["passed"] is True


def test_cpu_replay_full_capacity_and_learner():
    _launch("full_capacity")


def test_4096_physical_collection():
    _launch("physical")


def test_4096_collector_only():
    _launch("collector_only")


def test_collector_baseline_512():
    _launch("collector_baseline_512")


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in (
        "full_capacity", "physical", "collector_only", "collector_baseline_512")
    sys.path.insert(0, str(Path.cwd()))
    _worker(sys.argv[1])
