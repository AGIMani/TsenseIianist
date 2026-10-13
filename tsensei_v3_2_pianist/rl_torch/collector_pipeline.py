"""One-vector collector/learner overlap with explicit replay commit barriers.

The scheduler is independent of CUDA so ordering and failures can be tested on
CPU. One worker owns collection; the calling thread owns replay and learning.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

import torch


def execution_layout(collector_device, learner_device=None, pipeline=False):
    """Resolve explicit device indices without allocating CUDA tensors."""
    collector = torch.device(collector_device)
    learner = torch.device(learner_device or collector_device)
    if collector.type != 'cuda' or learner.type != 'cuda':
        raise ValueError('Collector and learner devices must both be CUDA devices')
    collector = torch.device('cuda', 0 if collector.index is None else collector.index)
    learner = torch.device('cuda', 0 if learner.index is None else learner.index)
    if pipeline and collector == learner:
        raise ValueError('--collector-pipeline requires two distinct CUDA devices; set --learner-device')
    return dict(mode=('one_vector_pipeline_v1' if pipeline else
                      'synchronous_split_v1' if collector != learner else 'single_device_v1'),
                collector_device=str(collector), learner_device=str(learner),
                collector_pipeline=bool(pipeline),
                max_pending_vector_batches=1 if pipeline else 0,
                max_actor_lag_vector_batches=1 if pipeline else 0,
                sampling_order_changed=bool(pipeline),
                replay_write_rule='Insert batch t only after all updates for batch t-1 complete',
                boundary_rule='Drain pending updates before evaluation, checkpoints and completion')


class CollectorPolicy:
    """Inference-only actor snapshot with the collector's own action RNG."""

    def __init__(self, agent, device):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.actor = deepcopy(agent.actor).to(self.device).requires_grad_(False)
            self.action_rng = torch.Generator(device=self.device)
            self.action_rng.set_state(agent.action_rng.get_state())
        self.version = agent.updates
        self.sync_from(agent)

    @torch.no_grad()
    def sync_from(self, agent):
        # The caller must have joined the collection worker. Complete the copy
        # before learning can mutate the source parameters on the other GPU.
        torch.cuda.synchronize(agent.device)
        with torch.cuda.device(self.device):
            self.actor.load_state_dict(agent.actor.state_dict())
            torch.cuda.synchronize(self.device)
        self.version = agent.updates
        return self.version

    @torch.no_grad()
    def act(self, obs, *, deterministic=False):
        return (self.actor.deterministic(obs) if deterministic else
                self.actor.sample(obs, generator=self.action_rng)[0])

    def copy_action_rng_to(self, agent):
        # Checkpoints retain the existing action_rng field and warm-restart
        # semantics. The learner itself never samples actions in split mode.
        agent.action_rng.set_state(self.action_rng.get_state())


class OneBatchPipeline:
    """Collect t concurrently with updates for t-1, then commit t.

    ``snapshot()`` finishes copying the actor before either operation starts.
    ``collect(step, version)`` is the only callback run in the worker. All other
    callbacks run in the caller thread. ``commit(batch)`` must retain terminal
    observations before resetting worlds. No second collection is submitted
    until the first has joined and committed, so there is at most one in flight.
    """

    def __init__(self, *, snapshot, collect, update, commit, learner_version=None):
        self.snapshot = snapshot
        self.collect = collect
        self.update = update
        self.commit = commit
        self.learner_version = learner_version
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='piano-collector')
        self.pending = None
        self.collected_steps = 0
        self.updated_through_step = 0
        self.overlapped_batches = 0
        self.collection_seconds = 0.
        self.update_seconds = 0.
        self.snapshot_seconds = 0.
        self.commit_seconds = 0.
        self.step_wall_seconds = 0.
        self.host_overlap_seconds = 0.
        self.last_update_interval = None
        self.max_actor_lag_updates = 0
        self.last_actor_lag_updates = 0
        self.actor_version = None
        self.failed = False
        self.closed = False

    def _require_open(self):
        if self.closed or self.failed:
            raise RuntimeError('Collector pipeline is closed or failed')

    def _collect(self, step, version):
        started = time.perf_counter()
        result = self.collect(step, version)
        return result, started, time.perf_counter()

    def flush(self):
        """Apply the pending block exactly once, propagating any learner error."""
        self._require_open()
        if self.pending is not None:
            started = time.perf_counter()
            try:
                self.update(self.pending)
            except BaseException:
                self.failed = True
                raise
            finished = time.perf_counter()
            self.update_seconds += finished - started
            self.last_update_interval = (started, finished)
            self.updated_through_step = self.pending
            self.pending = None

    def step(self, step):
        self._require_open()
        if step != self.collected_steps + 1:
            raise ValueError('Collector vector steps must be consecutive and start at one')
        step_started = time.perf_counter()
        try:
            snapshot_started = time.perf_counter()
            self.actor_version = self.snapshot()
            self.snapshot_seconds += time.perf_counter() - snapshot_started
            future = self.executor.submit(self._collect, step, self.actor_version)
            self.last_update_interval = None
            if self.pending is not None:
                self.overlapped_batches += 1
            try:
                self.flush()
            except BaseException:
                # Join the sole worker before an exception escapes. A failed
                # learner never permits this newly collected batch into replay.
                future.cancel()
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.closed = True
                raise
            result, collect_started, collect_finished = future.result()
            self.collection_seconds += collect_finished - collect_started
            if self.last_update_interval is not None:
                update_started, update_finished = self.last_update_interval
                self.host_overlap_seconds += max(
                    0., min(collect_finished, update_finished) - max(collect_started, update_started))
            if self.learner_version is not None:
                self.last_actor_lag_updates = self.learner_version() - self.actor_version
                self.max_actor_lag_updates = max(self.max_actor_lag_updates, self.last_actor_lag_updates)
            commit_started = time.perf_counter()
            self.commit(result)
            self.commit_seconds += time.perf_counter() - commit_started
            self.step_wall_seconds += time.perf_counter() - step_started
            self.collected_steps = step
            self.pending = step
            return result
        except BaseException:
            self.failed = True
            raise

    def state(self):
        return dict(collected_vector_steps=self.collected_steps,
                    learner_updated_through_vector_step=self.updated_through_step,
                    pending_vector_step=self.pending,
                    collector_actor_update_version=self.actor_version,
                    overlapped_vector_batches=self.overlapped_batches,
                    collection_seconds=self.collection_seconds,
                    update_seconds=self.update_seconds,
                    actor_snapshot_seconds=self.snapshot_seconds,
                    replay_commit_reset_seconds=self.commit_seconds,
                    pipeline_step_wall_seconds=self.step_wall_seconds,
                    host_overlap_seconds=self.host_overlap_seconds,
                    actor_lag_at_collection_end_updates=self.last_actor_lag_updates,
                    max_actor_lag_at_collection_end_updates=self.max_actor_lag_updates,
                    timing_note='Host callback overlap, not measured CUDA kernel overlap; step wall time excludes external boundary flushes')

    def close(self):
        if self.closed:
            return
        try:
            if not self.failed:
                self.flush()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.failed = True
        self.close()
