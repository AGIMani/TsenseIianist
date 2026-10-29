"""Full sequential SAC updates in a CUDA graph, with unchanged update budget.

Replay's randint stays outside the graph so its upper bound follows the growing
buffer exactly. CUDA replay gathering, original learner RNG draws,
losses/backwards, Adam, Polyak and per-update finite checks are captured without
altering their order. CPU replay gathers into two bounded pinned host buffers
and copies to one fixed CUDA batch before replaying the same learner graph.
"""
from copy import deepcopy

import torch

from .replay import Transition


class CUDALearner:
    """Capture a ready SAC/replay pair; call update(count) per vector step.

    Construct after checkpoint loading and once replay is ready. Do not load a
    checkpoint or replace either object's tensor storage while this object lives;
    discard it and construct a fresh graph instead. Returned metric tensors are
    views of the most recent graph execution and are overwritten by later calls.
    """

    def __init__(self, agent, replay, *, optimizer_layout=False, matmul_precision='ieee'):
        if agent.device.type != 'cuda' or replay.device not in (agent.device, torch.device('cpu')):
            raise ValueError('CUDA learner requires replay on CPU or the learner CUDA device')
        if not agent.fused_optimizer:
            raise ValueError('CUDA learner requires the existing fused Adam execution mode')
        if not replay.is_ready():
            raise ValueError('Replay is not ready for CUDA learner capture')
        if (replay.data.state.shape[1] != agent.obs_size
                or replay.data.action.shape[1] != agent.action_size):
            raise ValueError('Replay dimensions differ from learner')
        if matmul_precision not in ('ieee', 'tf32'):
            raise ValueError('Unknown learner matmul precision')
        self.matmul_precision = matmul_precision
        self.agent, self.replay = agent, replay
        self.optimizer_layout = optimizer_layout
        self.indices = torch.empty(replay.batch_size, dtype=torch.int64, device=replay.device)
        self._cpu_replay = replay.device.type == 'cpu'
        self._host_batches = []
        self._host_copy_events = []
        self._host_slot = 0
        self._device_batch = None
        if self._cpu_replay:
            def batch_on(device, *, pinned=False):
                return Transition(*(torch.empty((replay.batch_size, *value.shape[1:]),
                    dtype=value.dtype, device=device, pin_memory=pinned)
                    for value in replay.data))
            self._host_batches = [batch_on('cpu', pinned=True) for _ in range(2)]
            self._host_copy_events = [None, None]
            self._device_batch = batch_on(agent.device)
        self.finite = torch.ones((), dtype=torch.bool, device=agent.device)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(agent.update_rng)
        self._failed = False
        self._optimizer_step = None
        # The training loop drains collectors before this first capture. The
        # captured GEMM kernels retain their precision when replayed; restore
        # the process flag before any collector/evaluator can run again.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = matmul_precision == 'tf32'
            self._capture()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def _sample_indices(self):
        # Same call, shape, generator and integer distribution as Replay.sample.
        # Using out only removes allocation; it does not change Philox draws.
        torch.randint(self.replay.size, (self.replay.batch_size,),
                      generator=self.replay.generator, device=self.replay.device,
                      out=self.indices)
        if self._cpu_replay:
            slot = self._host_slot
            host_batch = self._host_batches[slot]
            copied = self._host_copy_events[slot]
            if copied is not None:
                # Protect the pinned source until its previous H2D completes;
                # never synchronize unrelated GPU work or the whole device.
                copied.synchronize()
            for storage, staging in zip(self.replay.data, host_batch):
                torch.index_select(storage, 0, self.indices, out=staging)
            with torch.cuda.device(self.agent.device):
                stream = torch.cuda.current_stream(self.agent.device)
                # One fixed device batch means copies and graph execution must
                # share a stream: previous graph -> H2D -> next graph. CPU
                # gathering for the other host slot may overlap the GPU work.
                for destination, source in zip(self._device_batch, host_batch):
                    destination.copy_(source, non_blocking=True)
                if copied is None:
                    copied = torch.cuda.Event()
                    self._host_copy_events[slot] = copied
                copied.record(stream)
            self._host_slot = (slot + 1) % len(self._host_batches)

    def _body(self):
        batch = (self._device_batch if self._cpu_replay else
                 Transition(*(value[self.indices] for value in self.replay.data)))
        return self.agent.update(batch, _finite_accumulator=self.finite, _fused_polyak=True,
                                 _optimizer_step=self._optimizer_step)

    def _capture(self):
        agent = self.agent
        parameters = (list(agent.actor.parameters()) + list(agent.critic.parameters())
                      + list(agent.target_critic.parameters()) + [agent.log_temperature])
        saved_parameters = [value.detach().clone() for value in parameters]
        optimizers = (agent.actor_optimizer, agent.critic_optimizer, agent.temp_optimizer)
        saved_optimizers = [{parameter: deepcopy(values) for parameter, values in optimizer.state.items()}
                            for optimizer in optimizers]
        saved_update_rng = agent.update_rng.get_state()
        saved_replay_rng = self.replay.generator.get_state()
        saved_updates = agent.updates
        stream = torch.cuda.Stream(device=agent.device)
        stream.wait_stream(torch.cuda.current_stream(agent.device))
        try:
            # Compile objectives/backwards and initialize Adam on the same side
            # stream before capture. These are dry updates: every mutable state
            # is restored in place so graph tensor addresses remain valid.
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._sample_indices()
                    self._body()
            torch.cuda.current_stream(agent.device).wait_stream(stream)
            torch.cuda.synchronize(agent.device)
            if self.optimizer_layout:
                from .adam_execution import AdamExecution
                self._optimizer_step = AdamExecution()
                # Load the extension kernel before capture; CUDA lazy loading
                # is not assumed to be safe on every supported driver.
                with torch.cuda.stream(stream):
                    self._sample_indices()
                    self._body()
                torch.cuda.current_stream(agent.device).wait_stream(stream)
                torch.cuda.synchronize(agent.device)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['capturable'] = True
            with torch.cuda.graph(self.graph, stream=stream):
                self.metrics = self._body()
        finally:
            torch.cuda.synchronize(agent.device)
            with torch.no_grad():
                for parameter, saved in zip(parameters, saved_parameters):
                    parameter.copy_(saved)
                for optimizer, saved in zip(optimizers, saved_optimizers):
                    for parameter, values in optimizer.state.items():
                        original = saved.get(parameter, {})
                        for name, value in values.items():
                            if isinstance(value, torch.Tensor):
                                if name in original:
                                    value.copy_(original[name])
                                else:
                                    # Adam's initially absent moments/step are
                                    # equivalent to allocated zeros before step 1.
                                    value.zero_()
                            elif name in original:
                                values[name] = deepcopy(original[name])
                self.finite.fill_(True)
            agent.update_rng.set_state(saved_update_rng)
            self.replay.generator.set_state(saved_replay_rng)
            agent.updates = saved_updates
        self._parameters = parameters
        self._parameter_addresses = tuple(value.data_ptr() for value in parameters)
        self._replay_addresses = tuple(value.data_ptr() for value in self.replay.data)
        self._optimizer_addresses = self._optimizer_storage()
        self._staging_storage_addresses = self._staging_storage()

    def _staging_storage(self):
        return (tuple(value.data_ptr() for batch in self._host_batches for value in batch)
                + (() if self._device_batch is None else
                   tuple(value.data_ptr() for value in self._device_batch)))

    @property
    def staging_contract(self):
        def size(batch):
            return sum(value.numel() * value.element_size() for value in batch)
        return dict(replay_device=str(self.replay.device),
                    host_buffer_count=len(self._host_batches),
                    host_pinned_bytes=sum(size(batch) for batch in self._host_batches),
                    device_batch_bytes=0 if self._device_batch is None else size(self._device_batch),
                    transfer=('pinned_double_host_single_cuda_batch_same_stream_v1'
                              if self._cpu_replay else 'cuda_replay_graph_gather_v1'),
                    host_reuse_barrier='per_buffer_h2d_event' if self._cpu_replay else None,
                    h2d_overlaps_learner=False,
                    cpu_gather_may_overlap_learner=self._cpu_replay)

    def _optimizer_storage(self):
        return tuple(value.data_ptr()
                     for optimizer in (self.agent.actor_optimizer, self.agent.critic_optimizer,
                                       self.agent.temp_optimizer)
                     for state in optimizer.state.values()
                     for value in state.values() if isinstance(value, torch.Tensor))

    def update(self, count):
        """Run exactly count sequential updates, checking every result's health.

        A nonfinite result aborts at this vector block's boundary, before another
        environment action, evaluation or save. The failure is sticky even when
        a subsequent metric would happen to become finite.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('CUDA learner update count must be a positive integer')
        if self._failed:
            raise FloatingPointError('CUDA learner previously failed; abort training')
        if not self.replay.is_ready():
            raise ValueError('Replay is not ready')
        if (tuple(value.data_ptr() for value in self._parameters) != self._parameter_addresses
                or tuple(value.data_ptr() for value in self.replay.data) != self._replay_addresses
                or self._optimizer_storage() != self._optimizer_addresses
                or self._staging_storage() != self._staging_storage_addresses):
            raise ValueError('CUDA learner storage changed; construct a new graph')
        self.finite.fill_(True)
        for _ in range(count):
            self._sample_indices()
            self.graph.replay()
        self.agent.updates += count
        if not bool(self.finite):
            self._failed = True
            raise FloatingPointError('Nonfinite SAC update inside CUDA learner block; abort training')
        return self.metrics
