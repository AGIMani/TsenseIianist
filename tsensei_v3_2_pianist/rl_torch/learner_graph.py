"""Full sequential SAC updates in a CUDA graph, with unchanged update budget.

Replay's randint stays outside the graph so its upper bound follows the growing
buffer exactly. Gathering, original learner RNG draws, losses/backwards, Adam,
Polyak and per-update finite checks are captured without altering their order.
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
        if agent.device.type != 'cuda' or replay.device != agent.device:
            raise ValueError('CUDA learner requires learner and replay on the same CUDA device')
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
        self.indices = torch.empty(replay.batch_size, dtype=torch.int64, device=agent.device)
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

    def _body(self):
        batch = Transition(*(value[self.indices] for value in self.replay.data))
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
                or self._optimizer_storage() != self._optimizer_addresses):
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
