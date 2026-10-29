"""Batched CPU/CUDA ring buffer. Store the ending observation before resetting.

The CPU pool uses ordinary pageable memory. Only a CUDA learner's bounded
minibatch staging buffers are pinned; a large replay must never pin the pool.
"""
from typing import NamedTuple

import torch


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_state: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class Replay:
    def __init__(self, state_dim, action_dim, capacity, batch_size, *, device='cuda:0', seed=0, action_profile='legacy'):
        for n in (state_dim, action_dim, capacity, batch_size):
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError('Replay dimensions, capacity and batch_size must be positive integers')
        if capacity < batch_size:
            raise ValueError('Replay capacity must be at least batch_size')
        self.device = torch.empty(0, device=device).device
        from ..mjlab_env.action_mapping import policy_bounds
        low, high = policy_bounds(action_profile, action_dim)
        self.action_low = torch.tensor(low, device=self.device)
        self.action_high = torch.tensor(high, device=self.device)
        self.capacity, self.batch_size = capacity, batch_size
        self.size = self.pointer = self.total_inserted = 0
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        def alloc(shape, dtype=torch.float32):
            return torch.empty((capacity, *shape), device=self.device, dtype=dtype)
        self.data = Transition(alloc((state_dim,)), alloc((action_dim,)), alloc(()),
                               alloc(()), alloc((state_dim,)), alloc((), torch.bool), alloc((), torch.bool))

    @property
    def bytes_allocated(self):
        return sum(x.numel() * x.element_size() for x in self.data)

    @property
    def storage_contract(self):
        return dict(device=str(self.device), storage_dtype='float32_with_bool_end_flags',
                    bytes_allocated=self.bytes_allocated,
                    pool_pinned=any(value.is_pinned() for value in self.data),
                    sampling='uniform_with_replacement',
                    sampling_rng_device=str(self.generator.device),
                    sampling_rng_backend=('pytorch_cpu' if self.device.type == 'cpu' else 'pytorch_cuda'),
                    cross_device_rng_bitwise_equivalent=False)

    def __len__(self):
        return self.size

    def is_ready(self):
        return self.size >= self.batch_size

    @torch.no_grad()
    def insert(self, batch):
        if not isinstance(batch, Transition):
            raise TypeError('Replay requires a Transition with explicit end flags and environment discount')
        converted = []
        n = None
        for storage, value in zip(self.data, batch):
            value = torch.as_tensor(value, device=self.device)
            if value.ndim != storage.ndim or value.shape[1:] != storage.shape[1:]:
                raise ValueError('Invalid batched transition shape')
            if n is None:
                n = value.shape[0]
            if value.shape[0] != n or not torch.isfinite(value).all():
                raise ValueError('Inconsistent batch lengths or nonfinite transition')
            if storage.dtype == torch.bool and not ((value == 0) | (value == 1)).all():
                raise ValueError('End flags must be boolean')
            value = value.to(storage.dtype)
            if not torch.isfinite(value).all():
                raise ValueError('Transition overflow after conversion to replay dtype')
            converted.append(value)
        batch = Transition(*converted)
        if ((batch.discount < 0) | (batch.discount > 1)).any():
            raise ValueError('Environment discount must be in [0, 1]')
        if ((batch.action < self.action_low-1e-6) | (batch.action > self.action_high+1e-6)).any():
            raise ValueError('Replay action is outside configured policy bounds')
        if (batch.terminated & batch.truncated).any():
            raise ValueError('A transition cannot be both terminated and truncated')
        if n == 0:
            return
        # When insertion exceeds capacity retain the final capacity rows, with
        # exactly the ring positions that sequential insertion would produce.
        offset = max(0, n - self.capacity)
        indices = (torch.arange(offset, n, device=self.device) + self.pointer) % self.capacity
        for storage, value in zip(self.data, batch):
            storage[indices] = value[offset:]
        self.pointer = (self.pointer + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_inserted += n

    def sample(self, *, device=None):
        if not self.is_ready():
            raise ValueError('Replay is not ready')
        indices = torch.randint(self.size, (self.batch_size,), generator=self.generator, device=self.device)
        # Advanced indexing owns memory; wrapping/inserting cannot mutate this batch.
        batch = Transition(*(x[indices] for x in self.data))
        if device is not None:
            # The non-graph path intentionally uses completed transfers. The
            # graph path owns reusable pinned buffers and explicit copy events.
            batch = Transition(*(value.to(device=device) for value in batch))
        return batch
