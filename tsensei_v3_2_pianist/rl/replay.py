"""Explicit single-transition replay; preserves upstream uniform sampling.

CPU storage for the first SAC baseline. A batched GPU replay is a separate stage.
"""
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_state: np.ndarray


class Buffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, seed=0):
        if min(state_dim, action_dim, max_size, batch_size) < 1 or batch_size > max_size:
            raise ValueError("Positive dimensions and capacity >= batch_size required")
        self.batch_size = batch_size
        self.capacity = max_size
        self._rng = np.random.RandomState(seed)
        self._data = Transition(
            np.empty((max_size, state_dim), np.float32),
            np.empty((max_size, action_dim), np.float32),
            np.empty(max_size, np.float32),
            np.empty(max_size, np.float32),
            np.empty((max_size, state_dim), np.float32),
        )
        self._size = self._ptr = 0

    def insert(self, state, action, reward, discount, next_state):
        """Insert the terminal observation BEFORE resetting the environment."""
        values = Transition(state, action, reward, discount, next_state)
        converted = []
        for storage, value in zip(self._data, values):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != storage.shape[1:] or not np.isfinite(array).all():
                raise ValueError("Invalid transition shape or non-finite value")
            converted.append(array)
        if not 0 <= float(converted[3]) <= 1:
            raise ValueError("Environment discount must be in [0, 1]")
        for storage, value in zip(self._data, converted):
            storage[self._ptr] = value
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self):
        if not self.is_ready():
            raise ValueError("Replay is not ready")
        indices = self._rng.randint(0, self._size, size=self.batch_size)
        return Transition(*(field[indices] for field in self._data))

    def is_ready(self):
        return self._size >= self.batch_size

    def __len__(self):
        return self._size
