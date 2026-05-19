"""Reservoir buffer for NFSP average-strategy supervised learning.

Algorithm R (Vitter 1985): after the buffer fills, each new item replaces
a uniformly chosen existing item with probability capacity/count.  The
buffer is therefore always a uniform random sample of all items ever seen.
"""
from __future__ import annotations

import numpy as np


class ReservoirBuffer:
    """Fixed-capacity uniform random sample of (obs, mask, action) tuples."""

    def __init__(self, capacity: int, obs_size: int, action_size: int, seed: int = 0) -> None:
        self.capacity = capacity
        self._obs = np.empty((capacity, obs_size), dtype=np.float32)
        self._masks = np.empty((capacity, action_size), dtype=bool)
        self._acts = np.empty(capacity, dtype=np.int64)
        self._size = 0    # items currently in buffer
        self._count = 0   # total items ever seen
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def add(self, obs: np.ndarray, mask: np.ndarray, action: int) -> None:
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            idx = int(self._rng.integers(0, self._count + 1))
            if idx >= self.capacity:
                self._count += 1
                return
        self._obs[idx] = obs
        self._masks[idx] = mask
        self._acts[idx] = int(action)
        self._count += 1

    def add_batch(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        for i in range(len(obs)):
            self.add(obs[i], masks[i], int(actions[i]))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self._size)
        idx = self._rng.choice(self._size, n, replace=False)
        return self._obs[idx], self._masks[idx], self._acts[idx]

    def __len__(self) -> int:
        return self._size
