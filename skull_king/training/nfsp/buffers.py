"""NFSP replay buffers.

RLBuffer  — (enc, mask, action, mc_return) for Q-net training.
SLBuffer  — (enc, mask, action) for average-strategy supervised learning.
"""
from __future__ import annotations

import numpy as np


class RLBuffer:
    """Circular buffer storing MC-return transitions for Q-learning."""

    def __init__(self, capacity: int, pbs_size: int, action_size: int, seed: int = 0):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self.encs = np.zeros((capacity, pbs_size), dtype=np.float32)
        self.masks = np.zeros((capacity, action_size), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add_batch(
        self,
        encs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        n = len(encs)
        idx = np.arange(self._ptr, self._ptr + n) % self._cap
        self.encs[idx] = encs
        self.masks[idx] = masks
        self.actions[idx] = actions
        self.returns[idx] = returns
        self._ptr = (self._ptr + n) % self._cap
        self._size = min(self._size + n, self._cap)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.encs[idx],
            self.masks[idx],
            self.actions[idx],
            self.returns[idx],
        )

    def __len__(self) -> int:
        return self._size


class SLBuffer:
    """Circular buffer storing (enc, mask, action) for supervised learning."""

    def __init__(self, capacity: int, pbs_size: int, action_size: int, seed: int = 0):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self.encs = np.zeros((capacity, pbs_size), dtype=np.float32)
        self.masks = np.zeros((capacity, action_size), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)

    def add_batch(
        self,
        encs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        n = len(encs)
        idx = np.arange(self._ptr, self._ptr + n) % self._cap
        self.encs[idx] = encs
        self.masks[idx] = masks
        self.actions[idx] = actions
        self._ptr = (self._ptr + n) % self._cap
        self._size = min(self._size + n, self._cap)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self.encs[idx], self.masks[idx], self.actions[idx]

    def __len__(self) -> int:
        return self._size
