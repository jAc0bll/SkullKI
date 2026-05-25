"""DouZero replay buffer — circular FIFO of (state, mask, action, MC-return).

Pure Monte-Carlo Q-learning: targets are full-episode returns, so there
is no temporal-difference bootstrapping. Old transitions become stale
only because the data distribution shifts (better policy → different
states visited), not because of TD-target drift. A simple circular FIFO
is the standard DouZero choice and gives the most up-to-date data given
the trade-off between recency and diversity.
"""
from __future__ import annotations

import numpy as np


class MCReplayBuffer:
    """Circular buffer keyed by global insertion order.

    `add_batch` is vectorized — single numpy fancy-index write into all
    four arrays, no per-sample Python overhead.
    """

    def __init__(
        self,
        capacity: int,
        pbs_size: int,
        action_size: int,
        seed: int = 0,
    ) -> None:
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self.encs    = np.zeros((capacity, pbs_size), dtype=np.float32)
        self.masks   = np.zeros((capacity, action_size), dtype=bool)
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
        idx = (self._ptr + np.arange(n)) % self._cap
        self.encs[idx]    = encs
        self.masks[idx]   = masks
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
