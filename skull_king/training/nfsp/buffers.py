"""NFSP replay buffers.

RLBuffer  — (enc, mask, action, mc_return) circular buffer for Q-net.
SLBuffer  — (enc, mask, action) reservoir-sampled buffer for SL of the
            average strategy. Reservoir sampling (Vitter 1985) keeps a
            uniform random sample across all historical BR transitions,
            which is what NFSP's average-strategy objective requires.
            A circular FIFO buffer would only retain the most recent
            BR policies and cause the avg-net to track a moving target
            instead of the true average.
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
    """Reservoir-sampled buffer for the NFSP average-strategy SL target.

    Vitter's Algorithm R, vectorized for batch inserts:
      - While not full, append items.
      - Once full, for each new item draw j ~ Uniform[0, count); if j < cap
        replace slot j. This gives a uniform sample over the full stream.
    """

    def __init__(self, capacity: int, pbs_size: int, action_size: int, seed: int = 0):
        self._cap = capacity
        self._size = 0
        self._count = 0   # total items seen across all time
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

        # Phase 1: fill until capacity (no sampling decisions yet)
        fill = min(n, self._cap - self._size)
        if fill > 0:
            self.encs[self._size:self._size + fill] = encs[:fill]
            self.masks[self._size:self._size + fill] = masks[:fill]
            self.actions[self._size:self._size + fill] = actions[:fill]
            self._size += fill
            self._count += fill

        # Phase 2: reservoir replacement for the remainder
        remaining = n - fill
        if remaining > 0:
            # For each i in remaining: draw j ~ U[0, count+i+1); if j < cap, replace slot j.
            counts = self._count + np.arange(1, remaining + 1, dtype=np.int64)
            js = self._rng.integers(0, counts)  # j_i in [0, count_after_i)
            keep = js < self._cap
            slots = js[keep]
            src   = fill + np.where(keep)[0]
            # Resolve duplicate target slots — last writer wins (stochastic anyway)
            self.encs[slots]    = encs[src]
            self.masks[slots]   = masks[src]
            self.actions[slots] = actions[src]
            self._count += remaining

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self.encs[idx], self.masks[idx], self.actions[idx]

    def __len__(self) -> int:
        return self._size
