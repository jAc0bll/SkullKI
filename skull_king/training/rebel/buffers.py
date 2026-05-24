"""Replay buffers for ReBeL value and policy network training."""
from __future__ import annotations

import numpy as np


class ValueBuffer:
    """Circular buffer storing (pbs_enc, values) pairs for value net training."""

    def __init__(self, capacity: int, pbs_size: int, n_players: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.pbs_size = pbs_size
        self.n_players = n_players
        self._obs = np.zeros((capacity, pbs_size), dtype=np.float32)
        self._vals = np.zeros((capacity, n_players), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, pbs_enc: np.ndarray, values: np.ndarray) -> None:
        self._obs[self._ptr] = pbs_enc
        self._vals[self._ptr] = values
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, pbs_encs: np.ndarray, values: np.ndarray) -> None:
        for enc, val in zip(pbs_encs, values):
            self.add(enc, val)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self._obs[idx], self._vals[idx]

    def __len__(self) -> int:
        return self._size


class PolicyBuffer:
    """Circular buffer storing (pbs_enc, action_mask, strategy) triples."""

    def __init__(
        self,
        capacity: int,
        pbs_size: int,
        n_actions: int,
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self.pbs_size = pbs_size
        self.n_actions = n_actions
        self._obs = np.zeros((capacity, pbs_size), dtype=np.float32)
        self._masks = np.zeros((capacity, n_actions), dtype=bool)
        self._strats = np.zeros((capacity, n_actions), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        pbs_enc: np.ndarray,
        action_mask: np.ndarray,
        strategy: np.ndarray,
    ) -> None:
        self._obs[self._ptr] = pbs_enc
        self._masks[self._ptr] = action_mask
        self._strats[self._ptr] = strategy
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        pbs_encs: np.ndarray,
        masks: np.ndarray,
        strategies: np.ndarray,
    ) -> None:
        for enc, mask, strat in zip(pbs_encs, masks, strategies):
            self.add(enc, mask, strat)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self._obs[idx], self._masks[idx], self._strats[idx]

    def __len__(self) -> int:
        return self._size
