"""AlphaZero replay buffer.

Stores (encoded_state, action_mask, policy_target, value_target) tuples.
Circular FIFO. ``policy_target`` is a normalized vector over the full
action space, with mass concentrated on the legal actions visited by
MCTS at the root.
"""
from __future__ import annotations

import numpy as np


class AZReplayBuffer:
    """Vectorized circular replay buffer for AlphaZero self-play data."""

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
        self.policy  = np.zeros((capacity, action_size), dtype=np.float32)
        self.values  = np.zeros(capacity, dtype=np.float32)

    def add_batch(
        self,
        encs: np.ndarray,
        masks: np.ndarray,
        policy: np.ndarray,
        values: np.ndarray,
    ) -> None:
        n = len(encs)
        idx = (self._ptr + np.arange(n)) % self._cap
        self.encs[idx]    = encs
        self.masks[idx]   = masks
        self.policy[idx]  = policy
        self.values[idx]  = values
        self._ptr = (self._ptr + n) % self._cap
        self._size = min(self._size + n, self._cap)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.encs[idx],
            self.masks[idx],
            self.policy[idx],
            self.values[idx],
        )

    def __len__(self) -> int:
        return self._size
