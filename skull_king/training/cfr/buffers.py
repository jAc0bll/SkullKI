"""Circular replay buffers for Deep CFR advantage and strategy samples."""
from __future__ import annotations

import numpy as np

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE


class AdvantageBuffer:
    """Circular buffer of (obs, mask, adv_target, action) tuples.

    The advantage target is non-zero only at the taken action index —
    training uses MSE loss restricted to that index.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._obs = np.empty((capacity, OBS_SIZE), dtype=np.float32)
        self._masks = np.empty((capacity, ACTION_SPACE_SIZE), dtype=bool)
        self._targets = np.empty((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        adv_target: np.ndarray,
        action: int,
    ) -> None:
        idx = self._pos % self.capacity
        self._obs[idx] = obs
        self._masks[idx] = mask
        self._targets[idx] = adv_target
        self._actions[idx] = int(action)
        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, samples: list[tuple]) -> None:
        for obs, mask, adv_target, action in samples:
            self.add(obs, mask, adv_target, action)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self._size)
        idx = np.random.choice(self._size, n, replace=False)
        return (
            self._obs[idx],
            self._masks[idx],
            self._targets[idx],
            self._actions[idx],
        )

    def __len__(self) -> int:
        return self._size


class StrategyBuffer:
    """Circular buffer of (obs, mask, strategy) tuples.

    Trained via cross-entropy to imitate the regret-matched strategy
    at each visited information set — this is the Nash approximation.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._obs = np.empty((capacity, OBS_SIZE), dtype=np.float32)
        self._masks = np.empty((capacity, ACTION_SPACE_SIZE), dtype=bool)
        self._strategies = np.empty((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        strategy: np.ndarray,
    ) -> None:
        idx = self._pos % self.capacity
        self._obs[idx] = obs
        self._masks[idx] = mask
        self._strategies[idx] = strategy
        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, samples: list[tuple]) -> None:
        for obs, mask, strategy in samples:
            self.add(obs, mask, strategy)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self._size)
        idx = np.random.choice(self._size, n, replace=False)
        return self._obs[idx], self._masks[idx], self._strategies[idx]

    def __len__(self) -> int:
        return self._size
