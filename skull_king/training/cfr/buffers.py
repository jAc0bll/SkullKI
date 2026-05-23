"""Circular replay buffers for Deep CFR advantage and strategy samples."""
from __future__ import annotations

from typing import Optional

import numpy as np

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE


def _slice_write(
    dst: np.ndarray, src: np.ndarray, pos: int, capacity: int
) -> None:
    """Write src into dst as a circular buffer starting at pos.

    Handles wraparound with at most two slice assignments — no Python loop.
    """
    n = src.shape[0]
    start = pos % capacity
    end = start + n
    if end <= capacity:
        dst[start:end] = src
    else:
        first = capacity - start
        dst[start:] = src[:first]
        dst[: n - first] = src[first:]


class AdvantageBuffer:
    """Circular buffer of (obs, mask, adv_target, action) tuples.

    The advantage target is non-zero only at the taken action index —
    training uses MSE loss restricted to that index.
    """

    def __init__(
        self,
        capacity: int,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self._obs = np.empty((capacity, obs_size), dtype=np.float32)
        self._masks = np.empty((capacity, action_size), dtype=bool)
        self._targets = np.empty((capacity, action_size), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._pos = 0
        self._size = 0
        # Per-buffer RNG so sampling is reproducible across runs given env_seed.
        self._rng = np.random.default_rng(seed)

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
        """Legacy per-sample loop — kept for callers that have lists of tuples."""
        for obs, mask, adv_target, action in samples:
            self.add(obs, mask, adv_target, action)

    def add_batch_vec(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Vectorised insert: write N samples with at most two slice assigns.

        Roughly 50× faster than ``add_batch`` for batches of a few thousand
        samples, since the Python loop overhead dominates per-element copies
        for tiny ndarrays.
        """
        n = obs.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Edge case: an unexpectedly huge batch — keep the tail only.
            tail = obs[-self.capacity :]
            self._obs[:] = tail
            self._masks[:] = masks[-self.capacity :]
            self._targets[:] = targets[-self.capacity :]
            self._actions[:] = actions[-self.capacity :]
            self._pos = 0
            self._size = self.capacity
            return
        _slice_write(self._obs, obs, self._pos, self.capacity)
        _slice_write(self._masks, masks, self._pos, self.capacity)
        _slice_write(self._targets, targets, self._pos, self.capacity)
        _slice_write(self._actions, actions, self._pos, self.capacity)
        self._pos += n
        self._size = min(self._size + n, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self._size)
        idx = self._rng.integers(0, self._size, size=n)  # O(n), reproducible
        return (
            self._obs[idx],
            self._masks[idx],
            self._targets[idx],
            self._actions[idx],
        )

    def clear(self) -> None:
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size


class StrategyBuffer:
    """Circular buffer of (obs, mask, strategy) tuples.

    Trained via cross-entropy to imitate the regret-matched strategy
    at each visited information set — this is the Nash approximation.
    """

    def __init__(
        self,
        capacity: int,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self._obs = np.empty((capacity, obs_size), dtype=np.float32)
        self._masks = np.empty((capacity, action_size), dtype=bool)
        self._strategies = np.empty((capacity, action_size), dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

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
        """Legacy per-sample loop — kept for callers that have lists of tuples."""
        for obs, mask, strategy in samples:
            self.add(obs, mask, strategy)

    def add_batch_vec(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        strategies: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            self._obs[:] = obs[-self.capacity :]
            self._masks[:] = masks[-self.capacity :]
            self._strategies[:] = strategies[-self.capacity :]
            self._pos = 0
            self._size = self.capacity
            return
        _slice_write(self._obs, obs, self._pos, self.capacity)
        _slice_write(self._masks, masks, self._pos, self.capacity)
        _slice_write(self._strategies, strategies, self._pos, self.capacity)
        self._pos += n
        self._size = min(self._size + n, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self._size)
        idx = self._rng.integers(0, self._size, size=n)
        return self._obs[idx], self._masks[idx], self._strategies[idx]

    def __len__(self) -> int:
        return self._size
