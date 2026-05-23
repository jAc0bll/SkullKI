"""Python wrapper around the skull_king_engine C extension.

Provides a drop-in replacement for traversal.worker_task / traverse that uses
the full C game engine (deal, bid, play, resolve, score) and C MLP inference
(optionally AVX2-accelerated), giving ~3-4x speedup over the Python traversal.

Usage in trainer.py
-------------------
    from skull_king.cfr_engine import CEngine

    engine = CEngine(n_players=4, heuristic_frac=0.4)
    engine.load_adv_weights(adv_state_dict)

    # Drop-in for traverse():
    result = engine.traverse(traverser=0, seed=42)
    adv_obs, adv_masks, adv_targets, adv_actions, strat_obs, strat_masks, strat_strats = result

Fallback
--------
If the C extension is not compiled, CEngine.available is False and you should
fall back to the pure-Python traversal in skull_king.training.cfr.traversal.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from skull_king._core.skull_king_engine import (
        set_adv_weights as _c_set_weights,
        traverse        as _c_traverse,
    )
    _C_ENGINE_AVAILABLE = True
except ImportError:
    _C_ENGINE_AVAILABLE = False

try:
    from skull_king._core.skull_king_engine import (
        set_bid_adv_weights  as _c_set_bid_weights,
        set_play_adv_weights as _c_set_play_weights,
        traverse_split       as _c_traverse_split,
    )
    _C_SPLIT_AVAILABLE = True
except ImportError:
    _C_SPLIT_AVAILABLE = False


class CEngine:
    """Wraps the skull_king_engine C extension for CFR traversal."""

    available: bool = _C_ENGINE_AVAILABLE

    def __init__(self, n_players: int = 4, heuristic_frac: float = 0.4) -> None:
        if not _C_ENGINE_AVAILABLE:
            raise RuntimeError(
                "skull_king_engine C extension not built. "
                "Run: python skull_king/_core/setup_engine.py build_ext --inplace"
            )
        self.n_players = n_players
        self.heuristic_frac = heuristic_frac
        self._weights_loaded = False

    def load_adv_weights(self, state_dict: dict) -> None:
        """Load PyTorch advantage-net state dict into C weight buffers.

        Expected keys: net.0.weight [512,244], net.0.bias [512],
                       net.2.weight [512,512], net.2.bias [512],
                       net.4.weight [82,512],  net.4.bias [82]
        """
        import torch
        def _np(key: str) -> np.ndarray:
            t = state_dict[key]
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().float().numpy()
            return np.asarray(t, dtype=np.float32)

        w1 = np.ascontiguousarray(_np("net.0.weight"), dtype=np.float32)
        b1 = np.ascontiguousarray(_np("net.0.bias"),   dtype=np.float32)
        w2 = np.ascontiguousarray(_np("net.2.weight"), dtype=np.float32)
        b2 = np.ascontiguousarray(_np("net.2.bias"),   dtype=np.float32)
        w3 = np.ascontiguousarray(_np("net.4.weight"), dtype=np.float32)
        b3 = np.ascontiguousarray(_np("net.4.bias"),   dtype=np.float32)

        _c_set_weights(w1, b1, w2, b2, w3, b3)
        self._weights_loaded = True

    def traverse(
        self, traverser: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Run one CFR traversal in C. Returns the same 7-tuple as traversal.traverse().

        Parameters
        ----------
        traverser : int
            Player whose regrets are updated (0 .. n_players-1).
        seed : int
            RNG seed for deal ordering and action sampling.

        Returns
        -------
        adv_obs, adv_masks, adv_targets, adv_actions,
        strat_obs, strat_masks, strat_strategies
        """
        if not self._weights_loaded:
            raise RuntimeError("Call load_adv_weights() before traverse().")
        return _c_traverse(traverser, int(seed), self.n_players, self.heuristic_frac)


class SplitCEngine:
    """Wraps the skull_king_engine C extension for split-net CFR traversal."""

    available: bool = _C_SPLIT_AVAILABLE

    def __init__(self, n_players: int = 4, heuristic_frac: float = 0.4) -> None:
        if not _C_SPLIT_AVAILABLE:
            raise RuntimeError(
                "skull_king_engine C extension does not support split nets. "
                "Rebuild after adding split-net support."
            )
        self.n_players = n_players
        self.heuristic_frac = heuristic_frac
        self._bid_loaded = False
        self._play_loaded = False

    def load_weights(self, bid_state_dict: dict, play_state_dict: dict) -> None:
        """Load bid and play advantage-net state dicts into C weight buffers.

        Expected keys in each dict:
            net.0.weight, net.0.bias, net.2.weight, net.2.bias,
            net.4.weight, net.4.bias
        """
        import torch

        def _np(sd: dict, key: str) -> np.ndarray:
            t = sd[key]
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().float().numpy()
            return np.asarray(t, dtype=np.float32)

        # Bid network
        bw1 = np.ascontiguousarray(_np(bid_state_dict, "net.0.weight"), dtype=np.float32)
        bb1 = np.ascontiguousarray(_np(bid_state_dict, "net.0.bias"),   dtype=np.float32)
        bw2 = np.ascontiguousarray(_np(bid_state_dict, "net.2.weight"), dtype=np.float32)
        bb2 = np.ascontiguousarray(_np(bid_state_dict, "net.2.bias"),   dtype=np.float32)
        bw3 = np.ascontiguousarray(_np(bid_state_dict, "net.4.weight"), dtype=np.float32)
        bb3 = np.ascontiguousarray(_np(bid_state_dict, "net.4.bias"),   dtype=np.float32)
        _c_set_bid_weights(bw1, bb1, bw2, bb2, bw3, bb3)
        self._bid_loaded = True

        # Play network
        pw1 = np.ascontiguousarray(_np(play_state_dict, "net.0.weight"), dtype=np.float32)
        pb1 = np.ascontiguousarray(_np(play_state_dict, "net.0.bias"),   dtype=np.float32)
        pw2 = np.ascontiguousarray(_np(play_state_dict, "net.2.weight"), dtype=np.float32)
        pb2 = np.ascontiguousarray(_np(play_state_dict, "net.2.bias"),   dtype=np.float32)
        pw3 = np.ascontiguousarray(_np(play_state_dict, "net.4.weight"), dtype=np.float32)
        pb3 = np.ascontiguousarray(_np(play_state_dict, "net.4.bias"),   dtype=np.float32)
        _c_set_play_weights(pw1, pb1, pw2, pb2, pw3, pb3)
        self._play_loaded = True

    def traverse(self, traverser: int, seed: int) -> tuple:
        """Run one split-net CFR traversal in C. Returns the same 14-tuple as
        traversal.traverse_split().

        Parameters
        ----------
        traverser : int
            Player whose regrets are updated (0 .. n_players-1).
        seed : int
            RNG seed for deal ordering and action sampling.

        Returns
        -------
        bid_adv_obs, bid_adv_masks, bid_adv_targets, bid_adv_actions,
        bid_strat_obs, bid_strat_masks, bid_strat_strategies,
        play_adv_obs, play_adv_masks, play_adv_targets, play_adv_actions,
        play_strat_obs, play_strat_masks, play_strat_strategies
        """
        if not self._bid_loaded or not self._play_loaded:
            raise RuntimeError("Call load_weights() before traverse().")
        return _c_traverse_split(traverser, int(seed), self.n_players, self.heuristic_frac)


# ---------------------------------------------------------------------------
# Worker-process integration
# ---------------------------------------------------------------------------

# Module-level singleton — built once per worker process in worker_init_c().
_ENGINE: Optional[CEngine] = None


def worker_init_c(
    adv_weights: dict,
    n_players: int,
    heuristic_frac: float = 0.4,
) -> None:
    """Initialise the C engine in a worker process (call as pool initializer).

    Mirrors the signature of traversal.worker_init so callers can swap them.
    """
    import torch
    torch.set_num_threads(1)
    global _ENGINE
    _ENGINE = CEngine(n_players=n_players, heuristic_frac=heuristic_frac)
    _ENGINE.load_adv_weights(adv_weights)


def worker_task_c(args: tuple) -> tuple:
    """Unpack args and run one C traversal. Drop-in for traversal.worker_task."""
    traverser, seed, n_players = args
    assert _ENGINE is not None, "worker_init_c() was not called"
    return _ENGINE.traverse(traverser, int(seed))


def reload_weights_c(adv_weights: dict) -> None:
    """Reload advantage-net weights in the current worker process."""
    assert _ENGINE is not None
    _ENGINE.load_adv_weights(adv_weights)
