"""Vectorized game runner — N games in parallel, batched GPU inference.

All games run in the main process (no subprocesses). At every decision
point we collect all pending states, fire a single batched GPU forward
pass, distribute actions, and step every game forward. This keeps GPU
utilization high without multiprocessing overhead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GamePhase
from skull_king.training.cfr.traversal import _utility_from_scores
from skull_king.training.rebel.public_belief_state import PublicBeliefState
from skull_king.training.rebel.subgame import _action_to_card, _build_action_mask

if TYPE_CHECKING:
    from skull_king.training.nfsp.buffers import RLBuffer, SLBuffer
    from skull_king.training.nfsp.networks import NfspAvgNet, NfspQNet
    from skull_king.training.nfsp.train import NfspConfig


def _bid_mask(eng: GameEngine) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for b in range(eng._round + 1):  # _round is 1-indexed
        mask[b] = True
    return mask


class VectorizedRunner:
    """Manages N parallel Skull King games with fully batched GPU inference."""

    def __init__(
        self,
        n_envs: int,
        n_players: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.n_envs = n_envs
        self.n_players = n_players
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._init_envs()

    def _init_envs(self) -> None:
        seeds = self._rng.integers(0, 2**31, size=self.n_envs)
        self.engines: list[GameEngine] = []
        for s in seeds:
            eng = GameEngine(n_players=self.n_players, seed=int(s))
            eng.start()
            self.engines.append(eng)
        # Per-game: list of (enc, mask, action, player_idx) accumulated during game
        self._pending: list[list[dict]] = [[] for _ in range(self.n_envs)]

    # ------------------------------------------------------------------

    def collect(
        self,
        q_net: NfspQNet,
        avg_net: NfspAvgNet,
        rl_buf: RLBuffer,
        sl_buf: SLBuffer,
        cfg: NfspConfig,
        n_decisions: int,
    ) -> int:
        """Collect at least n_decisions across all envs. Returns actual count."""
        total = 0

        while total < n_decisions:
            # ── flush finished games ─────────────────────────────────
            for i, eng in enumerate(self.engines):
                if eng._phase == GamePhase.GAME_OVER:
                    self._flush(i, rl_buf)
                    self._restart(i)

            # ── build batch of all pending decisions ─────────────────
            active_idx: list[int] = []
            encs: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            players: list[int] = []

            for i, eng in enumerate(self.engines):
                if eng._phase == GamePhase.GAME_OVER:
                    continue
                player = eng._current_player_index()
                pbs = PublicBeliefState.from_engine(eng, player)
                enc = pbs.encode()
                mask = _bid_mask(eng) if eng._phase == GamePhase.BIDDING else _build_action_mask(eng)
                active_idx.append(i)
                encs.append(enc)
                masks.append(mask)
                players.append(player)

            if not active_idx:
                break

            # ── batched GPU inference ────────────────────────────────
            enc_t = torch.from_numpy(np.stack(encs)).float().to(self.device)
            mask_t = torch.from_numpy(np.stack(masks)).bool().to(self.device)

            with torch.no_grad():
                q_vals = q_net(enc_t, mask_t).cpu().numpy()
                avg_lp = avg_net(enc_t, mask_t).cpu().numpy()

            use_br = self._rng.random(len(active_idx)) < cfg.eta

            # ── select and apply actions ─────────────────────────────
            for j, i in enumerate(active_idx):
                eng = self.engines[i]
                enc = encs[j]
                mask = masks[j]
                player = players[j]
                legal = np.where(mask)[0]

                if use_br[j]:
                    if self._rng.random() < cfg.epsilon:
                        action = int(self._rng.choice(legal))
                    else:
                        action = int(np.argmax(q_vals[j]))
                    sl_buf.add_batch(
                        enc[None], mask[None], np.array([action], dtype=np.int64)
                    )
                else:
                    probs = np.exp(avg_lp[j])
                    probs = np.maximum(probs, 0.0)
                    probs[~mask] = 0.0
                    s = probs.sum()
                    probs = probs / s if s > 1e-9 else mask.astype(np.float64) / mask.sum()
                    action = int(self._rng.choice(len(probs), p=probs))

                self._pending[i].append(
                    {"enc": enc, "mask": mask, "action": action, "player": player}
                )

                try:
                    if eng._phase == GamePhase.BIDDING:
                        eng.place_bid_no_state(player, action)
                    else:
                        card, tm = _action_to_card(action, eng)
                        eng.play_card_no_state(player, card, tm)
                except Exception:
                    # Invalid action (shouldn't happen with correct mask) — restart
                    self._pending[i].pop()
                    self._restart(i)

            total += len(active_idx)

        return total

    def _flush(self, i: int, rl_buf: RLBuffer) -> None:
        """Assign MC returns to all transitions of game i and push to RL buffer."""
        eng = self.engines[i]
        transitions = self._pending[i]
        if not transitions:
            return

        scores = [p.total_score for p in eng._players]
        n = len(transitions)
        encs = np.stack([t["enc"] for t in transitions])
        masks = np.stack([t["mask"] for t in transitions])
        actions = np.array([t["action"] for t in transitions], dtype=np.int32)
        returns = np.array(
            [_utility_from_scores(scores, t["player"]) for t in transitions],
            dtype=np.float32,
        )
        rl_buf.add_batch(encs, masks, actions, returns)
        self._pending[i] = []

    def _restart(self, i: int) -> None:
        seed = int(self._rng.integers(0, 2**31))
        eng = GameEngine(n_players=self.n_players, seed=seed)
        eng.start()
        self.engines[i] = eng
        self._pending[i] = []
