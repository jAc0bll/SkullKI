"""AlphaZero self-play runner — batched MCTS across parallel games.

Each iteration:
  1. Maintain a pool of in-progress games. For each game where it's seat-0's
     turn, advance opponent plays (in batched lock-step) to the next seat-0
     decision point.
  2. Run batched MCTS at all current decision points simultaneously.
  3. Sample one action per game from the MCTS visit-count distribution
     (with temperature) and append the (state, π, ...) tuple to a pending
     trajectory list for that game.
  4. Apply the chosen actions, possibly completing some games. When a game
     ends, compute the seat-0 outcome and finalise all of its pending
     transitions (filling in the value target).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GamePhase
from skull_king.training.alphazero.mcts import (
    _advance_opponents,
    _apply_action,
    _legal_mask,
    run_batched_mcts,
)
from skull_king.training.rebel.public_belief_state import (
    encode_pbs_batch,
    pbs_encoding_size,
)

if TYPE_CHECKING:
    from skull_king.training.alphazero.buffers import AZReplayBuffer
    from skull_king.training.alphazero.networks import AlphaZeroNet


class AlphaZeroRunner:
    """Vectorized self-play runner with batched MCTS."""

    AGENT_SEAT = 0
    VALUE_SCALE = 100.0   # divide raw round-score sums to land in [-1, +1]
    MAX_PEND = 80         # max seat-0 decisions per game (≤ 65 in worst case)

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
        self._pbs_size = pbs_encoding_size(n_players)

        # Per-env pending seat-0 transitions
        self._pend_encs    = np.empty((n_envs, self.MAX_PEND, self._pbs_size), dtype=np.float32)
        self._pend_masks   = np.empty((n_envs, self.MAX_PEND, ACTION_SPACE_SIZE), dtype=bool)
        self._pend_policy  = np.empty((n_envs, self.MAX_PEND, ACTION_SPACE_SIZE), dtype=np.float32)
        self._pend_rounds  = np.empty((n_envs, self.MAX_PEND), dtype=np.int8)
        self._pend_lens    = np.zeros(n_envs, dtype=np.int32)

        self._init_envs()

    def _init_envs(self) -> None:
        seeds = self._rng.integers(0, 2**31, size=self.n_envs)
        self.engines: list[GameEngine] = []
        for s in seeds:
            eng = GameEngine(n_players=self.n_players, seed=int(s))
            eng.start()
            self.engines.append(eng)
        self._pend_lens[:] = 0

    # ------------------------------------------------------------------
    # Main collection: do one MCTS-decision round across all active envs
    # ------------------------------------------------------------------

    def collect(
        self,
        network: "AlphaZeroNet",
        buf: "AZReplayBuffer",
        n_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        temperature: float,
        max_decisions: int,
    ) -> int:
        """Run self-play with batched MCTS until ``max_decisions`` seat-0
        decisions have been recorded across all envs. Returns actual count."""
        total = 0
        while total < max_decisions:
            # 1. Flush finished + restart, then advance opponents so it's
            #    seat-0's turn (or game terminal) in every env.
            for i, eng in enumerate(self.engines):
                if eng._phase == GamePhase.GAME_OVER:
                    self._flush(i, buf)
                    self._restart(i)

            terminal_after = _advance_opponents(
                self.engines, network, self.AGENT_SEAT, self.device, self._rng,
            )
            # Engines that hit terminal during opponent advance: flush + restart
            for i, t in enumerate(terminal_after):
                if t and self.engines[i]._phase == GamePhase.GAME_OVER:
                    self._flush(i, buf)
                    self._restart(i)
                    # Once restarted, advance opponents for THIS env so it's seat-0's turn
                    _advance_opponents(
                        [self.engines[i]], network, self.AGENT_SEAT, self.device, self._rng,
                    )

            # 2. At this point every engine is at a seat-0 decision (or just-restarted).
            #    Run batched MCTS across all of them.
            active_idx = [
                i for i, eng in enumerate(self.engines)
                if eng._phase != GamePhase.GAME_OVER and eng._current_player_index() == self.AGENT_SEAT
            ]
            if not active_idx:
                continue

            active_engs = [self.engines[i] for i in active_idx]

            policies, _ = run_batched_mcts(
                engines_root=active_engs,
                network=network,
                n_simulations=n_simulations,
                device=self.device,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                rng=self._rng,
                add_root_noise=True,
                agent_seat=self.AGENT_SEAT,
            )

            # 3. Encode the actual root states (for buffer storage), sample
            #    one action per env from the MCTS policy.
            root_encs = encode_pbs_batch(active_engs, [self.AGENT_SEAT] * len(active_engs))
            root_masks = np.stack([_legal_mask(eng) for eng in active_engs])

            for k, i in enumerate(active_idx):
                pi = policies[k]
                mask = root_masks[k]

                # Sampling policy: temperature τ shapes (pi^(1/τ)) over legal actions.
                # τ ≈ 1 → sample proportional to visits; τ → 0 → argmax.
                if temperature <= 1e-3:
                    legal = np.where(mask)[0]
                    action = int(legal[np.argmax(pi[legal])])
                else:
                    p = np.maximum(pi, 1e-12) ** (1.0 / max(temperature, 1e-3))
                    p[~mask] = 0.0
                    s = p.sum()
                    if s <= 1e-12:
                        legal = np.where(mask)[0]
                        action = int(self._rng.choice(legal))
                    else:
                        action = int(self._rng.choice(ACTION_SPACE_SIZE, p=p / s))

                # Store seat-0 transition with MCTS policy as target
                L = int(self._pend_lens[i])
                if L < self.MAX_PEND:
                    self._pend_encs[i, L]   = root_encs[k]
                    self._pend_masks[i, L]  = mask
                    self._pend_policy[i, L] = pi
                    self._pend_rounds[i, L] = self.engines[i]._round
                    self._pend_lens[i] = L + 1

                # Apply the chosen action
                try:
                    _apply_action(self.engines[i], self.AGENT_SEAT, action)
                except Exception:
                    if self._pend_lens[i] > 0:
                        self._pend_lens[i] -= 1
                    self._restart(i)

            total += len(active_idx)
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush(self, i: int, buf: "AZReplayBuffer") -> None:
        """At game end, write all pending seat-0 transitions to the buffer."""
        L = int(self._pend_lens[i])
        if L == 0:
            return
        eng = self.engines[i]
        score_history = eng._players[self.AGENT_SEAT].score_history
        if len(score_history) == 0:
            self._pend_lens[i] = 0
            return

        round_scores = np.array(
            [rs.total_score for rs in score_history], dtype=np.float32
        )
        tail = np.concatenate([round_scores, np.zeros(1, dtype=np.float32)])
        cumtail = tail[::-1].cumsum()[::-1] / self.VALUE_SCALE
        # Clip into [-1, +1] for tanh-bounded value head
        cumtail = np.clip(cumtail, -1.0, 1.0)

        rounds = self._pend_rounds[i, :L].astype(np.int32) - 1
        rounds = np.clip(rounds, 0, len(round_scores))
        values = cumtail[rounds].astype(np.float32)

        buf.add_batch(
            self._pend_encs[i, :L],
            self._pend_masks[i, :L],
            self._pend_policy[i, :L],
            values,
        )
        self._pend_lens[i] = 0

    def _restart(self, i: int) -> None:
        seed = int(self._rng.integers(0, 2**31))
        eng = GameEngine(n_players=self.n_players, seed=seed)
        eng.start()
        self.engines[i] = eng
        self._pend_lens[i] = 0
