"""Curriculum-aware vectorized self-play runner.

Single-process, GPU-batched. Manages N parallel Skull-King games. Each
game has a fixed (per-game) assignment of opponents to seats 1-3; seat 0
is always the live learning agent.

Inference is dispatched by opponent type:
  - SELF / LEAGUE  : batched GPU forward through the relevant Q-network
  - HEURISTIC      : per-engine call to HeuristicAgent (CPU)
  - RANDOM         : uniform legal action via numpy

Only seat-0 (learning agent) transitions are appended to the MC replay
buffer. Other seats still need actions executed in the engines so the
games progress, but their data is never trained on.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from skull_king.agents import HeuristicAgent
from skull_king.cards import CardType, TigressMode
from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GamePhase
from skull_king.training.cfr.traversal import _utility_from_scores
from skull_king.training.douzero.opponents import (
    LeaguePool,
    OPP_HEURISTIC,
    OPP_LEAGUE_0,
    OPP_RANDOM,
    OPP_SELF,
)
from skull_king.training.rebel.public_belief_state import (
    encode_pbs_batch,
    pbs_encoding_size,
)
from skull_king.training.rebel.subgame import _action_to_card, _build_action_mask

if TYPE_CHECKING:
    from skull_king.training.douzero.buffers import MCReplayBuffer
    from skull_king.training.douzero.networks import DouZeroQNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bid_mask(eng: GameEngine) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for b in range(eng._round + 1):  # _round is 1-indexed
        mask[b] = True
    return mask


def _heuristic_action(eng: GameEngine, player: int, heur: HeuristicAgent) -> int:
    """Resolve a HeuristicAgent decision down to an integer action.

    The heuristic agent operates on a frozen GameState. For PLAY actions we
    must convert the returned (Card, TigressMode) back into the discrete
    action index used by the engine.
    """
    state = eng.get_state()
    heur.before_move(eng)

    if eng._phase == GamePhase.BIDDING:
        return heur.bid(state, player)

    card, tm = heur.play(state, player)

    # Map (card, tigress mode) back to action index.
    # ACTION_SPACE_SIZE encodes plays in the same order as the canonical deck
    # plus two special Tigress-mode actions.
    from skull_king.env.skull_king_env import (
        N_BID_ACTIONS,
        TIGRESS_AS_ESCAPE_ACTION,
        TIGRESS_AS_PIRATE_ACTION,
        _HASH_TO_SLOTS,
    )
    if card.card_type == CardType.TIGRESS:
        return (
            TIGRESS_AS_PIRATE_ACTION if tm == TigressMode.PIRATE
            else TIGRESS_AS_ESCAPE_ACTION
        )
    slots = _HASH_TO_SLOTS[card._hash]
    return N_BID_ACTIONS + slots[0]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CurriculumRunner:
    """Vectorized runner with per-game opponent assignments."""

    MAX_PEND = 80  # max seat-0 decisions per game (≤ 65 in worst case)

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

        # CPU buffers for the current step (one row per env, sliced to active)
        self._enc_buf  = np.empty((n_envs, self._pbs_size), dtype=np.float32)
        self._mask_buf = np.empty((n_envs, ACTION_SPACE_SIZE), dtype=bool)
        # Pre-allocated GPU tensors — reused via copy_, no per-step allocs
        self._enc_t  = torch.empty((n_envs, self._pbs_size), dtype=torch.float32, device=device)
        self._mask_t = torch.empty((n_envs, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)

        # Per-env seat-0 pending transitions (we only ever store seat-0 data
        # because that's the learning agent). We also track the *round* each
        # transition belongs to, so we can assign per-round-shaped MC returns
        # (return-to-go from the start of that round until the game ends) —
        # this gives much sharper credit assignment than a single terminal
        # game return for a 10-round game.
        self._pend_encs    = np.empty((n_envs, self.MAX_PEND, self._pbs_size), dtype=np.float32)
        self._pend_masks   = np.empty((n_envs, self.MAX_PEND, ACTION_SPACE_SIZE), dtype=bool)
        self._pend_actions = np.empty((n_envs, self.MAX_PEND), dtype=np.int32)
        self._pend_rounds  = np.empty((n_envs, self.MAX_PEND), dtype=np.int8)
        self._pend_lens    = np.zeros(n_envs, dtype=np.int32)

        # Per-env opponent assignment table (assigned at game start, [n_envs, n_players])
        self._opp = np.zeros((n_envs, n_players), dtype=np.int16)

        # Shared heuristic agent (stateless across games)
        self._heur = HeuristicAgent()

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
    # Curriculum: assign opponents per env. Called by trainer before collect.
    # ------------------------------------------------------------------

    def set_opponent_assignments(self, opp_table: np.ndarray) -> None:
        assert opp_table.shape == (self.n_envs, self.n_players)
        self._opp = opp_table.astype(np.int16, copy=True)

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def collect(
        self,
        live_q: "DouZeroQNet",
        league: LeaguePool,
        buf: "MCReplayBuffer",
        epsilon: float,
        n_decisions: int,
    ) -> int:
        """Run games for at least ``n_decisions`` total decisions; return actual count."""
        total = 0
        n_envs = self.n_envs

        while total < n_decisions:
            # ── flush finished games + reset opponents for new games ────
            for i in range(n_envs):
                if self.engines[i]._phase == GamePhase.GAME_OVER:
                    self._flush(i, buf)
                    self._restart(i)  # opponent table stays — refreshed by trainer

            # ── gather active engines + current acting player ────────────
            active_idx: list[int] = []
            active_eng: list[GameEngine] = []
            players: list[int] = []
            opp_types: list[int] = []

            for i in range(n_envs):
                eng = self.engines[i]
                if eng._phase == GamePhase.GAME_OVER:
                    continue
                p = eng._current_player_index()
                active_idx.append(i)
                active_eng.append(eng)
                players.append(p)
                opp_types.append(int(self._opp[i, p]))

            if not active_idx:
                break

            N = len(active_idx)
            active_idx_arr = np.array(active_idx, dtype=np.int32)
            players_arr    = np.array(players,    dtype=np.int32)
            opp_arr        = np.array(opp_types,  dtype=np.int16)

            # ── batch-encode PBS for all active engines ─────────────────
            enc_np  = self._enc_buf[:N]
            mask_np = self._mask_buf[:N]
            encode_pbs_batch(active_eng, players, out=enc_np)
            for j, eng in enumerate(active_eng):
                mask_np[j] = _bid_mask(eng) if eng._phase == GamePhase.BIDDING else _build_action_mask(eng)

            actions = np.empty(N, dtype=np.int32)

            # ── dispatch by opponent type ───────────────────────────────
            self_mask = opp_arr == OPP_SELF
            heur_mask = opp_arr == OPP_HEURISTIC
            rand_mask = opp_arr == OPP_RANDOM
            league_mask = (opp_arr >= OPP_LEAGUE_0) & (opp_arr < OPP_HEURISTIC)

            # SELF (live Q-net) — batched GPU inference + ε-greedy
            if self_mask.any():
                idxs = np.where(self_mask)[0]
                actions[idxs] = self._batched_qnet_action(
                    live_q, enc_np[idxs], mask_np[idxs], epsilon
                )

            # LEAGUE snapshots — batched GPU inference per snapshot id, greedy
            if league_mask.any() and len(league) > 0:
                league_idxs_local = np.where(league_mask)[0]
                snap_ids = opp_arr[league_idxs_local] - OPP_LEAGUE_0
                for snap_id in np.unique(snap_ids):
                    snap = league.get(int(snap_id))
                    sub = league_idxs_local[snap_ids == snap_id]
                    actions[sub] = self._batched_qnet_action(
                        snap, enc_np[sub], mask_np[sub], epsilon=0.0
                    )
            elif league_mask.any():
                # League selected but pool empty — fall back to live net
                idxs = np.where(league_mask)[0]
                actions[idxs] = self._batched_qnet_action(
                    live_q, enc_np[idxs], mask_np[idxs], epsilon
                )

            # HEURISTIC — per-engine CPU calls
            if heur_mask.any():
                for j in np.where(heur_mask)[0]:
                    actions[j] = _heuristic_action(active_eng[j], players[j], self._heur)

            # RANDOM — uniform legal action
            if rand_mask.any():
                for j in np.where(rand_mask)[0]:
                    legal = np.where(mask_np[j])[0]
                    actions[j] = int(self._rng.choice(legal))

            # ── store seat-0 transitions + apply actions ────────────────
            for k in range(N):
                i = active_idx[k]
                p = players[k]
                a = int(actions[k])
                eng = active_eng[k]

                # Only seat 0 (learning agent) transitions go into the buffer
                if p == 0 and self._opp[i, 0] == OPP_SELF:
                    L = self._pend_lens[i]
                    if L < self.MAX_PEND:
                        self._pend_encs[i, L]    = enc_np[k]
                        self._pend_masks[i, L]   = mask_np[k]
                        self._pend_actions[i, L] = a
                        self._pend_rounds[i, L]  = eng._round    # 1-indexed round
                        self._pend_lens[i] = L + 1

                try:
                    if eng._phase == GamePhase.BIDDING:
                        eng.place_bid_no_state(p, a)
                    else:
                        card, tm = _action_to_card(a, eng)
                        eng.play_card_no_state(p, card, tm)
                except Exception:
                    # Invalid action — discard last seat-0 transition if any, restart env
                    if p == 0 and self._pend_lens[i] > 0:
                        self._pend_lens[i] -= 1
                    self._restart(i)

            total += N

        return total

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _batched_qnet_action(
        self,
        net: "DouZeroQNet",
        enc: np.ndarray,
        mask: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """Run ε-greedy on a batch through ``net``. Returns int32 actions."""
        N = enc.shape[0]
        # Copy into pre-allocated GPU tensor slices for zero-realloc transfer
        self._enc_t[:N].copy_(torch.from_numpy(enc))
        self._mask_t[:N].copy_(torch.from_numpy(mask))

        with torch.no_grad():
            q = net(self._enc_t[:N], self._mask_t[:N])           # [N, A]
            greedy = q.argmax(dim=-1)                            # [N]

            if epsilon > 0.0:
                legal_f = self._mask_t[:N].float()
                legal_p = legal_f / legal_f.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                rand_a  = torch.multinomial(legal_p, num_samples=1).squeeze(-1)
                explore = torch.rand(N, device=self.device) < epsilon
                a = torch.where(explore, rand_a, greedy)
            else:
                a = greedy

        return a.cpu().numpy().astype(np.int32)

    def _flush(self, i: int, buf: "MCReplayBuffer") -> None:
        """Flush seat-0 transitions with per-round-shaped MC returns.

        For each transition at round r, the return-to-go is the seat-0
        score earned in rounds r, r+1, ..., 10. This gives a much denser
        learning signal than a single terminal game return spanning 65
        decisions over 10 rounds.
        """
        L = int(self._pend_lens[i])
        if L == 0:
            return
        eng = self.engines[i]

        # seat-0 per-round scores (1-indexed by round number)
        score_history = eng._players[0].score_history  # list[RoundScore], length 10
        round_scores = np.array(
            [rs.total_score for rs in score_history], dtype=np.float32
        )                                                # shape [n_rounds]
        # tail_scores[r] = sum of scores from round r+1..end (1-indexed r)
        tail = np.concatenate([round_scores, np.zeros(1, dtype=np.float32)])
        cumtail = tail[::-1].cumsum()[::-1]              # cumtail[k] = sum(tail[k:])

        encs    = self._pend_encs[i, :L]
        masks   = self._pend_masks[i, :L]
        actions = self._pend_actions[i, :L]
        rounds  = self._pend_rounds[i, :L].astype(np.int32) - 1  # → 0-indexed
        rounds  = np.clip(rounds, 0, len(round_scores))
        returns = cumtail[rounds].astype(np.float32)

        buf.add_batch(encs, masks, actions, returns)
        self._pend_lens[i] = 0

    def _restart(self, i: int) -> None:
        seed = int(self._rng.integers(0, 2**31))
        eng = GameEngine(n_players=self.n_players, seed=seed)
        eng.start()
        self.engines[i] = eng
        self._pend_lens[i] = 0
