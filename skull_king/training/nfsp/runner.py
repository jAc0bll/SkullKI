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
from skull_king.training.rebel.public_belief_state import encode_pbs_batch, pbs_encoding_size
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
        pbs_size = pbs_encoding_size(n_players)
        self._pbs_size = pbs_size

        # Pre-allocated numpy buffers for the current step (one row per env)
        self._enc_buf  = np.empty((n_envs, pbs_size), dtype=np.float32)
        self._mask_buf = np.empty((n_envs, ACTION_SPACE_SIZE), dtype=bool)

        # Pre-allocated GPU tensors — avoids repeated from_numpy + .to(device)
        self._enc_t  = torch.empty((n_envs, pbs_size), dtype=torch.float32, device=device)
        self._mask_t = torch.empty((n_envs, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)

        # Per-env pending transitions stored in flat numpy arrays (no Python dicts).
        # MAX_DECISIONS_PER_GAME = 4 players × ~65 decisions/player (worst case round 10).
        MAX_PEND = 300
        self._pend_encs    = np.empty((n_envs, MAX_PEND, pbs_size), dtype=np.float32)
        self._pend_masks   = np.empty((n_envs, MAX_PEND, ACTION_SPACE_SIZE), dtype=bool)
        self._pend_actions = np.empty((n_envs, MAX_PEND), dtype=np.int32)
        self._pend_players = np.empty((n_envs, MAX_PEND), dtype=np.int8)
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

            # ── gather active engines ────────────────────────────────
            active_idx: list[int] = []
            active_eng: list = []
            players: list[int] = []

            for i, eng in enumerate(self.engines):
                if eng._phase != GamePhase.GAME_OVER:
                    player = eng._current_player_index()
                    active_idx.append(i)
                    active_eng.append(eng)
                    players.append(player)

            if not active_idx:
                break

            N = len(active_idx)

            # ── batch-encode PBS (no per-engine object creation) ─────
            enc_np  = self._enc_buf[:N]
            mask_np = self._mask_buf[:N]
            encode_pbs_batch(active_eng, players, out=enc_np)
            for j, eng in enumerate(active_eng):
                mask_np[j] = (
                    _bid_mask(eng)
                    if eng._phase == GamePhase.BIDDING
                    else _build_action_mask(eng)
                )

            # ── batched GPU inference (reuse pre-allocated tensors) ──
            self._enc_t[:N].copy_(torch.from_numpy(enc_np))
            self._mask_t[:N].copy_(torch.from_numpy(mask_np))
            enc_t  = self._enc_t[:N]
            mask_t = self._mask_t[:N]

            # ── GPU-side action selection ────────────────────────────
            # Decide BR vs Avg per-env upfront, then do action selection on GPU
            use_br_np = self._rng.random(N) < cfg.eta
            use_br_t  = torch.from_numpy(use_br_np).to(self.device)

            # torch.compile w/ reduce-overhead uses CUDA graphs that reuse output
            # buffers; mark step boundary so q_vals isn't overwritten by avg_lp.
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.no_grad():
                q_vals = q_net(enc_t, mask_t).clone()  # [N, A]  (clone: keep alive across next compile call)
                avg_lp = avg_net(enc_t, mask_t)        # [N, A]

                # BR branch: ε-greedy
                #   argmax_a Q(s,a) over legal actions (illegal already -1e9)
                br_greedy = q_vals.argmax(dim=-1)     # [N]
                #   ε-random: sample uniformly over legal actions
                legal_f = mask_t.float()
                legal_probs = legal_f / legal_f.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                br_random = torch.multinomial(legal_probs, num_samples=1).squeeze(-1)  # [N]
                explore = (torch.rand(N, device=self.device) < cfg.epsilon)
                br_action = torch.where(explore, br_random, br_greedy)  # [N]

                # Avg branch: sample from policy
                avg_probs = avg_lp.exp() * legal_f
                avg_probs = avg_probs / avg_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                avg_action = torch.multinomial(avg_probs, num_samples=1).squeeze(-1)  # [N]

                actions_t = torch.where(use_br_t, br_action, avg_action)

            # Single GPU→CPU transfer for the entire batch of actions
            actions_np = actions_t.cpu().numpy()

            # ── apply actions + store transitions ────────────────────
            br_mask_np = use_br_np  # boolean array of shape [N]
            n_br = int(br_mask_np.sum())
            if n_br > 0:
                # Indices of BR decisions for batched SL insert
                br_idx_local = np.where(br_mask_np)[0]
                sl_buf.add_batch(
                    enc_np[br_idx_local].copy(),
                    mask_np[br_idx_local].copy(),
                    actions_np[br_idx_local].astype(np.int64),
                )

            for j, i in enumerate(active_idx):
                eng    = self.engines[i]
                player = players[j]
                action = int(actions_np[j])

                # Direct slice-assign into pre-allocated arrays — no dict, no allocation
                k = self._pend_lens[i]
                self._pend_encs[i, k]    = enc_np[j]
                self._pend_masks[i, k]   = mask_np[j]
                self._pend_actions[i, k] = action
                self._pend_players[i, k] = player
                self._pend_lens[i] = k + 1

                try:
                    if eng._phase == GamePhase.BIDDING:
                        eng.place_bid_no_state(player, action)
                    else:
                        card, tm = _action_to_card(action, eng)
                        eng.play_card_no_state(player, card, tm)
                except Exception:
                    self._pend_lens[i] = k     # roll back the just-stored transition
                    self._restart(i)

            total += N

        return total

    def _flush(self, i: int, rl_buf: RLBuffer) -> None:
        """Assign MC returns to all transitions of game i and push to RL buffer."""
        L = int(self._pend_lens[i])
        if L == 0:
            return

        eng = self.engines[i]
        scores = [p.total_score for p in eng._players]

        # Zero-copy views into the pre-allocated arrays
        encs    = self._pend_encs[i, :L]
        masks   = self._pend_masks[i, :L]
        actions = self._pend_actions[i, :L]
        players = self._pend_players[i, :L]

        # Vectorized MC return assignment (per-player terminal utility)
        per_player = np.array(
            [_utility_from_scores(scores, p) for p in range(self.n_players)],
            dtype=np.float32,
        )
        returns = per_player[players]  # gather: [L]

        rl_buf.add_batch(encs, masks, actions, returns)
        self._pend_lens[i] = 0

    def _restart(self, i: int) -> None:
        seed = int(self._rng.integers(0, 2**31))
        eng = GameEngine(n_players=self.n_players, seed=seed)
        eng.start()
        self.engines[i] = eng
        self._pend_lens[i] = 0
