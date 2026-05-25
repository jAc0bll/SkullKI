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
        self._init_envs()

        # Pre-allocated numpy buffers — reused every step, avoids GC pressure
        pbs_size = pbs_encoding_size(n_players)
        self._enc_buf  = np.empty((n_envs, pbs_size), dtype=np.float32)
        self._mask_buf = np.empty((n_envs, ACTION_SPACE_SIZE), dtype=bool)
        # Pre-allocated GPU tensors — avoids repeated from_numpy + .to(device)
        self._enc_t  = torch.empty((n_envs, pbs_size), dtype=torch.float32, device=device)
        self._mask_t = torch.empty((n_envs, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)

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

            with torch.no_grad():
                q_vals = q_net(enc_t, mask_t)         # [N, A]
                avg_lp = avg_net(enc_t, mask_t)       # [N, A]

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

                # Store transition references (copy needed: enc_np/mask_np overwritten next step)
                self._pending[i].append({
                    "enc": enc_np[j].copy(),
                    "mask": mask_np[j].copy(),
                    "action": action,
                    "player": player,
                })

                try:
                    if eng._phase == GamePhase.BIDDING:
                        eng.place_bid_no_state(player, action)
                    else:
                        card, tm = _action_to_card(action, eng)
                        eng.play_card_no_state(player, card, tm)
                except Exception:
                    self._pending[i].pop()
                    self._restart(i)

            total += N

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
