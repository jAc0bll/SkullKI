"""NFSP self-play environment.

Each opponent independently uses either:
  - Best Response (BR): the frozen PPO copy  — with probability eta
  - Average Strategy (SL): the supervised-learning net — with probability 1-eta

The SL network lives inside each env instance (subprocess-safe).  The main
process syncs updated weights periodically via set_sl_weights().
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from skull_king.cards import Card, CardType, TigressMode
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE
from skull_king.game_state import GameState
from skull_king.resolver import TrickResolver
from skull_king.training.nfsp_sl import AverageStrategyNet
from skull_king.training.self_play_env import SelfPlaySkullKingEnv


class NFSPSelfPlayEnv(SelfPlaySkullKingEnv):
    """SelfPlaySkullKingEnv extended with NFSP average-strategy opponents.

    Parameters
    ----------
    eta:
        Probability that each opponent uses BR (frozen PPO) rather than SL.
        NFSP standard is 0.1.
    sl_hidden:
        Hidden layer sizes for the local AverageStrategyNet copy.
    """

    def __init__(
        self,
        eta: float = 0.1,
        sl_hidden: tuple[int, ...] = (256, 256),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eta = eta
        # Each subprocess holds its own SL net (CPU inference only).
        self.sl_net = AverageStrategyNet(
            obs_size=OBS_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden=sl_hidden,
        )
        self.sl_net.eval()
        # Per-episode mode flag: True → use SL, indexed by player seat.
        self._opp_use_sl: list[bool] = []

    # ------------------------------------------------------------------
    # Called from main process via vec_env.env_method(...)
    # ------------------------------------------------------------------

    def set_sl_weights(self, state_dict: dict) -> None:
        """Load new SL network weights from the main-process trainer."""
        cpu_sd = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in state_dict.items()
        }
        self.sl_net.load_state_dict(cpu_sd)
        self.sl_net.eval()

    # ------------------------------------------------------------------
    # Reset: assign BR/SL mode to each opponent for this episode
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any):  # type: ignore[override]
        obs, info = super().reset(**kwargs)
        self._opp_use_sl = [
            (i != self._controlled_player) and (self._rng.random() >= self.eta)
            for i in range(self.n_players)
        ]
        return obs, info

    # ------------------------------------------------------------------
    # Override opponent decision logic
    # ------------------------------------------------------------------

    def _opponent_bid(self, state: GameState, player_index: int) -> int:
        if self._use_heuristic():
            return self._heuristic.bid(state, player_index)
        if self._should_use_sl(player_index):
            return self._sl_bid(state, player_index)
        # Best response: frozen PPO
        action = self._opponent_act(state, player_index)
        return max(0, min(int(action), state.round_number))

    def _opponent_play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        if self._use_heuristic():
            return self._heuristic.play(state, player_index)
        if self._should_use_sl(player_index):
            return self._sl_play(state, player_index)
        # Best response: frozen PPO
        action = self._opponent_act(state, player_index)
        card, mode = self._decode_play_action(int(action))
        hand = list(state.player_states[player_index].hand)
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        if card not in hand or card not in legal:
            return self._heuristic.play(state, player_index)
        return card, mode

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_use_sl(self, player_index: int) -> bool:
        if not self._opp_use_sl or player_index >= len(self._opp_use_sl):
            return False
        return self._opp_use_sl[player_index]

    def _sl_bid(self, state: GameState, player_index: int) -> int:
        completed = self._engine.completed_tricks_this_round
        obs = self._build_observation_for(state, player_index, completed)
        mask = self._action_masks_for(state, player_index)
        action = self.sl_net.act(obs, mask, deterministic=False)
        return max(0, min(int(action), state.round_number))

    def _sl_play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        completed = self._engine.completed_tricks_this_round
        obs = self._build_observation_for(state, player_index, completed)
        mask = self._action_masks_for(state, player_index)
        action = self.sl_net.act(obs, mask, deterministic=False)
        card, mode = self._decode_play_action(int(action))
        hand = list(state.player_states[player_index].hand)
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        if card not in hand or card not in legal:
            return self._heuristic.play(state, player_index)
        return card, mode
