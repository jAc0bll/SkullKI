"""Tournament-compatible agents backed by trained Deep CFR strategy networks."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.env.skull_king_env import (
    N_BID_ACTIONS,
    SkullKingEnv,
    TIGRESS_AS_ESCAPE_ACTION,
    TIGRESS_AS_PIRATE_ACTION,
)
from skull_king.game_state import GameState
from skull_king.training.cfr.networks import (
    BiddingStratNet,
    PlayingStratNet,
    StrategyNet,
)
from skull_king.training.cfr.traversal import _global_to_play_mask, _play_local_to_global


class CFRAgent(BaseAgent):
    """Uses the Deep CFR average-strategy network for bidding and card play."""

    def __init__(
        self,
        strat_net: StrategyNet,
        n_players: int,
        name: str = "CFR",
        deterministic: bool = True,
    ) -> None:
        self.strat_net = strat_net
        self.name = name
        self._deterministic = deterministic
        self._util_env = SkullKingEnv(n_players=n_players)
        self._engine: Optional[Any] = None

    def before_move(self, engine: Any) -> None:
        self._engine = engine

    def bid(self, state: GameState, player_index: int) -> int:
        action = self._predict(state, player_index)
        return max(0, min(int(action), state.round_number))

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        action = self._predict(state, player_index)
        return self._util_env._decode_play_action(int(action))

    def _predict(self, state: GameState, player_index: int) -> int:
        completed = self._engine.completed_tricks_this_round if self._engine else []
        obs = self._util_env._build_observation_for(state, player_index, completed)
        mask = self._util_env._action_masks_for(state, player_index)
        return self.strat_net.predict(obs, mask, deterministic=self._deterministic)


class SplitCFRAgent(BaseAgent):
    """Tournament agent using separate bidding and playing strategy networks."""

    def __init__(
        self,
        bid_strat_net: BiddingStratNet,
        play_strat_net: PlayingStratNet,
        n_players: int,
        name: str = "CFR-split",
        deterministic: bool = True,
    ) -> None:
        self.bid_strat_net = bid_strat_net
        self.play_strat_net = play_strat_net
        self.name = name
        self._deterministic = deterministic
        self._util_env = SkullKingEnv(n_players=n_players)
        self._engine: Optional[Any] = None

    def before_move(self, engine: Any) -> None:
        self._engine = engine

    def bid(self, state: GameState, player_index: int) -> int:
        completed = self._engine.completed_tricks_this_round if self._engine else []
        obs = self._util_env._build_observation_for(state, player_index, completed)
        global_mask = self._util_env._action_masks_for(state, player_index)
        bid_mask = global_mask[:N_BID_ACTIONS]
        action = self.bid_strat_net.predict(obs, bid_mask, deterministic=self._deterministic)
        return max(0, min(int(action), state.round_number))

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        completed = self._engine.completed_tricks_this_round if self._engine else []
        obs = self._util_env._build_observation_for(state, player_index, completed)
        global_mask = self._util_env._action_masks_for(state, player_index)
        play_mask = _global_to_play_mask(global_mask)
        local_action = self.play_strat_net.predict(
            obs, play_mask, deterministic=self._deterministic
        )
        global_action = _play_local_to_global(int(local_action))
        return self._util_env._decode_play_action(global_action)
