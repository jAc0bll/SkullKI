"""Tournament-compatible agent backed by a trained Deep CFR strategy network."""
from __future__ import annotations

from typing import Any, Optional

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.env.skull_king_env import SkullKingEnv
from skull_king.game_state import GameState
from skull_king.training.cfr.networks import StrategyNet


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
