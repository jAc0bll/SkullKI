"""Self-play wrapper and model-based agent adapter for tournament evaluation."""
from __future__ import annotations

import copy
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from skull_king.agents.base_agent import BaseAgent
from skull_king.agents.heuristic_agent import HeuristicAgent
from skull_king.cards import Card, CardType, TigressMode
from skull_king.env.skull_king_env import SkullKingEnv
from skull_king.game_state import GamePhase, GameState
from skull_king.resolver import TrickResolver

if TYPE_CHECKING:
    from skull_king.engine import GameEngine


# ---------------------------------------------------------------------------
# Self-play environment
# ---------------------------------------------------------------------------


class SelfPlaySkullKingEnv(SkullKingEnv):
    """Extends SkullKingEnv so non-controlled players are driven by a frozen
    copy of the learning model rather than random policy.

    Call ``set_opponent(model)`` from a training callback to update the
    opponent after each policy iteration.  Until a model is provided, the env
    falls back to uniform-random auto-play so training can begin immediately.
    """

    def __init__(self, heuristic_mix: float = 0.3, **env_kwargs: Any) -> None:
        super().__init__(**env_kwargs)
        self._opponent: Optional[Any] = None  # MaskablePPO or None
        self._heuristic_mix = heuristic_mix   # fraction of opp turns using heuristic
        self._heuristic = HeuristicAgent()

    def set_opponent(self, model: Any) -> None:
        """Freeze a copy of *model* as the opponent policy."""
        self._opponent = copy.deepcopy(model)

    def set_heuristic_mix(self, mix: float) -> None:
        """Update the fraction of opponent turns that use the heuristic agent.
        Called by CurriculumCallback to anneal the mix over training."""
        self._heuristic_mix = float(np.clip(mix, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Override: opponents use the frozen model when available
    # ------------------------------------------------------------------

    def _advance_others(self, state: GameState) -> GameState:
        while state.phase not in (GamePhase.GAME_OVER,):
            if state.phase == GamePhase.BIDDING:
                cur = state.current_player_index
                if cur == self._controlled_player:
                    break
                bid = self._opponent_bid(state, cur)
                state = self._engine.place_bid(cur, bid)

            elif state.phase == GamePhase.PLAYING:
                cur = state.current_player_index
                if cur == self._controlled_player:
                    break
                card, mode = self._opponent_play(state, cur)
                state = self._engine.play_card(cur, card, mode)

            else:
                break

        return state

    def _use_heuristic(self) -> bool:
        """True when this opponent turn should use the heuristic agent."""
        return self._opponent is None or self._rng.random() < self._heuristic_mix

    def _opponent_bid(self, state: GameState, player_index: int) -> int:
        if self._use_heuristic():
            return self._heuristic.bid(state, player_index)
        action = self._opponent_act(state, player_index)
        return max(0, min(int(action), state.round_number))

    def _opponent_play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        if self._use_heuristic():
            return self._heuristic.play(state, player_index)

        action = self._opponent_act(state, player_index)
        card, mode = self._decode_play_action(int(action))
        # Fall back to heuristic if the decoded action is illegal (early training).
        hand = list(state.player_states[player_index].hand)
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        if card not in hand or card not in legal:
            return self._heuristic.play(state, player_index)
        return card, mode

    def _opponent_act(self, state: GameState, player_index: int) -> int:
        completed = self._engine.completed_tricks_this_round
        obs = self._build_observation_for(state, player_index, completed)
        mask = self._action_masks_for(state, player_index)
        action, _ = self._opponent.predict(
            obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=False
        )
        return int(action[0])


# ---------------------------------------------------------------------------
# Tournament-compatible wrapper around a trained SB3 model
# ---------------------------------------------------------------------------


class SB3AgentWrapper(BaseAgent):
    """Wraps a trained ``MaskablePPO`` model as a ``BaseAgent`` for use in
    the tournament runner.

    Parameters
    ----------
    model:
        A trained ``MaskablePPO`` (or any SB3 model with ``predict``).
    n_players:
        Player count of the game being evaluated.
    name:
        Display name in tournament results.
    deterministic:
        If True, use greedy (argmax) policy; False samples from distribution.
    """

    def __init__(
        self,
        model: Any,
        n_players: int,
        name: str = "PPO",
        deterministic: bool = True,
    ) -> None:
        self._model = model
        self._n_players = n_players
        self.name = name
        self._deterministic = deterministic
        # Utility env — only used for obs/mask building, never stepped.
        self._util_env = SkullKingEnv(n_players=n_players)
        self._engine: Optional["GameEngine"] = None

    def before_move(self, engine: "GameEngine") -> None:
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
        action, _ = self._model.predict(
            obs[np.newaxis], action_masks=mask[np.newaxis],
            deterministic=self._deterministic,
        )
        return int(action[0])
