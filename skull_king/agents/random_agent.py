"""Random agent — picks uniformly from legal actions."""
from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

import numpy as np

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, CardType, TigressMode
from skull_king.game_state import GameState
from skull_king.resolver import TrickResolver

if TYPE_CHECKING:
    from skull_king.env.skull_king_env import SkullKingEnv


class RandomAgent(BaseAgent):
    """Bids and plays uniformly at random among legal options."""

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # BaseAgent (tournament) interface
    # ------------------------------------------------------------------

    def bid(self, state: GameState, player_index: int) -> int:
        return self._rng.randint(0, state.round_number)

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        hand = list(state.player_states[player_index].hand)
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        card = self._rng.choice(legal)
        mode: Optional[TigressMode] = None
        if card.card_type == CardType.TIGRESS:
            mode = self._rng.choice([TigressMode.PIRATE, TigressMode.ESCAPE])
        return card, mode

    # ------------------------------------------------------------------
    # Gymnasium env interface (backward-compatible)
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Pick a uniformly random legal action given a boolean action mask."""
        legal = [i for i, ok in enumerate(mask) if ok]
        if not legal:
            raise RuntimeError("No legal actions available")
        return self._rng.choice(legal)

    def run_episode(self, env: "SkullKingEnv") -> dict:
        """Run one full Gymnasium episode; return episode statistics."""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False

        while not terminated:
            mask = env.action_masks()
            action = self.act(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if truncated:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "final_score": info["score"],
        }
