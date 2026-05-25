"""DouZero eval agent — greedy over the Q-network."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GameState
from skull_king.training.rebel.public_belief_state import PublicBeliefState
from skull_king.training.rebel.subgame import _action_to_card, _build_action_mask


class DouZeroAgent(BaseAgent):
    """Greedy Q-network agent for tournament evaluation."""

    def __init__(
        self,
        q_net,
        n_players: int = 4,
        name: str = "DouZero",
        device: torch.device | None = None,
    ) -> None:
        self.q_net = q_net
        self.n_players = n_players
        self._name = name
        self.device = device or next(q_net.parameters()).device
        self.q_net.eval()
        self._engine = None

    @property
    def name(self) -> str:
        return self._name

    def before_move(self, engine) -> None:
        self._engine = engine

    def bid(self, state: GameState, player_index: int) -> int:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for b in range(state.round_number + 1):
            mask[b] = True
        return self._argmax(player_index, mask)

    def play(self, state: GameState, player_index: int) -> tuple[Card, Optional[TigressMode]]:
        mask = _build_action_mask(self._engine)
        a = self._argmax(player_index, mask)
        return _action_to_card(a, self._engine)

    def _argmax(self, player_index: int, mask: np.ndarray) -> int:
        pbs = PublicBeliefState.from_engine(self._engine, player_index)
        enc = torch.from_numpy(pbs.encode()).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(enc, mask_t).squeeze(0).cpu().numpy()
        legal = np.where(mask)[0]
        return int(legal[np.argmax(q[legal])])
