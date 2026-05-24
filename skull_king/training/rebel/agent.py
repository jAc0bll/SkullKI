"""ReBeL inference agent — uses the trained policy network for decisions."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.game_state import GameState


class RebelAgent(BaseAgent):
    """Greedy policy-network agent for tournament evaluation."""

    def __init__(
        self,
        policy_net,
        n_players: int = 4,
        name: str = "ReBeL",
        device: torch.device | None = None,
    ) -> None:
        self.policy_net = policy_net
        self.n_players = n_players
        self._name = name
        self.device = device or next(policy_net.parameters()).device
        self.policy_net.eval()
        self._engine = None  # populated by before_move()

    @property
    def name(self) -> str:
        return self._name

    def before_move(self, engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def bid(self, state: GameState, player_index: int) -> int:
        from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
        from skull_king.training.rebel.public_belief_state import PublicBeliefState

        pbs_enc = self._encode(player_index)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for b in range(state.round_number + 1):
            mask[b] = True

        action = self._best_action(pbs_enc, mask)
        return int(action)  # bid action index == bid value

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        from skull_king.training.rebel.subgame import (
            _build_action_mask,
            _action_to_card,
        )

        pbs_enc = self._encode(player_index)
        mask = _build_action_mask(self._engine)
        action = self._best_action(pbs_enc, mask)
        return _action_to_card(action, self._engine)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, player_index: int) -> np.ndarray:
        from skull_king.training.rebel.public_belief_state import (
            PublicBeliefState,
            pbs_encoding_size,
        )
        if self._engine is not None:
            pbs = PublicBeliefState.from_engine(self._engine, player_index)
            return pbs.encode()
        # Fallback: zero vector (should never happen in normal tournament use)
        return np.zeros(pbs_encoding_size(self.n_players), dtype=np.float32)

    def _best_action(self, pbs_enc: np.ndarray, mask: np.ndarray) -> int:
        enc_t = torch.from_numpy(pbs_enc).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.policy_net(enc_t, mask_t)
            probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0
        return int(legal[np.argmax(probs[legal])])
