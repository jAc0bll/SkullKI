"""AlphaZero tournament agent — runs MCTS at each decision.

Uses ``run_batched_mcts`` with a single-game batch (B=1). Greedy over
visit counts (no temperature, no Dirichlet noise).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GameState
from skull_king.training.alphazero.mcts import _legal_mask, run_batched_mcts
from skull_king.training.rebel.public_belief_state import PublicBeliefState
from skull_king.training.rebel.subgame import _action_to_card


class AlphaZeroAgent(BaseAgent):
    """Tournament agent that runs MCTS for every decision."""

    def __init__(
        self,
        network,
        n_players: int = 4,
        name: str = "AlphaZero",
        device: torch.device | None = None,
        n_simulations: int = 50,
        c_puct: float = 2.0,
        use_mcts: bool = True,
    ) -> None:
        self.network = network
        self.n_players = n_players
        self._name = name
        self.device = device or next(network.parameters()).device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.use_mcts = use_mcts        # if False, skip MCTS and play raw policy
        self.network.eval()
        self._engine = None
        self._rng = np.random.default_rng(0)

    @property
    def name(self) -> str:
        return self._name

    def before_move(self, engine) -> None:
        self._engine = engine

    def bid(self, state: GameState, player_index: int) -> int:
        return self._mcts_pick(player_index)

    def play(self, state: GameState, player_index: int) -> tuple[Card, Optional[TigressMode]]:
        action = self._mcts_pick(player_index)
        return _action_to_card(action, self._engine)

    def _mcts_pick(self, player_index: int) -> int:
        mask = _legal_mask(self._engine)
        legal = np.where(mask)[0]
        if len(legal) == 1:
            return int(legal[0])

        # Fast path: raw policy argmax — no MCTS overhead, batch=1 forward only.
        if not self.use_mcts:
            from skull_king.training.rebel.public_belief_state import PublicBeliefState
            pbs = PublicBeliefState.from_engine(self._engine, player_index)
            enc = torch.from_numpy(pbs.encode()).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_probs, _ = self.network(enc, mask_t)
            scores = log_probs.squeeze(0).cpu().numpy()
            return int(legal[np.argmax(scores[legal])])

        # Full MCTS — slower but stronger
        policies, _ = run_batched_mcts(
            engines_root=[self._engine],
            network=self.network,
            n_simulations=self.n_simulations,
            device=self.device,
            c_puct=self.c_puct,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
            rng=self._rng,
            add_root_noise=False,
            agent_seat=player_index,
        )
        pi = policies[0]
        return int(legal[np.argmax(pi[legal])])
