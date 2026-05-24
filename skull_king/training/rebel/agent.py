"""ReBeL inference agent — value-greedy decisions using the trained value network."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, TigressMode
from skull_king.game_state import GameState


class RebelAgent(BaseAgent):
    """Value-greedy agent: picks the action that maximises the value net's
    prediction for the current player.  Falls back to policy net if no
    value net is provided.
    """

    def __init__(
        self,
        policy_net,
        n_players: int = 4,
        name: str = "ReBeL",
        device: torch.device | None = None,
        value_net=None,
    ) -> None:
        self.policy_net = policy_net
        self.value_net = value_net
        self.n_players = n_players
        self._name = name
        self.device = device or next(policy_net.parameters()).device
        self.policy_net.eval()
        if self.value_net is not None:
            self.value_net.eval()
        self._engine = None

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
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for b in range(state.round_number + 1):
            mask[b] = True
        return int(self._best_action(player_index, mask))

    def play(self, state: GameState, player_index: int) -> tuple[Card, Optional[TigressMode]]:
        from skull_king.training.rebel.subgame import _build_action_mask, _action_to_card
        mask = _build_action_mask(self._engine)
        action = self._best_action(player_index, mask)
        return _action_to_card(action, self._engine)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _best_action(self, player_index: int, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0
        if self.value_net is not None:
            return self._value_greedy(player_index, legal)
        return self._policy_greedy(player_index, mask, legal)

    def _value_greedy(self, player_index: int, legal: np.ndarray) -> int:
        """Pick the action with the highest predicted value for player_index.

        Always encode PBS from player_index's perspective — consistent with
        how the value buffer was built during self-play.
        """
        from skull_king.training.rebel.subgame import _action_to_card, _fast_clone_engine
        from skull_king.training.rebel.public_belief_state import PublicBeliefState
        from skull_king.training.cfr.traversal import _utility_from_scores
        from skull_king.game_state import GamePhase

        best_val = float("-inf")
        best_action = int(legal[0])

        for action in legal:
            eng2 = _fast_clone_engine(self._engine)
            try:
                if eng2._phase == GamePhase.BIDDING:
                    eng2.place_bid_no_state(player_index, int(action))
                else:
                    card, tm = _action_to_card(int(action), eng2)
                    eng2.play_card_no_state(player_index, card, tm)
            except Exception:
                continue

            if eng2._phase == GamePhase.GAME_OVER:
                scores = [p.total_score for p in eng2._players]
                val = _utility_from_scores(scores, player_index)
            else:
                # Always encode from player_index's perspective (matches training distribution)
                pbs = PublicBeliefState.from_engine(eng2, player_index)
                enc = torch.from_numpy(pbs.encode()).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    val = self.value_net(enc)[0, player_index].item()

            if val > best_val:
                best_val = val
                best_action = int(action)

        return best_action

    def _policy_greedy(self, player_index: int, mask: np.ndarray, legal: np.ndarray) -> int:
        pbs_enc = self._encode(player_index)
        enc_t = torch.from_numpy(pbs_enc).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.policy_net(enc_t, mask_t)
            probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
        return int(legal[np.argmax(probs[legal])])

    def _encode(self, player_index: int) -> np.ndarray:
        from skull_king.training.rebel.public_belief_state import PublicBeliefState, pbs_encoding_size
        if self._engine is not None:
            return PublicBeliefState.from_engine(self._engine, player_index).encode()
        return np.zeros(pbs_encoding_size(self.n_players), dtype=np.float32)
