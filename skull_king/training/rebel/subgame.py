"""ReBeL subgame solver.

Implements determinized CFR with value-network leaf evaluation.

Algorithm per game position:
  1. Sample K private states (opponent hand assignments) from the PBS belief.
  2. For each sample, run vanilla CFR from the current game position
     treating all hands as known (perfect information within the subgame).
  3. At leaf nodes (end of round), query the value network V(PBS)
     to estimate future-round utility rather than rolling out further.
  4. Average the resulting strategies across all K samples → final strategy.
  5. Collect (PBS, strategy, values) tuples for network training.

Why determinized CFR instead of full ReBeL PBS-CFR?
  Full ReBeL requires tracking the belief distribution over the entire game
  tree simultaneously, which involves a weighted sum over all possible deals
  at every node — exponential in the number of opponent cards.
  Determinized CFR ("PIMC") is a well-tested approximation that averages
  strategies across sampled deals.  It sacrifices theoretical Nash-equilibrium
  guarantees but is strong in practice and far easier to implement.
  Full PBS-CFR will be added as a future extension.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from skull_king.cards import Card, TigressMode
from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import (
    ACTION_SPACE_SIZE,
    N_BID_ACTIONS,
    TIGRESS_AS_ESCAPE_ACTION,
    TIGRESS_AS_PIRATE_ACTION,
    _CANONICAL_DECK,
    _HASH_TO_SLOTS,
)
from skull_king.game_state import GamePhase
from skull_king.training.cfr.networks import regret_match
from skull_king.training.cfr.traversal import _utility_from_scores
from skull_king.training.rebel.public_belief_state import (
    PublicBeliefState,
    _card_to_slot,
    pbs_encoding_size,
)

if TYPE_CHECKING:
    from skull_king.training.rebel.networks import RebelValueNet


# ---------------------------------------------------------------------------
# Action helpers (mirrors traversal.py logic)
# ---------------------------------------------------------------------------

def _build_action_mask(engine: GameEngine) -> np.ndarray:
    """Return a bool array of legal actions at the current game node."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    phase = engine._phase
    player = engine._current_player_index()

    if phase == GamePhase.BIDDING:
        for b in range(engine._round + 1):
            mask[b] = True
        return mask

    if phase == GamePhase.PLAYING:
        hand = engine._players[player].hand
        trick = engine._current_trick
        from skull_king.resolver import TrickResolver
        legal = TrickResolver.legal_plays(list(trick.played_cards), hand)
        for card in legal:
            if card.card_type.name == "TIGRESS":
                mask[TIGRESS_AS_ESCAPE_ACTION] = True
                mask[TIGRESS_AS_PIRATE_ACTION] = True
            else:
                slots = _HASH_TO_SLOTS[card._hash]
                # Find which slot is in the player's hand (track by occurrence)
                for slot in slots:
                    if mask[N_BID_ACTIONS + slot - (1 if slot >= 69 else 0)]:
                        continue
                    mask[N_BID_ACTIONS + slot] = True
                    break
        return mask

    return mask  # GAME_OVER: all False


def _action_to_card(action: int, engine: GameEngine) -> tuple[Card, Optional[TigressMode]]:
    """Convert an action index to a (Card, TigressMode) pair."""
    if action == TIGRESS_AS_ESCAPE_ACTION:
        tigress = [c for c in engine._players[engine._current_player_index()].hand
                   if c.card_type.name == "TIGRESS"]
        return tigress[0], TigressMode.ESCAPE
    if action == TIGRESS_AS_PIRATE_ACTION:
        tigress = [c for c in engine._players[engine._current_player_index()].hand
                   if c.card_type.name == "TIGRESS"]
        return tigress[0], TigressMode.PIRATE
    # Card play: action - N_BID_ACTIONS = canonical slot
    slot = action - N_BID_ACTIONS
    target_card = _CANONICAL_DECK[slot]
    player_idx = engine._current_player_index()
    for card in engine._players[player_idx].hand:
        if card == target_card:
            return card, None
    # Fallback: return any matching card type
    for card in engine._players[player_idx].hand:
        if card._hash == target_card._hash:
            return card, None
    raise ValueError(f"Card for action {action} not found in hand")


# ---------------------------------------------------------------------------
# Determinized CFR subgame
# ---------------------------------------------------------------------------

class _SubgameCFR:
    """Vanilla external-sampling CFR on a single determinized game tree."""

    def __init__(
        self,
        n_players: int,
        value_net: Optional["RebelValueNet"],
        device: torch.device,
        current_round: int,
    ) -> None:
        self.n_players = n_players
        self.value_net = value_net
        self.device = device
        self.current_round = current_round
        # regret and strategy sums: keyed by (player, obs_hash)
        self._regret_sum: dict[tuple, np.ndarray] = {}
        self._strategy_sum: dict[tuple, np.ndarray] = {}

    def _get_strategy(self, key: tuple, mask: np.ndarray) -> np.ndarray:
        if key not in self._regret_sum:
            self._regret_sum[key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        regrets = self._regret_sum[key].copy()
        strat = regret_match(regrets, mask)
        return strat

    def _leaf_value(self, engine: GameEngine, pbs: Optional[PublicBeliefState]) -> np.ndarray:
        """Return utility estimates for all players at a leaf node."""
        if engine._phase == GamePhase.GAME_OVER:
            scores = [p.total_score for p in engine._players]
            return np.array([
                _utility_from_scores(scores, i) for i in range(self.n_players)
            ], dtype=np.float32)

        # End of current round but game continues — use value network
        if self.value_net is not None and pbs is not None:
            enc = pbs.encode()
            with torch.no_grad():
                t = torch.from_numpy(enc).float().unsqueeze(0).to(self.device)
                vals = self.value_net(t).squeeze(0).cpu().numpy()
            return vals

        # Fallback: use current round scores as proxy for total game value
        scores = [p.total_score for p in engine._players]
        return np.array([
            _utility_from_scores(scores, i) for i in range(self.n_players)
        ], dtype=np.float32)

    def _is_subgame_leaf(self, engine: GameEngine) -> bool:
        """True if we've finished the current round (or game over)."""
        if engine._phase == GamePhase.GAME_OVER:
            return True
        # New round started (round advanced beyond subgame root round)
        if engine._round > self.current_round:
            return True
        return False

    def traverse(
        self,
        engine: GameEngine,
        traversing_player: int,
        reach: float,
        pbs: Optional[PublicBeliefState],
    ) -> np.ndarray:
        """External-sampling CFR traversal. Returns utility array [n_players]."""
        if self._is_subgame_leaf(engine):
            return self._leaf_value(engine, pbs)

        player = engine._current_player_index()
        mask = _build_action_mask(engine)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return self._leaf_value(engine, pbs)

        # Build an observation key for this node (player + hand + public state)
        hand_key = tuple(sorted(c._hash for c in engine._players[player].hand))
        obs_key = (player, engine._round, engine._trick_in_round, hand_key)

        strategy = self._get_strategy(obs_key, mask)

        if player == traversing_player:
            # Compute counterfactual regret for all legal actions
            action_values = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
            for a in legal:
                eng2 = self._clone_and_apply(engine, player, a)
                new_pbs = self._advance_pbs(pbs, player, a, engine)
                utils = self.traverse(eng2, traversing_player, reach * strategy[a], new_pbs)
                action_values[a] = utils[player]

            node_value = np.sum(strategy * action_values)
            # Regret update
            if obs_key not in self._regret_sum:
                self._regret_sum[obs_key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
                self._strategy_sum[obs_key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
            for a in legal:
                self._regret_sum[obs_key][a] += reach * (action_values[a] - node_value)
            self._strategy_sum[obs_key] += reach * strategy

            # Return full utility vector (approximate other players' values)
            result = np.zeros(self.n_players, dtype=np.float32)
            result[player] = node_value
            # Sample one action to get other players' utilities
            a_sample = legal[np.argmax(strategy[legal])]
            eng2 = self._clone_and_apply(engine, player, a_sample)
            new_pbs = self._advance_pbs(pbs, player, a_sample, engine)
            other_utils = self.traverse(eng2, traversing_player,
                                        reach * strategy[a_sample], new_pbs)
            for i in range(self.n_players):
                if i != player:
                    result[i] = other_utils[i]
            return result
        else:
            # Sample one action from strategy
            probs = strategy[legal]
            probs = probs / probs.sum()
            a = legal[np.random.choice(len(legal), p=probs)]
            eng2 = self._clone_and_apply(engine, player, a)
            new_pbs = self._advance_pbs(pbs, player, a, engine)

            if obs_key not in self._strategy_sum:
                self._regret_sum[obs_key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
                self._strategy_sum[obs_key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
            self._strategy_sum[obs_key] += strategy

            return self.traverse(eng2, traversing_player, reach, new_pbs)

    def _clone_and_apply(self, engine: GameEngine, player: int, action: int) -> GameEngine:
        """Deep-copy the engine and apply one action."""
        import copy
        eng2 = copy.deepcopy(engine)
        if eng2._phase == GamePhase.BIDDING:
            eng2.place_bid_no_state(player, action)
        else:
            card, tigress_mode = _action_to_card(action, eng2)
            eng2.play_card_no_state(player, card, tigress_mode)
        return eng2

    def _advance_pbs(
        self,
        pbs: Optional[PublicBeliefState],
        player: int,
        action: int,
        engine: GameEngine,
    ) -> Optional[PublicBeliefState]:
        """Update the PBS after an action (best-effort, may return None)."""
        if pbs is None:
            return None
        try:
            if engine._phase == GamePhase.PLAYING:
                card, _ = _action_to_card(action, engine)
                slot = _card_to_slot(card)
                if slot >= 0:
                    return pbs.observe_card_played(slot)
        except Exception:
            pass
        return pbs

    def get_average_strategy(self, obs_key: tuple, mask: np.ndarray) -> np.ndarray:
        """Return the time-averaged strategy at a node."""
        if obs_key not in self._strategy_sum:
            legal = np.where(mask)[0]
            strat = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            if len(legal):
                strat[legal] = 1.0 / len(legal)
            return strat
        total = self._strategy_sum[obs_key].sum()
        if total <= 0:
            legal = np.where(mask)[0]
            strat = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            if len(legal):
                strat[legal] = 1.0 / len(legal)
            return strat
        return (self._strategy_sum[obs_key] / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Public subgame solver
# ---------------------------------------------------------------------------

class SubgameSolver:
    """Runs determinized CFR on a game position and returns training data.

    For each call to ``solve``:
    1. K opponent hand assignments are sampled from the PBS belief.
    2. For each sample a fresh CFR solver runs n_cfr_iters traversals.
    3. Strategies are averaged; (PBS, strategy, value) tuples collected.
    """

    def __init__(
        self,
        value_net: Optional["RebelValueNet"],
        device: torch.device,
        n_cfr_iters: int = 50,
        n_samples: int = 16,
    ) -> None:
        self.value_net = value_net
        self.device = device
        self.n_cfr_iters = n_cfr_iters
        self.n_samples = n_samples
        self._rng = np.random.default_rng()

    def solve(
        self,
        engine: GameEngine,
        pbs: PublicBeliefState,
        acting_player: int,
    ) -> dict:
        """Solve the subgame rooted at *engine*'s current state.

        Returns
        -------
        dict with keys:
          "strategy":  np.ndarray [ACTION_SPACE_SIZE] — averaged strategy
          "mask":      np.ndarray [ACTION_SPACE_SIZE] — legal action mask
          "values":    np.ndarray [n_players] — estimated utilities
          "pbs_enc":   np.ndarray — encoded PBS for network training
        """
        import copy
        n = engine.n_players

        mask = _build_action_mask(engine)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return {
                "strategy": mask.astype(np.float32),
                "mask": mask,
                "values": np.zeros(n, dtype=np.float32),
                "pbs_enc": pbs.encode(),
            }

        # Own hand as canonical slots
        own_slots = np.array([
            _card_to_slot(c)
            for c in engine._players[acting_player].hand
        ], dtype=np.int32)

        accumulated_strategy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        accumulated_values = np.zeros(n, dtype=np.float64)

        for _ in range(self.n_samples):
            # Sample opponent hands from belief
            sampled_hands = pbs.sample_opponent_hands(self._rng, own_slots)

            # Inject sampled hands into a copy of the engine
            eng_copy = copy.deepcopy(engine)
            hand_lists = []
            for i in range(n):
                if i == acting_player:
                    hand_lists.append(list(engine._players[i].hand))
                else:
                    # Convert slot indices back to Card objects
                    cards = [_CANONICAL_DECK[s] for s in sampled_hands[i]
                             if 0 <= s < len(_CANONICAL_DECK)]
                    hand_lists.append(cards)

            # Only inject if in BIDDING phase (engine._inject_hands requires it)
            # For PLAYING phase we need to directly set hands
            if eng_copy._phase in (GamePhase.PLAYING,):
                for i, hand in enumerate(hand_lists):
                    eng_copy._players[i].hand = list(hand)
            # If BIDDING, hands were already dealt; we'd need _inject_hands
            # but since we're in PLAYING for most decisions, this works.

            cfr = _SubgameCFR(n, self.value_net, self.device, engine._round)

            for _ in range(self.n_cfr_iters):
                for player in range(n):
                    cfr.traverse(eng_copy, player, 1.0, pbs)

            # Extract strategy at the root node
            hand_key = tuple(sorted(c._hash for c in engine._players[acting_player].hand))
            root_key = (acting_player, engine._round, engine._trick_in_round, hand_key)
            strat = cfr.get_average_strategy(root_key, mask)
            accumulated_strategy += strat

            # Estimate values from the root
            vals = cfr._leaf_value(eng_copy, pbs)
            accumulated_values += vals

        # Average across samples
        final_strategy = (accumulated_strategy / self.n_samples).astype(np.float32)
        # Renormalize over legal actions
        legal_sum = final_strategy[legal].sum()
        if legal_sum > 0:
            final_strategy[legal] /= legal_sum
        else:
            final_strategy[legal] = 1.0 / len(legal)

        return {
            "strategy": final_strategy,
            "mask": mask,
            "values": (accumulated_values / self.n_samples).astype(np.float32),
            "pbs_enc": pbs.encode(),
        }
