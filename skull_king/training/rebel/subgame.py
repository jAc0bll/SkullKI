"""ReBeL subgame solver — proper PBS-CFR (single belief-weighted tree).

Algorithm (PBS-CFR, as in ReBeL / Meta AI):
  For each CFR iteration and each traversing player:
    traverse(engine, traversing_player, beliefs, reach_opp=1.0, depth=0)

  At TRAVERSING player nodes:
    - Enumerate all legal actions.
    - Recurse on each, updating beliefs via Bayesian inference.
    - Regret update weighted by reach_opp (opponent reach probability).
    - regrets[a] += reach_opp * (action_value[a] - node_value)

  At OPPONENT nodes:
    - Sample one action from current strategy.
    - Update reach_opp *= strategy[sampled_action].
    - Beliefs updated by Bayesian inference on the sampled play.

  Leaf nodes (end of round or depth > max_depth):
    - Use value_net(pbs_encoding_with_beliefs) for mid-game leaves.
    - Use actual terminal utilities at GAME_OVER.

This is the correct algorithm: one tree, beliefs tracked as a matrix
[n_players, DECK_TOTAL], regrets weighted by opponent reach probability.
No K-fold determinization, no separate per-sample CFRs.
"""
from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from skull_king.cards import Card, TigressMode, DECK_TOTAL
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
# PBS-CFR core
# ---------------------------------------------------------------------------

def _bayesian_update(
    beliefs: np.ndarray,
    actor: int,
    action: int,
    engine: GameEngine,
    n_players: int,
) -> np.ndarray:
    """Update belief matrix after observing *actor* play *action*.

    During PLAYING phase a card play is a public observation: the card
    is removed from all players' belief distributions and rows renormalized.
    During BIDDING there is no card information to infer, so beliefs are
    returned unchanged.

    Parameters
    ----------
    beliefs:   [n_players, DECK_TOTAL] float32 — current beliefs
    actor:     player index who just acted
    action:    action index applied
    engine:    game state BEFORE the action was applied (used for phase check)
    n_players: number of players

    Returns
    -------
    new_beliefs: [n_players, DECK_TOTAL] float32 (copy, never modifies in-place)
    """
    new_beliefs = beliefs.copy()

    if engine._phase == GamePhase.PLAYING:
        try:
            card, _ = _action_to_card(action, engine)
            slot = _card_to_slot(card)
            if slot >= 0:
                # Card is now publicly known — remove from all belief rows
                new_beliefs[:, slot] = 0.0
                # Renormalize each player's row independently
                for i in range(n_players):
                    row_sum = new_beliefs[i].sum()
                    if row_sum > 0.0:
                        new_beliefs[i] /= row_sum
        except Exception:
            pass  # If card lookup fails, leave beliefs unchanged

    return new_beliefs


class _PBSCFRTree:
    """Single PBS-CFR tree for one subgame solve.

    State (regret/strategy sums) is keyed by (player, round, trick_in_round,
    hand_tuple) — the same observable information used in the old solver.
    """

    def __init__(
        self,
        n_players: int,
        value_net: Optional["RebelValueNet"],
        device: torch.device,
        current_round: int,
        max_depth: Optional[int],
    ) -> None:
        self.n_players = n_players
        self.value_net = value_net
        self.device = device
        self.current_round = current_round
        self.max_depth = max_depth

        # Keyed by obs_key = (player, round, trick_in_round, hand_tuple)
        self._regret_sum: dict[tuple, np.ndarray] = {}
        self._strategy_sum: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Node key + strategy
    # ------------------------------------------------------------------

    def _obs_key(self, engine: GameEngine) -> tuple:
        player = engine._current_player_index()
        hand_key = tuple(sorted(c._hash for c in engine._players[player].hand))
        return (player, engine._round, engine._trick_in_round, hand_key)

    def _ensure_key(self, key: tuple) -> None:
        if key not in self._regret_sum:
            self._regret_sum[key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)

    def _current_strategy(self, key: tuple, mask: np.ndarray) -> np.ndarray:
        self._ensure_key(key)
        return regret_match(self._regret_sum[key].copy(), mask)

    # ------------------------------------------------------------------
    # Leaf detection + evaluation
    # ------------------------------------------------------------------

    def _is_leaf(self, engine: GameEngine, depth: int) -> bool:
        if engine._phase == GamePhase.GAME_OVER:
            return True
        if engine._round > self.current_round:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _leaf_value(
        self,
        engine: GameEngine,
        beliefs: np.ndarray,
    ) -> np.ndarray:
        """Return utility vector [n_players] at a leaf node."""
        if engine._phase == GamePhase.GAME_OVER:
            scores = [p.total_score for p in engine._players]
            return np.array(
                [_utility_from_scores(scores, i) for i in range(self.n_players)],
                dtype=np.float32,
            )

        # End of round but game continues — use value network if available
        if self.value_net is not None:
            # Build a minimal PBS encoding using current beliefs
            enc = self._pbs_encode_with_beliefs(engine, beliefs)
            with torch.no_grad():
                t = torch.from_numpy(enc).float().unsqueeze(0).to(self.device)
                vals = self.value_net(t).squeeze(0).cpu().numpy()
            return vals.astype(np.float32)

        # Fallback: use accumulated scores
        scores = [p.total_score for p in engine._players]
        return np.array(
            [_utility_from_scores(scores, i) for i in range(self.n_players)],
            dtype=np.float32,
        )

    def _pbs_encode_with_beliefs(
        self,
        engine: GameEngine,
        beliefs: np.ndarray,
    ) -> np.ndarray:
        """Encode game state + current belief matrix as a float32 vector.

        Mirrors PublicBeliefState.encode() but injects the live belief matrix
        rather than a reconstructed one from from_engine().
        """
        from skull_king.cards import NUM_ROUNDS
        n = self.n_players
        phase_map = {GamePhase.BIDDING: 0, GamePhase.PLAYING: 1, GamePhase.GAME_OVER: 2}
        phase_val = phase_map.get(engine._phase, 2)

        bids_arr = np.full(n, -1, dtype=np.float32)
        for i in engine._bids_placed:
            bids_arr[i] = engine._players[i].bid / NUM_ROUNDS

        tricks_won = np.array(
            [p.tricks_won_this_round for p in engine._players], dtype=np.float32
        ) / max(engine._round, 1)

        total_scores = np.clip(
            np.array([p.total_score for p in engine._players], dtype=np.float32) / 100.0,
            -2.0, 2.0,
        )

        bid_revealed = np.array(
            [float(i in engine._bids_placed) for i in range(n)], dtype=np.float32
        )

        leader_one_hot = np.zeros(n, dtype=np.float32)
        leader_one_hot[engine._trick_leader] = 1.0

        seen_mask = np.zeros(DECK_TOTAL, dtype=np.float32)
        for trick in engine._completed_tricks:
            for pc in trick.played_cards:
                slot = _card_to_slot(pc.card)
                if slot >= 0:
                    seen_mask[slot] = 1.0

        curr_mask = np.zeros(DECK_TOTAL, dtype=np.float32)
        for pc in engine._current_trick.played_cards:
            slot = _card_to_slot(pc.card)
            if slot >= 0:
                curr_mask[slot] = 1.0

        parts = [
            np.array([
                engine._round / NUM_ROUNDS,
                engine._trick_in_round / max(engine._round, 1),
                float(phase_val) / 2.0,
                engine._current_player_index() / n,
            ], dtype=np.float32),
            np.where(bids_arr >= 0, bids_arr, -1.0).astype(np.float32),
            tricks_won,
            total_scores,
            bid_revealed,
            leader_one_hot,
            seen_mask,
            curr_mask,
            beliefs.flatten().astype(np.float32),
        ]
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Engine helpers
    # ------------------------------------------------------------------

    def _clone_and_apply(
        self, engine: GameEngine, player: int, action: int
    ) -> GameEngine:
        eng2 = copy.deepcopy(engine)
        if eng2._phase == GamePhase.BIDDING:
            eng2.place_bid_no_state(player, action)
        else:
            card, tigress_mode = _action_to_card(action, eng2)
            eng2.play_card_no_state(player, card, tigress_mode)
        return eng2

    # ------------------------------------------------------------------
    # PBS-CFR traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        engine: GameEngine,
        traversing_player: int,
        beliefs: np.ndarray,
        reach_opp: float,
        depth: int,
    ) -> np.ndarray:
        """Belief-weighted CFR traversal.

        Returns
        -------
        np.ndarray [n_players] — utility vector at this node under current strategy
        """
        if self._is_leaf(engine, depth):
            return self._leaf_value(engine, beliefs)

        player = engine._current_player_index()
        mask = _build_action_mask(engine)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return self._leaf_value(engine, beliefs)

        key = self._obs_key(engine)
        strategy = self._current_strategy(key, mask)

        if player == traversing_player:
            # --- Traversing player node ---
            # Compute counterfactual value for each legal action
            action_utils: dict[int, np.ndarray] = {}
            for a in legal:
                new_beliefs = _bayesian_update(beliefs, player, a, engine, self.n_players)
                eng2 = self._clone_and_apply(engine, player, a)
                action_utils[a] = self.traverse(
                    eng2, traversing_player, new_beliefs, reach_opp, depth + 1
                )

            # Node value = strategy-weighted sum of traversing player's utilities
            node_util = sum(
                strategy[a] * action_utils[a][traversing_player] for a in legal
            )

            # Regret update: weighted by opponent reach probability
            self._ensure_key(key)
            for a in legal:
                self._regret_sum[key][a] += reach_opp * (
                    action_utils[a][traversing_player] - node_util
                )
            # Strategy sum: also weighted by reach_opp (importance sampling)
            self._strategy_sum[key] += reach_opp * strategy

            # Return strategy-weighted average utility across ALL players
            result = np.zeros(self.n_players, dtype=np.float32)
            for a in legal:
                result += strategy[a] * action_utils[a]
            return result

        else:
            # --- Opponent node (external sampling) ---
            # Sample one action, update reach probability
            probs = strategy[legal]
            probs = probs / probs.sum()
            sampled_idx = np.random.choice(len(legal), p=probs)
            a = legal[sampled_idx]

            new_beliefs = _bayesian_update(beliefs, player, a, engine, self.n_players)
            eng2 = self._clone_and_apply(engine, player, a)

            # Accumulate strategy sum (no reach weighting for opponent nodes
            # in external sampling CFR)
            self._ensure_key(key)
            self._strategy_sum[key] += strategy

            # Pass updated reach: reach_opp *= prob of sampled action
            return self.traverse(
                eng2, traversing_player, new_beliefs,
                reach_opp * strategy[a], depth + 1
            )

    # ------------------------------------------------------------------
    # Extract average strategy
    # ------------------------------------------------------------------

    def get_average_strategy(self, key: tuple, mask: np.ndarray) -> np.ndarray:
        """Return the time-averaged strategy at a node."""
        if key not in self._strategy_sum:
            legal = np.where(mask)[0]
            strat = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            if len(legal):
                strat[legal] = 1.0 / len(legal)
            return strat
        total = self._strategy_sum[key].sum()
        if total <= 0:
            legal = np.where(mask)[0]
            strat = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            if len(legal):
                strat[legal] = 1.0 / len(legal)
            return strat
        return (self._strategy_sum[key] / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Public subgame solver (PBS-CFR / ReBeL)
# ---------------------------------------------------------------------------

class SubgameSolver:
    """Proper PBS-CFR subgame solver (ReBeL algorithm).

    Runs a single belief-weighted CFR tree per call to ``solve``.
    No K-fold determinization — beliefs are tracked as a probability
    matrix [n_players, DECK_TOTAL] and updated via Bayesian inference
    after each observed card play.

    Regrets are weighted by the opponent reach probability, giving
    counterfactual regret estimates that account for how likely the
    opponent would reach this node under the current strategy profile.
    """

    def __init__(
        self,
        value_net: Optional["RebelValueNet"],
        device: torch.device,
        n_cfr_iters: int = 50,
        max_depth: Optional[int] = None,
        # Legacy parameter kept for API compatibility with trainer.py
        n_samples: int = 1,
    ) -> None:
        self.value_net = value_net
        self.device = device
        self.n_cfr_iters = n_cfr_iters
        self.max_depth = max_depth

    def solve(
        self,
        engine: GameEngine,
        pbs: PublicBeliefState,
        acting_player: int,
    ) -> dict:
        """Run PBS-CFR from the current game position.

        Parameters
        ----------
        engine:        current game state
        pbs:           public belief state (provides initial belief matrix)
        acting_player: index of the player to act (used for key construction)

        Returns
        -------
        dict with keys:
          "strategy":  np.ndarray [ACTION_SPACE_SIZE] — average strategy
          "mask":      np.ndarray [ACTION_SPACE_SIZE] — legal action mask
          "values":    np.ndarray [n_players] — estimated utilities
          "pbs_enc":   np.ndarray — encoded PBS for network training
        """
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

        # Initial beliefs from PBS
        beliefs = pbs.belief.copy()  # [n_players, DECK_TOTAL]

        cfr = _PBSCFRTree(
            n_players=n,
            value_net=self.value_net,
            device=self.device,
            current_round=engine._round,
            max_depth=self.max_depth,
        )

        # Run n_cfr_iters iterations, alternating traversing player
        root_values = np.zeros(n, dtype=np.float64)
        n_value_samples = 0

        for iteration in range(self.n_cfr_iters):
            for traversing_player in range(n):
                vals = cfr.traverse(
                    engine=copy.deepcopy(engine),
                    traversing_player=traversing_player,
                    beliefs=beliefs.copy(),
                    reach_opp=1.0,
                    depth=0,
                )
                if traversing_player == acting_player:
                    root_values += vals
                    n_value_samples += 1

        # Extract average strategy at the root node for the acting player
        hand_key = tuple(sorted(c._hash for c in engine._players[acting_player].hand))
        root_key = (acting_player, engine._round, engine._trick_in_round, hand_key)
        avg_strategy = cfr.get_average_strategy(root_key, mask)

        # Renormalize over legal actions
        legal_sum = avg_strategy[legal].sum()
        if legal_sum > 0:
            avg_strategy[legal] /= legal_sum
        else:
            avg_strategy[legal] = 1.0 / len(legal)

        estimated_values = (
            root_values / n_value_samples if n_value_samples > 0
            else np.zeros(n, dtype=np.float32)
        ).astype(np.float32)

        return {
            "strategy": avg_strategy,
            "mask": mask,
            "values": estimated_values,
            "pbs_enc": pbs.encode(),
        }
