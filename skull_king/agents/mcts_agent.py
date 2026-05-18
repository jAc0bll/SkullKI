"""MCTS agent with proper determinization for imperfect information.

At each decision point the agent:
  1. Builds the pool of cards it cannot see (not in its own hand, not publicly
     played in completed or current trick).
  2. For every rollout, randomly assigns those unknown cards to opponents
     according to their hand sizes.  This is one *determinization* — a single
     sample from the distribution of possible worlds.
  3. Runs a random rollout on that determinized state.
  4. Picks the action with the highest average outcome over all rollouts.

This is Perfect-Information Monte Carlo (PIMC / determinization-MCTS), the
standard approach for trick-taking card games.  The key difference from a
naive deepcopy approach: the agent never uses the actual opponent hands from
the engine object — it only knows what a real player would know.

Duplicate special cards (5 Escapes, 5 Pirates, 2 Mermaids) are handled with
a Counter (multiset) so deduplication does not shrink the pool incorrectly.

Performance: n_simulations=20 takes ~1–3 s per episode.  Use 5 for benchmarks.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Optional

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, CardType, TigressMode, build_deck
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase, GameState
from skull_king.resolver import TrickResolver

# Full deck as a list preserving duplicates (5 Escapes, 5 Pirates, 2 Mermaids, …)
_FULL_DECK: list[Card] = build_deck()


class MCTSAgent(BaseAgent):
    """Determinization-MCTS agent for imperfect-information Skull King.

    Parameters
    ----------
    n_simulations:
        Number of random rollouts per candidate action.
    seed:
        RNG seed for reproducibility.
    """

    name = "MCTS"

    def __init__(self, n_simulations: int = 20, seed: int = 0) -> None:
        self._n_sim = n_simulations
        self._rng = random.Random(seed)
        self._engine_snapshot: Optional[GameEngine] = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def before_move(self, engine: GameEngine) -> None:
        self._engine_snapshot = copy.deepcopy(engine)

    def bid(self, state: GameState, player_index: int) -> int:
        if self._engine_snapshot is None:
            return self._rng.randint(0, state.round_number)

        best_bid, best_avg = 0, float("-inf")
        for bid_value in range(state.round_number + 1):
            total = 0
            for _ in range(self._n_sim):
                eng = self._determinize(self._engine_snapshot, player_index)
                eng.place_bid(player_index, bid_value)
                total += self._rollout(eng, player_index)
            avg = total / self._n_sim
            if avg > best_avg:
                best_avg = avg
                best_bid = bid_value
        return best_bid

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        hand = list(state.player_states[player_index].hand)
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        candidates = _deduplicated_candidates(legal)

        if self._engine_snapshot is None or len(candidates) == 1:
            return candidates[0]

        best_card: Card = candidates[0][0]
        best_mode: Optional[TigressMode] = candidates[0][1]
        best_avg = float("-inf")

        for card, mode in candidates:
            total = 0
            for _ in range(self._n_sim):
                eng = self._determinize(self._engine_snapshot, player_index)
                eng.play_card(player_index, card, mode)
                total += self._rollout(eng, player_index)
            avg = total / self._n_sim
            if avg > best_avg:
                best_avg = avg
                best_card, best_mode = card, mode

        return best_card, best_mode

    # ------------------------------------------------------------------
    # Determinization
    # ------------------------------------------------------------------

    def _determinize(self, engine: GameEngine, player_index: int) -> GameEngine:
        """Return a deepcopy of *engine* with opponent hands randomly re-dealt.

        Only cards that are invisible to *player_index* (not in their hand,
        not played in the current or completed tricks this round) are eligible
        to be redistributed.  The resulting state is consistent with the real
        game state from the agent's perspective.
        """
        state = engine.get_state()

        # ── Cards the agent can observe ──────────────────────────────────
        # Start with the full deck as a multiset and subtract visible cards.
        pool: Counter[Card] = Counter(_FULL_DECK)

        for card in state.player_states[player_index].hand:
            pool[card] -= 1

        for pc in state.current_trick_cards:
            pool[pc.card] -= 1

        for trick in engine.completed_tricks_this_round:
            for pc in trick.played_cards:
                pool[pc.card] -= 1

        # Build the unknown-card list (multiset → flat list)
        unknown: list[Card] = []
        for card, count in pool.items():
            if count > 0:
                unknown.extend([card] * count)

        self._rng.shuffle(unknown)

        # ── Re-deal unknown cards to opponents ───────────────────────────
        eng = copy.deepcopy(engine)
        idx = 0
        for i, ps in enumerate(state.player_states):
            if i == player_index:
                continue
            n = len(ps.hand)
            # Fallback: if pool is exhausted (shouldn't happen), leave hand as-is
            if idx + n <= len(unknown):
                eng._players[i].set_hand(unknown[idx : idx + n])
            idx += n

        return eng

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, engine: GameEngine, player_index: int) -> int:
        """Play out the rest of the game randomly; return the agent's final score."""
        state = engine.get_state()
        while state.phase != GamePhase.GAME_OVER:
            if state.phase == GamePhase.BIDDING:
                cur = state.current_player_index
                bid = self._rng.randint(0, state.round_number)
                state = engine.place_bid(cur, bid)
            else:
                cur = state.current_player_index
                hand = list(state.player_states[cur].hand)
                legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
                card = self._rng.choice(legal)
                mode: Optional[TigressMode] = None
                if card.card_type == CardType.TIGRESS:
                    mode = self._rng.choice([TigressMode.PIRATE, TigressMode.ESCAPE])
                state = engine.play_card(cur, card, mode)
        return state.player_states[player_index].total_score


def _deduplicated_candidates(
    legal: list[Card],
) -> list[tuple[Card, Optional[TigressMode]]]:
    seen_tigress = False
    result: list[tuple[Card, Optional[TigressMode]]] = []
    for card in dict.fromkeys(legal):
        if card.card_type == CardType.TIGRESS:
            if not seen_tigress:
                result.append((card, TigressMode.PIRATE))
                result.append((card, TigressMode.ESCAPE))
                seen_tigress = True
        else:
            result.append((card, None))
    return result
