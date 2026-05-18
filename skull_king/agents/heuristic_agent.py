"""Rule-based heuristic agent for Skull King."""
from __future__ import annotations

from typing import Optional

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, CardType, Suit, TigressMode, TRUMP_SUIT
from skull_king.game_state import GameState
from skull_king.resolver import TrickResolver
from skull_king.trick import PlayedCard


# Expected-trick-win probability per card type / value used for bid estimation.
# These weights are calibrated heuristically rather than theoretically.
_SPECIAL_WEIGHT: dict[CardType, float] = {
    CardType.SKULL_KING: 0.95,
    CardType.PIRATE:     0.75,
    CardType.TIGRESS:    0.45,  # average of PIRATE (0.75) and ESCAPE (0.0) modes
    CardType.MERMAID:    0.25,  # only wins when SK is present and no Pirate
    CardType.ESCAPE:     0.00,
}


def _card_strength(card: Card, mode: Optional[TigressMode] = None) -> int:
    """Numeric rank for sorting candidates.  Higher = stronger."""
    if card.card_type == CardType.SKULL_KING:
        return 10_000
    if card.card_type == CardType.PIRATE:
        return 9_000
    if card.card_type == CardType.TIGRESS:
        return 9_000 if mode == TigressMode.PIRATE else 0
    if card.card_type == CardType.MERMAID:
        return 5_000
    if card.card_type == CardType.ESCAPE:
        return 0
    # Numbered cards
    assert card.value is not None
    return (1_000 if card.suit == TRUMP_SUIT else 0) + card.value * 10


def _candidates(legal: list[Card]) -> list[tuple[Card, Optional[TigressMode]]]:
    """Deduplicated (card, mode) pairs for all legal plays."""
    seen_tigress = False
    result: list[tuple[Card, Optional[TigressMode]]] = []
    for card in dict.fromkeys(legal):  # preserves order, removes duplicates
        if card.card_type == CardType.TIGRESS:
            if not seen_tigress:
                result.append((card, TigressMode.PIRATE))
                result.append((card, TigressMode.ESCAPE))
                seen_tigress = True
        else:
            result.append((card, None))
    return result


def _would_beat(
    card: Card,
    played: tuple[PlayedCard, ...],
    player_index: int,
    mode: Optional[TigressMode],
) -> bool:
    """True if *card* would currently win the trick against the played cards so far."""
    candidate = PlayedCard(
        card=card,
        player_index=player_index,
        play_order=len(played) + 1,
        tigress_mode=mode,
    )
    result = TrickResolver.resolve(list(played) + [candidate])
    return result.winner_player_index == player_index


class HeuristicAgent(BaseAgent):
    """Rule-based agent with three layers of logic.

    Bidding
    -------
    Sums expected-win weights for each card in hand; rounds to the nearest
    integer and clamps to ``[0, round_number]``.

    Playing to win  (``tricks_won < bid``)
    ---------------------------------------
    When following: plays the *weakest* card that beats the current trick
    leader (conserve strong cards).  When leading: plays the *strongest* card.

    Playing to lose  (``bid == 0`` or ``tricks_won >= bid``)
    ----------------------------------------------------------
    When following: plays the *weakest* card that does *not* beat the current
    leader.  When leading: plays an Escape first, then the weakest non-special.

    Tigress is declared PIRATE when trying to win and ESCAPE when trying to
    lose; it is evaluated accordingly in ``_would_beat``.
    """

    name = "Heuristic"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def bid(self, state: GameState, player_index: int) -> int:
        hand = list(state.player_states[player_index].hand)
        weight = 0.0
        for card in hand:
            if card.card_type == CardType.NUMBERED:
                assert card.value is not None
                if card.suit == TRUMP_SUIT:
                    weight += 0.15 + (card.value / 14) * 0.60   # 0.15 – 0.75
                else:
                    weight += (card.value / 14) * 0.15           # 0.00 – 0.15
            else:
                weight += _SPECIAL_WEIGHT[card.card_type]
        bid = round(weight)
        return max(0, min(bid, state.round_number))

    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        ps = state.player_states[player_index]
        hand = list(ps.hand)
        played = state.current_trick_cards

        legal = TrickResolver.legal_plays(list(played), hand)
        bid = ps.bid if ps.bid is not None else 0
        want_to_win = ps.tricks_won_this_round < bid

        candidates = _candidates(legal)
        return self._choose(candidates, played, player_index, want_to_win)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _choose(
        candidates: list[tuple[Card, Optional[TigressMode]]],
        played: tuple[PlayedCard, ...],
        player_index: int,
        want_to_win: bool,
    ) -> tuple[Card, Optional[TigressMode]]:
        key = lambda cm: _card_strength(cm[0], cm[1])  # noqa: E731

        if not played:
            # Leading the trick
            if want_to_win:
                return max(candidates, key=key)
            # Prefer Escape → then weakest non-winner-likely card
            escapes = [(c, m) for c, m in candidates if c.card_type == CardType.ESCAPE]
            if escapes:
                return escapes[0]
            safe = [
                (c, m) for c, m in candidates
                if c.card_type not in (CardType.SKULL_KING, CardType.PIRATE)
                and not (c.card_type == CardType.TIGRESS and m == TigressMode.PIRATE)
            ]
            pool = safe if safe else candidates
            return min(pool, key=key)

        # Following in trick
        if want_to_win:
            winners = [(c, m) for c, m in candidates
                       if _would_beat(c, played, player_index, m)]
            pool = winners if winners else candidates
            return min(pool, key=key)  # weakest winner first (conserve strong cards)
        else:
            losers = [(c, m) for c, m in candidates
                      if not _would_beat(c, played, player_index, m)]
            pool = losers if losers else candidates
            return min(pool, key=key)
