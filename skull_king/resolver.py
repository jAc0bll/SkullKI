from __future__ import annotations

from skull_king.cards import Card
from skull_king.trick import PlayedCard, Trick, TrickResult


class TrickResolver:
    """Stateless facade over Trick — the public API for resolving tricks.

    All logic lives in Trick; this class exists so callers never need to
    construct a Trick themselves.
    """

    @staticmethod
    def resolve(played_cards: list[PlayedCard]) -> TrickResult:
        """Determine the winner and bonuses for a completed trick."""
        return Trick(list(played_cards)).resolve()

    @staticmethod
    def legal_plays(played_so_far: list[PlayedCard], hand: list[Card]) -> list[Card]:
        """Cards in *hand* that are legal to play given what has been played so far."""
        return Trick(list(played_so_far)).legal_cards(hand)
