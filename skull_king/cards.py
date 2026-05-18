from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Suit(Enum):
    BLACK = "BLACK"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    PURPLE = "PURPLE"


class CardType(Enum):
    NUMBERED = "NUMBERED"
    ESCAPE = "ESCAPE"
    PIRATE = "PIRATE"
    MERMAID = "MERMAID"
    SKULL_KING = "SKULL_KING"
    TIGRESS = "TIGRESS"


class TigressMode(Enum):
    PIRATE = "PIRATE"
    ESCAPE = "ESCAPE"


TRUMP_SUIT = Suit.BLACK
NUM_ROUNDS = 10
MAX_PLAYERS = 6
DECK_TOTAL = 70

_SPECIAL_COUNTS: dict[CardType, int] = {
    CardType.ESCAPE: 5,
    CardType.PIRATE: 5,
    CardType.MERMAID: 2,
    CardType.SKULL_KING: 1,
    CardType.TIGRESS: 1,
}


@dataclass(frozen=True)
class Card:
    card_type: CardType
    suit: Optional[Suit] = None
    value: Optional[int] = None

    def __post_init__(self) -> None:
        if self.card_type == CardType.NUMBERED:
            if self.suit is None:
                raise ValueError("Numbered card requires a suit")
            if self.value is None or not (1 <= self.value <= 14):
                raise ValueError(f"Numbered card value must be 1–14, got {self.value!r}")
        else:
            if self.suit is not None:
                raise ValueError(f"{self.card_type.value} card must not have a suit")
            if self.value is not None:
                raise ValueError(f"{self.card_type.value} card must not have a value")

    @property
    def is_special(self) -> bool:
        return self.card_type != CardType.NUMBERED

    @property
    def is_trump(self) -> bool:
        return self.card_type == CardType.NUMBERED and self.suit == TRUMP_SUIT

    @property
    def numeric_value(self) -> int:
        """Value as int — only valid for NUMBERED cards."""
        if self.value is None:
            raise TypeError(f"{self.card_type.value} card has no numeric value")
        return self.value

    def __repr__(self) -> str:
        if self.card_type == CardType.NUMBERED:
            return f"Card({self.suit.value} {self.value})"  # type: ignore[union-attr]
        return f"Card({self.card_type.value})"


def build_deck() -> list[Card]:
    cards: list[Card] = []
    for suit in Suit:
        for v in range(1, 15):
            cards.append(Card(card_type=CardType.NUMBERED, suit=suit, value=v))
    for card_type, count in _SPECIAL_COUNTS.items():
        for _ in range(count):
            cards.append(Card(card_type=card_type))
    assert len(cards) == DECK_TOTAL, f"Expected {DECK_TOTAL} cards, got {len(cards)}"
    return cards


@dataclass
class Deck:
    _cards: list[Card] = field(default_factory=build_deck)

    def shuffle(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        rng.shuffle(self._cards)

    def deal(
        self, n_players: int, cards_per_player: int
    ) -> tuple[list[list[Card]], list[Card]]:
        """Return (hands, remainder). Does not modify the deck."""
        needed = n_players * cards_per_player
        if needed > len(self._cards):
            raise ValueError(
                f"Cannot deal {needed} cards from a deck of {len(self._cards)}"
            )
        hands = [
            list(self._cards[i * cards_per_player : (i + 1) * cards_per_player])
            for i in range(n_players)
        ]
        remainder = list(self._cards[needed:])
        return hands, remainder

    def __len__(self) -> int:
        return len(self._cards)
