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

# Compact per-identity hash table.  Used by Card.__hash__ to avoid the slow
# default dataclass hash which calls Enum.__hash__ on every Suit and CardType
# (profiling showed ~20% of CFR traversal time was spent in enum hashing for
# dict[Card, …] lookups in observation / mask building).
#
# Encoding (60 distinct equality classes — Pirate ×5 etc. collapse since they
# compare equal):
#   0..55:   NUMBERED  (suit_idx × 14 + (value-1))
#   56:      ESCAPE
#   57:      PIRATE
#   58:      MERMAID
#   59:      SKULL_KING
#   60:      TIGRESS
_SUIT_HASH_IDX: dict["Suit", int] = {}  # populated below once Suit is defined
_TYPE_HASH_OFFSET: dict["CardType", int] = {}


def _init_hash_tables() -> None:
    suits = list(Suit)  # deterministic order from Enum
    for i, s in enumerate(suits):
        _SUIT_HASH_IDX[s] = i
    _TYPE_HASH_OFFSET[CardType.ESCAPE] = 56
    _TYPE_HASH_OFFSET[CardType.PIRATE] = 57
    _TYPE_HASH_OFFSET[CardType.MERMAID] = 58
    _TYPE_HASH_OFFSET[CardType.SKULL_KING] = 59
    _TYPE_HASH_OFFSET[CardType.TIGRESS] = 60


_init_hash_tables()


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
            h = _SUIT_HASH_IDX[self.suit] * 14 + (self.value - 1)
        else:
            if self.suit is not None:
                raise ValueError(f"{self.card_type.value} card must not have a suit")
            if self.value is not None:
                raise ValueError(f"{self.card_type.value} card must not have a value")
            h = _TYPE_HASH_OFFSET[self.card_type]
        # frozen dataclass: must use object.__setattr__ to write attributes.
        object.__setattr__(self, "_hash", h)

    def __hash__(self) -> int:
        # Equal cards always have the same hash because we derive `_hash`
        # deterministically from (card_type, suit, value).  Dataclass-generated
        # __eq__ stays in effect and is the source of truth for equality.
        return self._hash

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


_DECK_TEMPLATE: list[Card] = []  # populated lazily below


def build_deck() -> list[Card]:
    """Return a fresh shallow copy of the canonical 70-card deck.

    Cards are immutable, so we build the master deck once and return
    copies of the *list* on subsequent calls. Without this cache, MCTS
    rebuilt the deck (70 Card objects via __post_init__ each) on every
    round-deal — measured at 3.8s of a 27s training iter.
    """
    if not _DECK_TEMPLATE:
        for suit in Suit:
            for v in range(1, 15):
                _DECK_TEMPLATE.append(Card(card_type=CardType.NUMBERED, suit=suit, value=v))
        for card_type, count in _SPECIAL_COUNTS.items():
            for _ in range(count):
                _DECK_TEMPLATE.append(Card(card_type=card_type))
        assert len(_DECK_TEMPLATE) == DECK_TOTAL, (
            f"Expected {DECK_TOTAL} cards, got {len(_DECK_TEMPLATE)}"
        )
    return list(_DECK_TEMPLATE)  # fresh list, same immutable Card refs


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
