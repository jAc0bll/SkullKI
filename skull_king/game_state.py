from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.trick import PlayedCard


class GamePhase(Enum):
    BIDDING = "BIDDING"
    PLAYING = "PLAYING"
    ROUND_END = "ROUND_END"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True)
class FrozenPlayerState:
    """Immutable snapshot of one player's state.

    When building an observation for player P, set hand=() for all other players
    to model hidden information.
    """

    player_index: int
    hand: tuple[Card, ...]  # () for opponents in partial-obs settings
    bid: Optional[int]      # None during the bidding phase (before reveal)
    tricks_won_this_round: int
    accumulated_bonus: int
    total_score: int


@dataclass(frozen=True)
class GameState:
    """Fully immutable snapshot of the game — the RL observation source.

    All mutable Python containers are replaced with tuples so the object is
    hashable and safe to store in replay buffers without copying.
    """

    round_number: int           # 1–10
    trick_number: int           # 1–round_number
    phase: GamePhase
    n_players: int
    current_player_index: int   # whose action is next
    trick_leader_index: int     # who led the current trick
    player_states: tuple[FrozenPlayerState, ...]
    current_trick_cards: tuple[PlayedCard, ...]  # cards played so far this trick

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> FrozenPlayerState:
        return self.player_states[self.current_player_index]

    @property
    def scores(self) -> tuple[int, ...]:
        return tuple(ps.total_score for ps in self.player_states)

    @property
    def bids(self) -> tuple[Optional[int], ...]:
        return tuple(ps.bid for ps in self.player_states)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict (full state, all hands visible)."""
        return {
            "round_number": self.round_number,
            "trick_number": self.trick_number,
            "phase": self.phase.value,
            "n_players": self.n_players,
            "current_player_index": self.current_player_index,
            "trick_leader_index": self.trick_leader_index,
            "player_states": [_frozen_player_to_dict(ps) for ps in self.player_states],
            "current_trick_cards": [_played_card_to_dict(pc) for pc in self.current_trick_cards],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> GameState:
        """Reconstruct a GameState from a serialized dict."""
        return cls(
            round_number=data["round_number"],
            trick_number=data["trick_number"],
            phase=GamePhase(data["phase"]),
            n_players=data["n_players"],
            current_player_index=data["current_player_index"],
            trick_leader_index=data["trick_leader_index"],
            player_states=tuple(
                _frozen_player_from_dict(ps) for ps in data["player_states"]
            ),
            current_trick_cards=tuple(
                _played_card_from_dict(pc) for pc in data["current_trick_cards"]
            ),
        )

    # ------------------------------------------------------------------
    # RL observation
    # ------------------------------------------------------------------

    def observation_for(self, player_index: int) -> dict:
        """Partial-information view for *player_index*: opponents' hands are hidden."""
        masked_states = tuple(
            FrozenPlayerState(
                player_index=ps.player_index,
                hand=ps.hand if ps.player_index == player_index else (),
                bid=ps.bid,
                tricks_won_this_round=ps.tricks_won_this_round,
                accumulated_bonus=ps.accumulated_bonus,
                total_score=ps.total_score,
            )
            for ps in self.player_states
        )
        masked = GameState(
            round_number=self.round_number,
            trick_number=self.trick_number,
            phase=self.phase,
            n_players=self.n_players,
            current_player_index=self.current_player_index,
            trick_leader_index=self.trick_leader_index,
            player_states=masked_states,
            current_trick_cards=self.current_trick_cards,
        )
        return masked.to_dict()


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------

def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "suit": card.suit.value if card.suit else None,
        "value": card.value,
    }


def _card_from_dict(data: dict) -> Card:
    return Card(
        card_type=CardType(data["card_type"]),
        suit=Suit(data["suit"]) if data["suit"] else None,
        value=data["value"],
    )


def _played_card_to_dict(pc: PlayedCard) -> dict:
    return {
        "card": _card_to_dict(pc.card),
        "player_index": pc.player_index,
        "play_order": pc.play_order,
        "tigress_mode": pc.tigress_mode.value if pc.tigress_mode else None,
    }


def _played_card_from_dict(data: dict) -> PlayedCard:
    return PlayedCard(
        card=_card_from_dict(data["card"]),
        player_index=data["player_index"],
        play_order=data["play_order"],
        tigress_mode=TigressMode(data["tigress_mode"]) if data["tigress_mode"] else None,
    )


def _frozen_player_to_dict(ps: FrozenPlayerState) -> dict:
    return {
        "player_index": ps.player_index,
        "hand": [_card_to_dict(c) for c in ps.hand],
        "bid": ps.bid,
        "tricks_won_this_round": ps.tricks_won_this_round,
        "accumulated_bonus": ps.accumulated_bonus,
        "total_score": ps.total_score,
    }


def _frozen_player_from_dict(data: dict) -> FrozenPlayerState:
    return FrozenPlayerState(
        player_index=data["player_index"],
        hand=tuple(_card_from_dict(c) for c in data["hand"]),
        bid=data["bid"],
        tricks_won_this_round=data["tricks_won_this_round"],
        accumulated_bonus=data["accumulated_bonus"],
        total_score=data["total_score"],
    )
