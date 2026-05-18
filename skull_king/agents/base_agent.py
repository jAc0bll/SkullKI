"""Abstract base class shared by all tournament-compatible agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from skull_king.cards import Card, TigressMode
from skull_king.game_state import GameState

if TYPE_CHECKING:
    from skull_king.engine import GameEngine


class BaseAgent(ABC):
    """Interface every tournament agent must implement.

    The tournament runner calls ``before_move(engine)`` immediately before
    each ``bid()`` or ``play()`` call so that agents needing engine state
    (e.g. MCTS) can take a snapshot without the runner needing to know about
    implementation details.
    """

    name: str = "Agent"

    def before_move(self, engine: "GameEngine") -> None:
        """Optional hook called before every bid/play decision."""

    @abstractmethod
    def bid(self, state: GameState, player_index: int) -> int:
        """Return a bid in [0, state.round_number]."""

    @abstractmethod
    def play(
        self, state: GameState, player_index: int
    ) -> tuple[Card, Optional[TigressMode]]:
        """Return ``(card, tigress_mode)`` for a legal play.

        ``tigress_mode`` must be non-None iff the card is Tigress.
        """
