from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from skull_king.cards import Card
from skull_king.scoring import RoundScore


@dataclass
class PlayerState:
    """Mutable per-player state for one game.

    Round-level fields (bid, tricks_won_this_round, accumulated_bonus) are reset
    at the start of each round via reset_for_round(). Completed rounds are
    archived in score_history.
    """

    player_index: int
    hand: list[Card] = field(default_factory=list)
    bid: Optional[int] = None
    tricks_won_this_round: int = 0
    accumulated_bonus: int = 0  # sum of TrickResult.bonus_points for tricks won
    score_history: list[RoundScore] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Score queries
    # ------------------------------------------------------------------

    @property
    def total_score(self) -> int:
        return sum(rs.total_score for rs in self.score_history)

    @property
    def rounds_played(self) -> int:
        return len(self.score_history)

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def set_hand(self, cards: list[Card]) -> None:
        self.hand = list(cards)

    def place_bid(self, bid: int, round_number: int) -> None:
        if bid < 0 or bid > round_number:
            raise ValueError(
                f"Bid {bid} out of range [0, {round_number}] for round {round_number}"
            )
        self.bid = bid

    def record_trick_win(self, bonus_points: int) -> None:
        """Called when this player wins a trick."""
        self.tricks_won_this_round += 1
        self.accumulated_bonus += bonus_points

    def finalize_round(self, round_number: int) -> RoundScore:
        """Compute and archive the score for the completed round."""
        if self.bid is None:
            raise RuntimeError("Cannot finalize round: no bid was placed")
        rs = RoundScore(
            player_index=self.player_index,
            round_number=round_number,
            bid=self.bid,
            tricks_won=self.tricks_won_this_round,
            bonus_points=self.accumulated_bonus,
        )
        self.score_history.append(rs)
        return rs

    def reset_for_round(self) -> None:
        """Clear per-round fields. Call at the start of each new round after dealing."""
        self.hand = []
        self.bid = None
        self.tricks_won_this_round = 0
        self.accumulated_bonus = 0

    def __repr__(self) -> str:
        return (
            f"PlayerState(index={self.player_index} "
            f"bid={self.bid} won={self.tricks_won_this_round} "
            f"total={self.total_score})"
        )
