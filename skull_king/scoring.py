from __future__ import annotations

from dataclasses import dataclass

# Scoring constants (spec §10)
BID_HIT_PER_TRICK = 20
BID_MISS_PER_TRICK = -10
BID_ZERO_HIT_PER_ROUND = 10
BID_ZERO_MISS_PER_ROUND = -10


@dataclass
class RoundScore:
    """Scoring outcome for one player in one round.

    Instantiate after all tricks are played; pass the accumulated bonus_points
    from the tricks that player won.
    """

    player_index: int
    round_number: int  # 1–10
    bid: int
    tricks_won: int
    bonus_points: int = 0  # accumulated from TrickResult.bonus_points for won tricks

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def bid_successful(self) -> bool:
        """True iff the player bid > 0 and matched exactly."""
        return self.bid > 0 and self.tricks_won == self.bid

    @property
    def base_score(self) -> int:
        """Score before bonuses (spec §6.1)."""
        if self.bid == 0:
            if self.tricks_won == 0:
                return self.round_number * BID_ZERO_HIT_PER_ROUND
            return self.round_number * BID_ZERO_MISS_PER_ROUND

        if self.tricks_won == self.bid:
            return self.bid * BID_HIT_PER_TRICK

        return abs(self.bid - self.tricks_won) * BID_MISS_PER_TRICK

    @property
    def total_score(self) -> int:
        """Bonuses apply only on a successful bid (spec CONFIRMED-04)."""
        if self.bid_successful:
            return self.base_score + self.bonus_points
        return self.base_score

    def __repr__(self) -> str:
        return (
            f"RoundScore(p={self.player_index} r={self.round_number} "
            f"bid={self.bid} won={self.tricks_won} "
            f"bonus={self.bonus_points} total={self.total_score})"
        )
