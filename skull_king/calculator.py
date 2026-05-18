from __future__ import annotations

from dataclasses import dataclass

from skull_king.player import PlayerState
from skull_king.scoring import RoundScore


@dataclass(frozen=True)
class GameResult:
    """Final standings after all rounds have been scored."""

    # Sorted highest score first: tuple of (player_index, total_score).
    rankings: tuple[tuple[int, int], ...]

    @property
    def winner(self) -> int:
        """Player index of the highest scorer."""
        return self.rankings[0][0]

    @property
    def winning_score(self) -> int:
        return self.rankings[0][1]


class ScoreCalculator:
    """Game-level scoring on top of the per-round RoundScore."""

    @staticmethod
    def round_score(
        player_index: int,
        round_number: int,
        bid: int,
        tricks_won: int,
        bonus_points: int = 0,
    ) -> RoundScore:
        """Factory for a single round's score record."""
        return RoundScore(
            player_index=player_index,
            round_number=round_number,
            bid=bid,
            tricks_won=tricks_won,
            bonus_points=bonus_points,
        )

    @staticmethod
    def game_result(players: list[PlayerState]) -> GameResult:
        """Rank players by total score (descending)."""
        standings = sorted(
            ((p.player_index, p.total_score) for p in players),
            key=lambda x: x[1],
            reverse=True,
        )
        return GameResult(rankings=tuple(standings))

    @staticmethod
    def leaderboard(players: list[PlayerState]) -> list[str]:
        """Human-readable ranked lines, e.g. '1. Player 2: +140'."""
        result = ScoreCalculator.game_result(players)
        return [
            f"{rank}. Player {idx}: {score:+d}"
            for rank, (idx, score) in enumerate(result.rankings, start=1)
        ]

    @staticmethod
    def score_delta(players: list[PlayerState], round_number: int) -> dict[int, int]:
        """Points each player earned in the given round (0 if round not played)."""
        return {
            p.player_index: next(
                (rs.total_score for rs in p.score_history if rs.round_number == round_number),
                0,
            )
            for p in players
        }
