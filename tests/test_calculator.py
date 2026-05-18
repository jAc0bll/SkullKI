"""ScoreCalculator — game-level scoring and rankings."""
import pytest
from skull_king.calculator import GameResult, ScoreCalculator
from skull_king.player import PlayerState
from skull_king.scoring import RoundScore


def make_player(index: int, scores: list[int]) -> PlayerState:
    """Create a PlayerState with pre-populated score_history using fixed scores."""
    p = PlayerState(player_index=index)
    for rn, sc in enumerate(scores, start=1):
        # Build a RoundScore whose total_score equals sc.
        # bid=1, tricks_won=1 gives base=20; pad with bonus.
        rs = RoundScore(
            player_index=index,
            round_number=rn,
            bid=1,
            tricks_won=1,
            bonus_points=sc - 20,
        )
        p.score_history.append(rs)
    return p


class TestRoundScoreFactory:
    def test_returns_round_score(self):
        rs = ScoreCalculator.round_score(0, 3, bid=2, tricks_won=2, bonus_points=30)
        assert rs.total_score == 40 + 30  # bid hit: 2×20 + 30

    def test_default_bonus_zero(self):
        rs = ScoreCalculator.round_score(0, 1, bid=1, tricks_won=1)
        assert rs.bonus_points == 0
        assert rs.total_score == 20

    def test_missed_bid_no_bonus(self):
        rs = ScoreCalculator.round_score(0, 5, bid=3, tricks_won=2, bonus_points=40)
        assert rs.total_score == -10  # miss, bonus ignored


class TestGameResult:
    def test_winner_is_highest_scorer(self):
        players = [make_player(0, [20]), make_player(1, [80]), make_player(2, [50])]
        result = ScoreCalculator.game_result(players)
        assert result.winner == 1

    def test_rankings_sorted_descending(self):
        players = [make_player(0, [20]), make_player(1, [80]), make_player(2, [50])]
        result = ScoreCalculator.game_result(players)
        scores = [score for _, score in result.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_rankings_contain_all_players(self):
        players = [make_player(i, [20]) for i in range(4)]
        result = ScoreCalculator.game_result(players)
        assert len(result.rankings) == 4
        assert {idx for idx, _ in result.rankings} == {0, 1, 2, 3}

    def test_winning_score(self):
        players = [make_player(0, [20]), make_player(1, [80])]
        result = ScoreCalculator.game_result(players)
        assert result.winning_score == 80

    def test_negative_scores_ranked_correctly(self):
        players = [make_player(0, [-30]), make_player(1, [-10])]
        result = ScoreCalculator.game_result(players)
        assert result.winner == 1  # -10 > -30

    def test_game_result_is_frozen(self):
        players = [make_player(0, [20])]
        result = ScoreCalculator.game_result(players)
        with pytest.raises(Exception):
            result.rankings = ()  # type: ignore[misc]


class TestLeaderboard:
    def test_leaderboard_line_count(self):
        players = [make_player(i, [20]) for i in range(3)]
        lb = ScoreCalculator.leaderboard(players)
        assert len(lb) == 3

    def test_leaderboard_rank_1_is_highest(self):
        players = [make_player(0, [20]), make_player(1, [80])]
        lb = ScoreCalculator.leaderboard(players)
        assert "Player 1" in lb[0]
        assert "+80" in lb[0]

    def test_leaderboard_format(self):
        players = [make_player(0, [40])]
        lb = ScoreCalculator.leaderboard(players)
        assert lb[0].startswith("1. Player 0:")

    def test_leaderboard_negative_score_format(self):
        p = PlayerState(player_index=0)
        rs = RoundScore(player_index=0, round_number=1, bid=2, tricks_won=0)
        p.score_history.append(rs)
        lb = ScoreCalculator.leaderboard([p])
        assert "-" in lb[0]


class TestScoreDelta:
    def test_score_delta_correct_round(self):
        players = [make_player(0, [20, 40]), make_player(1, [60, 30])]
        delta = ScoreCalculator.score_delta(players, round_number=2)
        assert delta[0] == 40
        assert delta[1] == 30

    def test_score_delta_missing_round_is_zero(self):
        players = [make_player(0, [20])]
        delta = ScoreCalculator.score_delta(players, round_number=5)
        assert delta[0] == 0
