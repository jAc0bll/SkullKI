"""Tests for RoundScore — all four base-score formulas and bonus gating."""
import pytest
from skull_king.scoring import RoundScore


def rs(bid: int, won: int, round_number: int = 5, bonus: int = 0) -> RoundScore:
    return RoundScore(
        player_index=0,
        round_number=round_number,
        bid=bid,
        tricks_won=won,
        bonus_points=bonus,
    )


# ---------------------------------------------------------------------------
# bid_successful
# ---------------------------------------------------------------------------


class TestBidSuccessful:
    def test_hit(self):
        assert rs(3, 3).bid_successful is True

    def test_miss_over(self):
        assert rs(3, 5).bid_successful is False

    def test_miss_under(self):
        assert rs(3, 2).bid_successful is False

    def test_bid_zero_never_successful(self):
        assert rs(0, 0).bid_successful is False

    def test_bid_one_won_one(self):
        assert rs(1, 1).bid_successful is True


# ---------------------------------------------------------------------------
# Base score (spec §6.1)
# ---------------------------------------------------------------------------


class TestBaseScore:
    # Bid 0 success: +round_number × 10
    def test_bid_zero_hit_round_1(self):
        assert rs(0, 0, round_number=1).base_score == 10

    def test_bid_zero_hit_round_10(self):
        assert rs(0, 0, round_number=10).base_score == 100

    def test_bid_zero_hit_round_5(self):
        assert rs(0, 0, round_number=5).base_score == 50

    # Bid 0 failure: -round_number × 10
    def test_bid_zero_miss_one_trick(self):
        assert rs(0, 1, round_number=3).base_score == -30

    def test_bid_zero_miss_multiple_tricks(self):
        # Penalty is flat per round regardless of how many tricks won
        assert rs(0, 4, round_number=7).base_score == -70

    # Bid > 0 success: bid × 20
    def test_bid_hit(self):
        assert rs(3, 3).base_score == 60

    def test_bid_one_hit(self):
        assert rs(1, 1).base_score == 20

    def test_bid_ten_hit(self):
        assert rs(10, 10, round_number=10).base_score == 200

    # Bid > 0 miss: -|bid - won| × 10
    def test_bid_miss_under(self):
        assert rs(3, 2).base_score == -10

    def test_bid_miss_over(self):
        assert rs(3, 5).base_score == -20

    def test_bid_miss_by_three(self):
        assert rs(5, 2).base_score == -30


# ---------------------------------------------------------------------------
# Total score = base + bonuses (only on successful bid, CONFIRMED-04)
# ---------------------------------------------------------------------------


class TestTotalScore:
    def test_successful_bid_adds_bonus(self):
        score = rs(3, 3, bonus=40)
        assert score.total_score == 60 + 40

    def test_missed_bid_ignores_bonus(self):
        score = rs(3, 2, bonus=40)
        assert score.total_score == -10  # no bonus applied

    def test_bid_zero_hit_ignores_bonus(self):
        score = rs(0, 0, round_number=5, bonus=30)
        assert score.total_score == 50  # bonus never applies to bid-0

    def test_bid_zero_miss_ignores_bonus(self):
        score = rs(0, 1, round_number=5, bonus=30)
        assert score.total_score == -50

    def test_zero_bonus_hit(self):
        score = rs(2, 2, bonus=0)
        assert score.total_score == 40

    def test_large_bonus_hit(self):
        # SK + 2 Pirates + Black 14: 60 + 70 = 130
        score = rs(3, 3, round_number=3, bonus=70)
        assert score.total_score == 60 + 70

    def test_negative_base_score_zero_bonus(self):
        score = rs(4, 1, bonus=0)
        assert score.total_score == -30


# ---------------------------------------------------------------------------
# Round number boundary checks
# ---------------------------------------------------------------------------


class TestRoundNumberBoundaries:
    @pytest.mark.parametrize("rn", range(1, 11))
    def test_bid_zero_hit_all_rounds(self, rn: int):
        assert rs(0, 0, round_number=rn).base_score == rn * 10

    @pytest.mark.parametrize("rn", range(1, 11))
    def test_bid_zero_miss_all_rounds(self, rn: int):
        assert rs(0, 1, round_number=rn).base_score == -rn * 10
