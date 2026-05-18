import pytest
from skull_king.cards import Card, CardType, Suit
from skull_king.player import PlayerState
from skull_king.scoring import RoundScore


def make_player(index: int = 0) -> PlayerState:
    return PlayerState(player_index=index)


def num(suit: Suit, value: int) -> Card:
    return Card(card_type=CardType.NUMBERED, suit=suit, value=value)


# ---------------------------------------------------------------------------
# Bidding
# ---------------------------------------------------------------------------


class TestBidding:
    def test_place_bid_valid(self):
        p = make_player()
        p.place_bid(3, round_number=5)
        assert p.bid == 3

    def test_place_bid_zero(self):
        p = make_player()
        p.place_bid(0, round_number=5)
        assert p.bid == 0

    def test_bid_equal_to_round_number(self):
        p = make_player()
        p.place_bid(10, round_number=10)
        assert p.bid == 10

    def test_bid_above_round_number_raises(self):
        p = make_player()
        with pytest.raises(ValueError):
            p.place_bid(6, round_number=5)

    def test_negative_bid_raises(self):
        p = make_player()
        with pytest.raises(ValueError):
            p.place_bid(-1, round_number=5)


# ---------------------------------------------------------------------------
# Trick win accumulation
# ---------------------------------------------------------------------------


class TestTrickAccumulation:
    def test_record_trick_win_increments(self):
        p = make_player()
        p.record_trick_win(bonus_points=0)
        p.record_trick_win(bonus_points=30)
        assert p.tricks_won_this_round == 2
        assert p.accumulated_bonus == 30

    def test_multiple_bonuses_accumulate(self):
        p = make_player()
        p.record_trick_win(40)
        p.record_trick_win(30)
        p.record_trick_win(20)
        assert p.accumulated_bonus == 90


# ---------------------------------------------------------------------------
# finalize_round
# ---------------------------------------------------------------------------


class TestFinalizeRound:
    def test_finalize_without_bid_raises(self):
        p = make_player()
        with pytest.raises(RuntimeError):
            p.finalize_round(round_number=1)

    def test_finalize_creates_round_score(self):
        p = make_player()
        p.place_bid(3, round_number=5)
        p.record_trick_win(0)
        p.record_trick_win(0)
        p.record_trick_win(30)  # bonus for SK capture
        rs = p.finalize_round(round_number=5)
        assert isinstance(rs, RoundScore)
        assert rs.bid == 3
        assert rs.tricks_won == 3
        assert rs.bonus_points == 30

    def test_finalize_appends_to_history(self):
        p = make_player()
        p.place_bid(2, round_number=3)
        p.record_trick_win(0)
        p.record_trick_win(0)
        p.finalize_round(round_number=3)
        assert len(p.score_history) == 1

    def test_total_score_sums_history(self):
        p = make_player()
        # Round 1: bid 1, won 1 → +20
        p.place_bid(1, round_number=1)
        p.record_trick_win(0)
        p.finalize_round(1)
        p.reset_for_round()

        # Round 2: bid 0, won 0 → +20
        p.place_bid(0, round_number=2)
        p.finalize_round(2)

        assert p.total_score == 20 + 20


# ---------------------------------------------------------------------------
# reset_for_round
# ---------------------------------------------------------------------------


class TestResetForRound:
    def test_reset_clears_hand(self):
        p = make_player()
        p.set_hand([num(Suit.YELLOW, 5), num(Suit.GREEN, 3)])
        p.reset_for_round()
        assert p.hand == []

    def test_reset_clears_bid(self):
        p = make_player()
        p.place_bid(3, round_number=5)
        p.reset_for_round()
        assert p.bid is None

    def test_reset_clears_tricks_and_bonus(self):
        p = make_player()
        p.record_trick_win(40)
        p.record_trick_win(30)
        p.reset_for_round()
        assert p.tricks_won_this_round == 0
        assert p.accumulated_bonus == 0

    def test_reset_preserves_score_history(self):
        p = make_player()
        p.place_bid(2, round_number=2)
        p.record_trick_win(0)
        p.record_trick_win(0)
        p.finalize_round(2)
        p.reset_for_round()
        assert len(p.score_history) == 1  # history preserved


# ---------------------------------------------------------------------------
# set_hand
# ---------------------------------------------------------------------------


class TestSetHand:
    def test_set_hand_replaces(self):
        p = make_player()
        cards = [num(Suit.YELLOW, 5)]
        p.set_hand(cards)
        assert p.hand == cards

    def test_set_hand_is_copy(self):
        p = make_player()
        cards = [num(Suit.YELLOW, 5)]
        p.set_hand(cards)
        cards.clear()
        assert len(p.hand) == 1  # not affected by external mutation


# ---------------------------------------------------------------------------
# rounds_played
# ---------------------------------------------------------------------------


def test_rounds_played():
    p = make_player()
    assert p.rounds_played == 0
    p.place_bid(1, 1)
    p.record_trick_win(0)
    p.finalize_round(1)
    assert p.rounds_played == 1
