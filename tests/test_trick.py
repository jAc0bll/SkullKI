"""Tests for Trick resolution and legal_cards.

Each test maps to a named case in docs/rules_spec.md §3, §5, §6, or §9.
"""
import pytest
from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.trick import PlayedCard, Trick


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def num(suit: Suit, value: int) -> Card:
    return Card(card_type=CardType.NUMBERED, suit=suit, value=value)


def play(card: Card, player: int, order: int, tigress_mode: TigressMode | None = None) -> PlayedCard:
    return PlayedCard(card=card, player_index=player, play_order=order, tigress_mode=tigress_mode)


SK = Card(card_type=CardType.SKULL_KING)
MERMAID = Card(card_type=CardType.MERMAID)
PIRATE = Card(card_type=CardType.PIRATE)
ESCAPE = Card(card_type=CardType.ESCAPE)
TIGRESS = Card(card_type=CardType.TIGRESS)
BLACK_14 = num(Suit.BLACK, 14)
BLACK_13 = num(Suit.BLACK, 13)
BLACK_1 = num(Suit.BLACK, 1)
YELLOW_7 = num(Suit.YELLOW, 7)
YELLOW_9 = num(Suit.YELLOW, 9)
YELLOW_3 = num(Suit.YELLOW, 3)
GREEN_9 = num(Suit.GREEN, 9)
PURPLE_5 = num(Suit.PURPLE, 5)


# ---------------------------------------------------------------------------
# PlayedCard validation
# ---------------------------------------------------------------------------


class TestPlayedCardValidation:
    def test_tigress_without_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            PlayedCard(card=TIGRESS, player_index=0, play_order=1)

    def test_non_tigress_with_mode_raises(self):
        with pytest.raises(ValueError, match="tigress_mode"):
            PlayedCard(card=PIRATE, player_index=0, play_order=1, tigress_mode=TigressMode.PIRATE)

    def test_play_order_zero_raises(self):
        with pytest.raises(ValueError, match="play_order"):
            PlayedCard(card=ESCAPE, player_index=0, play_order=0)

    def test_tigress_as_pirate_effective_type(self):
        pc = play(TIGRESS, 0, 1, TigressMode.PIRATE)
        assert pc.effective_type == CardType.PIRATE

    def test_tigress_as_escape_effective_type(self):
        pc = play(TIGRESS, 0, 1, TigressMode.ESCAPE)
        assert pc.effective_type == CardType.ESCAPE


# ---------------------------------------------------------------------------
# Skull King interactions (spec §3 interaction matrix)
# ---------------------------------------------------------------------------


class TestSkullKingInteractions:
    def test_sk_alone_wins(self):
        t = Trick([play(SK, 0, 1), play(YELLOW_7, 1, 2)])
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_sk_beats_pirate(self):
        t = Trick([play(PIRATE, 1, 1), play(SK, 0, 2)])
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_sk_beats_multiple_pirates(self):
        t = Trick([play(PIRATE, 1, 1), play(PIRATE, 2, 2), play(SK, 0, 3)])
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_mermaid_beats_sk(self):
        t = Trick([play(SK, 0, 1), play(MERMAID, 1, 2)])
        result = t.resolve()
        assert result.winner_player_index == 1

    def test_sk_wins_when_pirate_negates_mermaid(self):
        """SK + Mermaid + Pirate: Pirate beats Mermaid (Mermaid loses her SK counter),
        then SK beats Pirate. SK wins (spec §3)."""
        t = Trick([play(SK, 0, 1), play(MERMAID, 1, 2), play(PIRATE, 2, 3)])
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_first_pirate_wins_among_multiple(self):
        t = Trick([play(PIRATE, 2, 2), play(PIRATE, 1, 1)])  # player 1 played first
        result = t.resolve()
        assert result.winner_player_index == 1

    def test_first_mermaid_wins_among_multiple_no_sk(self):
        t = Trick([play(MERMAID, 1, 2), play(MERMAID, 0, 1)])  # player 0 played first
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_first_mermaid_wins_among_multiple_with_sk(self):
        t = Trick([play(SK, 2, 3), play(MERMAID, 1, 2), play(MERMAID, 0, 1)])
        result = t.resolve()
        assert result.winner_player_index == 0  # first mermaid

    def test_mermaid_alone_wins(self):
        t = Trick([play(MERMAID, 0, 1), play(YELLOW_7, 1, 2)])
        result = t.resolve()
        assert result.winner_player_index == 0

    def test_pirate_beats_mermaid_without_sk(self):
        t = Trick([play(MERMAID, 0, 1), play(PIRATE, 1, 2)])
        result = t.resolve()
        assert result.winner_player_index == 1


# ---------------------------------------------------------------------------
# Numbered card resolution (spec §5)
# ---------------------------------------------------------------------------


class TestNumberedResolution:
    def test_led_suit_highest_wins(self):
        t = Trick([play(YELLOW_7, 0, 1), play(YELLOW_9, 1, 2), play(YELLOW_3, 2, 3)])
        assert t.resolve().winner_player_index == 1  # Yellow 9

    def test_trump_beats_led_suit(self):
        t = Trick([play(YELLOW_9, 0, 1), play(BLACK_1, 1, 2)])
        assert t.resolve().winner_player_index == 1  # Black 1 > Yellow 9

    def test_highest_trump_wins(self):
        t = Trick([play(BLACK_1, 0, 1), play(BLACK_13, 1, 2)])
        assert t.resolve().winner_player_index == 1

    def test_off_suit_cannot_win(self):
        t = Trick([play(YELLOW_7, 0, 1), play(GREEN_9, 1, 2)])  # Green not led, can't win
        assert t.resolve().winner_player_index == 0  # Yellow 7 wins

    def test_all_escapes_leader_wins(self):
        """All Escapes → trick leader (first player, play_order=1) wins (spec §9.1)."""
        t = Trick([play(ESCAPE, 2, 1), play(ESCAPE, 0, 2), play(ESCAPE, 1, 3)])
        assert t.resolve().winner_player_index == 2  # first escape played

    def test_escape_led_then_colored_card(self):
        """Escape leads, then Yellow 7 — Yellow 7 wins (spec §9.2)."""
        t = Trick([play(ESCAPE, 0, 1), play(YELLOW_7, 1, 2)])
        assert t.resolve().winner_player_index == 1

    def test_escape_led_then_black_beats_colored(self):
        """Escape leads, Black 5 and Yellow 7 played — Black wins (spec §9.2)."""
        t = Trick([
            play(ESCAPE, 0, 1),
            play(YELLOW_7, 1, 2),
            play(num(Suit.BLACK, 5), 2, 3),
        ])
        assert t.resolve().winner_player_index == 2  # Black 5

    def test_escape_led_multiple_colored_first_suit_wins(self):
        """Escape leads, Yellow 7 then Green 9 — Yellow 7 wins (first color is informal led)."""
        t = Trick([
            play(ESCAPE, 0, 1),
            play(YELLOW_7, 1, 2),
            play(GREEN_9, 2, 3),
        ])
        assert t.resolve().winner_player_index == 1  # Yellow 7


# ---------------------------------------------------------------------------
# Tigress (spec §4.4)
# ---------------------------------------------------------------------------


class TestTigress:
    def test_tigress_as_pirate_beats_numbered(self):
        t = Trick([play(YELLOW_9, 0, 1), play(TIGRESS, 1, 2, TigressMode.PIRATE)])
        assert t.resolve().winner_player_index == 1

    def test_tigress_as_escape_loses(self):
        t = Trick([play(YELLOW_9, 0, 1), play(TIGRESS, 1, 2, TigressMode.ESCAPE)])
        assert t.resolve().winner_player_index == 0

    def test_tigress_as_pirate_loses_to_sk(self):
        t = Trick([play(SK, 0, 1), play(TIGRESS, 1, 2, TigressMode.PIRATE)])
        assert t.resolve().winner_player_index == 0

    def test_all_escapes_includes_tigress_escape(self):
        """Tigress-as-Escape joins all-Escape pool; trick leader wins."""
        t = Trick([
            play(TIGRESS, 0, 1, TigressMode.ESCAPE),
            play(ESCAPE, 1, 2),
        ])
        assert t.resolve().winner_player_index == 0  # first played


# ---------------------------------------------------------------------------
# Bonus computation (spec §6.2–6.3, CONFIRMED-04, CONFIRMED-05)
# ---------------------------------------------------------------------------


class TestBonuses:
    def test_sk_captures_one_pirate(self):
        t = Trick([play(SK, 0, 1), play(PIRATE, 1, 2)])
        result = t.resolve()
        assert result.bonus_points == 30

    def test_sk_captures_two_pirates(self):
        t = Trick([play(SK, 0, 1), play(PIRATE, 1, 2), play(PIRATE, 2, 3)])
        result = t.resolve()
        assert result.bonus_points == 60

    def test_sk_no_pirates_no_bonus(self):
        t = Trick([play(SK, 0, 1), play(YELLOW_7, 1, 2)])
        result = t.resolve()
        assert result.bonus_points == 0

    def test_mermaid_captures_sk(self):
        t = Trick([play(SK, 0, 1), play(MERMAID, 1, 2)])
        result = t.resolve()
        assert result.bonus_points == 40

    def test_mermaid_no_sk_no_bonus(self):
        t = Trick([play(MERMAID, 0, 1), play(YELLOW_7, 1, 2)])
        result = t.resolve()
        assert result.bonus_points == 0

    def test_black14_captured_by_pirate_gives_bonus(self):
        t = Trick([play(YELLOW_9, 0, 1), play(BLACK_14, 1, 2), play(PIRATE, 2, 3)])
        result = t.resolve()
        assert result.winner_player_index == 2  # Pirate wins
        assert result.bonus_points == 20        # Black 14 bonus, won by special

    def test_black14_won_by_numbered_no_bonus(self):
        """Black 13 wins, Black 14 is present — no bonus (CONFIRMED-05)."""
        t = Trick([play(BLACK_14, 0, 1), play(BLACK_13, 1, 2)])
        # Black 14 wins its own trick — but winner is a NUMBERED card
        result = t.resolve()
        assert result.winner_player_index == 0  # Black 14 wins (highest trump)
        assert result.bonus_points == 0         # non-special winner → no bonus

    def test_black14_won_by_sk_gives_bonus(self):
        t = Trick([play(BLACK_14, 0, 1), play(SK, 1, 2)])
        result = t.resolve()
        assert result.winner_player_index == 1
        assert result.bonus_points == 20  # Black 14 bonus only (no pirates)

    def test_sk_captures_pirate_and_black14(self):
        """SK wins trick with 1 Pirate and Black 14: +30 + +20 = +50."""
        t = Trick([play(SK, 0, 1), play(PIRATE, 1, 2), play(BLACK_14, 2, 3)])
        result = t.resolve()
        assert result.winner_player_index == 0
        assert result.bonus_points == 50

    def test_tigress_as_pirate_counts_for_sk_bonus(self):
        """Tigress-as-Pirate captured by SK counts as a Pirate (+30) (CONFIRMED-06)."""
        t = Trick([play(SK, 0, 1), play(TIGRESS, 1, 2, TigressMode.PIRATE)])
        result = t.resolve()
        assert result.bonus_points == 30


# ---------------------------------------------------------------------------
# Legal cards (spec §4.2, CONFIRMED-03)
# ---------------------------------------------------------------------------


class TestLegalCards:
    def test_no_led_suit_all_legal(self):
        t = Trick()
        hand = [YELLOW_7, BLACK_14, PIRATE, ESCAPE]
        assert set(t.legal_cards(hand)) == set(hand)

    def test_led_suit_must_follow(self):
        t = Trick([play(YELLOW_7, 0, 1)])
        hand = [YELLOW_9, YELLOW_3, GREEN_9, PIRATE]
        legal = t.legal_cards(hand)
        # Must play Yellow; Pirate (special) always legal; Green is not
        assert set(legal) == {YELLOW_9, YELLOW_3, PIRATE}
        assert GREEN_9 not in legal

    def test_black_not_playable_over_colored_led_when_holding_led_suit(self):
        """CONFIRMED-03: Black follows same must-follow rule."""
        t = Trick([play(YELLOW_7, 0, 1)])
        hand = [YELLOW_3, BLACK_14]
        legal = t.legal_cards(hand)
        assert YELLOW_3 in legal
        assert BLACK_14 not in legal  # must follow Yellow

    def test_void_in_led_suit_all_legal(self):
        """Void in led suit → any card legal, including Black trump."""
        t = Trick([play(YELLOW_7, 0, 1)])
        hand = [GREEN_9, BLACK_14, PIRATE]
        legal = t.legal_cards(hand)
        assert set(legal) == {GREEN_9, BLACK_14, PIRATE}

    def test_special_always_playable_even_with_led_suit(self):
        t = Trick([play(YELLOW_7, 0, 1)])
        hand = [YELLOW_3, PIRATE, ESCAPE, MERMAID, SK, TIGRESS]
        legal = t.legal_cards(hand)
        for special in [PIRATE, ESCAPE, MERMAID, SK, TIGRESS]:
            assert special in legal

    def test_empty_trick_all_legal(self):
        t = Trick()
        hand = [YELLOW_7, PIRATE, SK]
        assert set(t.legal_cards(hand)) == set(hand)

    def test_led_suit_none_when_escape_led(self):
        """An Escape leading means no led suit — all cards legal for next player."""
        t = Trick([play(ESCAPE, 0, 1)])
        assert t.led_suit is None
        hand = [YELLOW_7, BLACK_14]
        assert set(t.legal_cards(hand)) == {YELLOW_7, BLACK_14}


# ---------------------------------------------------------------------------
# Empty trick guard
# ---------------------------------------------------------------------------


def test_resolve_empty_trick_raises():
    with pytest.raises(ValueError, match="empty"):
        Trick().resolve()
