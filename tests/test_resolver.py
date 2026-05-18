"""TrickResolver — verifies the public facade API works correctly.
Exhaustive interaction tests live in test_trick.py; these cover the API contract.
"""
import pytest
from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.resolver import TrickResolver
from skull_king.trick import PlayedCard


def num(suit: Suit, value: int) -> Card:
    return Card(card_type=CardType.NUMBERED, suit=suit, value=value)


def play(card: Card, player: int, order: int, mode: TigressMode | None = None) -> PlayedCard:
    return PlayedCard(card=card, player_index=player, play_order=order, tigress_mode=mode)


SK = Card(card_type=CardType.SKULL_KING)
PIRATE = Card(card_type=CardType.PIRATE)
MERMAID = Card(card_type=CardType.MERMAID)
ESCAPE = Card(card_type=CardType.ESCAPE)
BLACK_14 = num(Suit.BLACK, 14)
YELLOW_9 = num(Suit.YELLOW, 9)
YELLOW_3 = num(Suit.YELLOW, 3)
GREEN_7 = num(Suit.GREEN, 7)


class TestTrickResolverResolve:
    def test_returns_trick_result(self):
        from skull_king.trick import TrickResult
        result = TrickResolver.resolve([play(SK, 0, 1), play(YELLOW_9, 1, 2)])
        assert isinstance(result, TrickResult)

    def test_sk_beats_numbered(self):
        result = TrickResolver.resolve([play(SK, 0, 1), play(YELLOW_9, 1, 2)])
        assert result.winner_player_index == 0

    def test_mermaid_beats_sk(self):
        result = TrickResolver.resolve([play(SK, 0, 1), play(MERMAID, 1, 2)])
        assert result.winner_player_index == 1
        assert result.bonus_points == 40

    def test_sk_captures_pirate_bonus(self):
        result = TrickResolver.resolve([play(SK, 0, 1), play(PIRATE, 1, 2)])
        assert result.winner_player_index == 0
        assert result.bonus_points == 30

    def test_black14_bonus_pirate_winner(self):
        result = TrickResolver.resolve([
            play(YELLOW_9, 0, 1),
            play(BLACK_14, 1, 2),
            play(PIRATE, 2, 3),
        ])
        assert result.winner_player_index == 2
        assert result.bonus_points == 20

    def test_does_not_mutate_input(self):
        cards = [play(SK, 0, 1), play(PIRATE, 1, 2)]
        original_len = len(cards)
        TrickResolver.resolve(cards)
        assert len(cards) == original_len


class TestTrickResolverLegalPlays:
    def test_no_cards_played_all_legal(self):
        hand = [YELLOW_9, BLACK_14, PIRATE]
        legal = TrickResolver.legal_plays([], hand)
        assert set(legal) == set(hand)

    def test_must_follow_led_suit(self):
        played = [play(YELLOW_9, 0, 1)]
        hand = [YELLOW_3, GREEN_7, PIRATE]
        legal = TrickResolver.legal_plays(played, hand)
        assert set(legal) == {YELLOW_3, PIRATE}
        assert GREEN_7 not in legal

    def test_void_in_led_suit_all_legal(self):
        played = [play(YELLOW_9, 0, 1)]
        hand = [GREEN_7, BLACK_14]
        legal = TrickResolver.legal_plays(played, hand)
        assert set(legal) == {GREEN_7, BLACK_14}

    def test_does_not_mutate_inputs(self):
        played = [play(YELLOW_9, 0, 1)]
        hand = [YELLOW_3, GREEN_7]
        played_copy = list(played)
        hand_copy = list(hand)
        TrickResolver.legal_plays(played, hand)
        assert played == played_copy
        assert hand == hand_copy
