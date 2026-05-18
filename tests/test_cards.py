import pytest
from skull_king.cards import (
    Card,
    CardType,
    Deck,
    Suit,
    TigressMode,
    DECK_TOTAL,
    build_deck,
    _SPECIAL_COUNTS,
)


# ---------------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------------


class TestCardConstruction:
    def test_numbered_card_valid(self):
        c = Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=7)
        assert c.card_type == CardType.NUMBERED
        assert c.suit == Suit.YELLOW
        assert c.value == 7

    def test_special_card_valid(self):
        for ct in (CardType.ESCAPE, CardType.PIRATE, CardType.MERMAID,
                   CardType.SKULL_KING, CardType.TIGRESS):
            c = Card(card_type=ct)
            assert c.suit is None
            assert c.value is None

    def test_numbered_requires_suit(self):
        with pytest.raises(ValueError, match="suit"):
            Card(card_type=CardType.NUMBERED, value=5)

    def test_numbered_value_out_of_range(self):
        with pytest.raises(ValueError, match="1–14"):
            Card(card_type=CardType.NUMBERED, suit=Suit.GREEN, value=0)
        with pytest.raises(ValueError, match="1–14"):
            Card(card_type=CardType.NUMBERED, suit=Suit.GREEN, value=15)

    def test_numbered_requires_value(self):
        with pytest.raises(ValueError):
            Card(card_type=CardType.NUMBERED, suit=Suit.BLACK)

    def test_special_must_not_have_suit(self):
        with pytest.raises(ValueError, match="suit"):
            Card(card_type=CardType.PIRATE, suit=Suit.BLACK)

    def test_special_must_not_have_value(self):
        with pytest.raises(ValueError, match="value"):
            Card(card_type=CardType.ESCAPE, value=3)

    def test_card_is_frozen(self):
        c = Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=14)
        with pytest.raises(Exception):
            c.value = 13  # type: ignore[misc]

    def test_card_equality_structural(self):
        a = Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=14)
        b = Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=14)
        assert a == b

    def test_card_hashing(self):
        c = Card(card_type=CardType.PIRATE)
        s = {c}
        assert c in s


# ---------------------------------------------------------------------------
# Card properties
# ---------------------------------------------------------------------------


class TestCardProperties:
    def test_is_special_numbered(self):
        c = Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=1)
        assert not c.is_special

    def test_is_special_for_all_specials(self):
        for ct in (CardType.ESCAPE, CardType.PIRATE, CardType.MERMAID,
                   CardType.SKULL_KING, CardType.TIGRESS):
            assert Card(card_type=ct).is_special

    def test_is_trump(self):
        assert Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=7).is_trump
        assert not Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=7).is_trump

    def test_numeric_value_on_numbered(self):
        c = Card(card_type=CardType.NUMBERED, suit=Suit.GREEN, value=11)
        assert c.numeric_value == 11

    def test_numeric_value_raises_on_special(self):
        with pytest.raises(TypeError):
            Card(card_type=CardType.SKULL_KING).numeric_value  # noqa: B018


# ---------------------------------------------------------------------------
# Deck composition
# ---------------------------------------------------------------------------


class TestDeckComposition:
    def test_total_card_count(self):
        deck = build_deck()
        assert len(deck) == DECK_TOTAL

    def test_numbered_cards_count(self):
        deck = build_deck()
        numbered = [c for c in deck if c.card_type == CardType.NUMBERED]
        assert len(numbered) == 56  # 4 suits × 14 values

    def test_all_suits_and_values_present(self):
        deck = build_deck()
        numbered = [c for c in deck if c.card_type == CardType.NUMBERED]
        for suit in Suit:
            for v in range(1, 15):
                assert any(c.suit == suit and c.value == v for c in numbered), \
                    f"Missing {suit} {v}"

    def test_special_card_counts(self):
        deck = build_deck()
        for card_type, expected in _SPECIAL_COUNTS.items():
            actual = sum(1 for c in deck if c.card_type == card_type)
            assert actual == expected, f"{card_type}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Deck — deal
# ---------------------------------------------------------------------------


class TestDeckDeal:
    def test_deal_basic(self):
        d = Deck()
        hands, remainder = d.deal(n_players=3, cards_per_player=5)
        assert len(hands) == 3
        assert all(len(h) == 5 for h in hands)
        assert len(remainder) == DECK_TOTAL - 15

    def test_deal_no_overlap(self):
        d = Deck()
        hands, remainder = d.deal(n_players=4, cards_per_player=10)
        all_cards = [c for h in hands for c in h] + remainder
        assert len(all_cards) == DECK_TOTAL

    def test_deal_full_deck(self):
        d = Deck()
        hands, remainder = d.deal(n_players=1, cards_per_player=DECK_TOTAL)
        assert len(hands[0]) == DECK_TOTAL
        assert remainder == []

    def test_deal_too_many_cards_raises(self):
        d = Deck()
        with pytest.raises(ValueError):
            d.deal(n_players=2, cards_per_player=DECK_TOTAL)

    def test_shuffle_changes_order(self):
        d1 = Deck()
        d2 = Deck()
        d2.shuffle(seed=42)
        h1, _ = d1.deal(4, 10)
        h2, _ = d2.deal(4, 10)
        assert h1 != h2  # overwhelmingly likely with seed ≠ unshuffled

    def test_shuffle_is_reproducible(self):
        d1 = Deck()
        d2 = Deck()
        d1.shuffle(seed=99)
        d2.shuffle(seed=99)
        h1, _ = d1.deal(2, 5)
        h2, _ = d2.deal(2, 5)
        assert h1 == h2
