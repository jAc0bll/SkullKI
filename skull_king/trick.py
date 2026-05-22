from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from skull_king.cards import Card, CardType, Suit, TigressMode, TRUMP_SUIT

# ── C extension (built separately; falls back to pure Python gracefully) ──
try:
    from skull_king._core.skull_king_core import (
        resolve_trick     as _c_resolve_trick,
        legal_cards_mask  as _c_legal_cards_mask,
    )
    _C_AVAILABLE = True
except ImportError:
    _C_AVAILABLE = False

# Integer encodings shared with skull_king_core.c (must stay in sync).
_CT_INT: dict[CardType, int] = {
    CardType.NUMBERED:   0,
    CardType.ESCAPE:     1,
    CardType.PIRATE:     2,
    CardType.MERMAID:    3,
    CardType.SKULL_KING: 4,
    CardType.TIGRESS:    5,
}

# Fast decoders from Card._hash — avoids dict lookup per card.
# _hash 0-55 → NUMBERED (suit = _hash//14, value = _hash%14+1)
# _hash 56 → ESCAPE (CT=1), 57 → PIRATE (2), 58 → MERMAID (3),
#            59 → SKULL_KING (4), 60 → TIGRESS (5)
_HASH_TO_CT_INT: tuple[int, ...] = (
    *(0 for _ in range(56)),    # NUMBERED
    1, 2, 3, 4, 5,              # special types
)
_HASH_TO_SUIT_INT: tuple[int, ...] = (
    *(h // 14 for h in range(56)),   # 0=BLACK 1=YELLOW 2=GREEN 3=PURPLE
    *(-1 for _ in range(5)),         # specials have no suit
)
_HASH_TO_VALUE: tuple[int, ...] = (
    *(h % 14 + 1 for h in range(56)),
    *(0 for _ in range(5)),
)


def _pc_to_c_tuple(pc: PlayedCard) -> tuple:  # type: ignore[name-defined]
    """(eff_type_int, suit_int, value_int, play_order, player_index) for C."""
    et = pc.effective_type
    if et == CardType.NUMBERED:
        h = pc.card._hash
        return (_HASH_TO_CT_INT[h], _HASH_TO_SUIT_INT[h], _HASH_TO_VALUE[h],
                pc.play_order, pc.player_index)
    return (_CT_INT[et], -1, 0, pc.play_order, pc.player_index)


def _pc_to_led_c_tuple(pc: PlayedCard) -> tuple:  # type: ignore[name-defined]
    """(card_type_int, suit_int, play_order) for legal_cards_mask played list."""
    et = pc.effective_type
    if et == CardType.NUMBERED:
        h = pc.card._hash
        return (0, _HASH_TO_SUIT_INT[h], pc.play_order)
    return (1, -1, pc.play_order)   # any non-NUMBERED type (won't set led_suit)


def _card_to_c_tuple(card: Card) -> tuple:
    """(card_type_int, suit_int) for legal_cards_mask hand list."""
    h = card._hash
    return (_HASH_TO_CT_INT[h], _HASH_TO_SUIT_INT[h])

# Special card types that can trigger the Black-14 bonus when they win a trick.
_BONUS_ELIGIBLE_WINNERS = frozenset(
    {CardType.SKULL_KING, CardType.MERMAID, CardType.PIRATE}
)


@dataclass(frozen=True)
class PlayedCard:
    """A card as played in a trick, including player identity and Tigress declaration."""

    card: Card
    player_index: int
    play_order: int  # 1-based; lower = played earlier; drives all tie-breaks
    tigress_mode: Optional[TigressMode] = None

    def __post_init__(self) -> None:
        if self.card.card_type == CardType.TIGRESS and self.tigress_mode is None:
            raise ValueError("Tigress must have a declared mode (PIRATE or ESCAPE) when played")
        if self.card.card_type != CardType.TIGRESS and self.tigress_mode is not None:
            raise ValueError("tigress_mode is only valid for the Tigress card")
        if self.play_order < 1:
            raise ValueError(f"play_order must be >= 1, got {self.play_order}")

    @property
    def effective_type(self) -> CardType:
        """Tigress resolves to PIRATE or ESCAPE; all other cards return their own type."""
        if self.card.card_type == CardType.TIGRESS:
            return (
                CardType.PIRATE
                if self.tigress_mode == TigressMode.PIRATE
                else CardType.ESCAPE
            )
        return self.card.card_type


@dataclass(frozen=True)
class TrickResult:
    winner_player_index: int
    winner_played_card: PlayedCard
    bonus_points: int  # earned by the winner if they hit their bid this round


@dataclass
class Trick:
    played_cards: list[PlayedCard] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_card(self, played_card: PlayedCard) -> None:
        self.played_cards.append(played_card)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def led_suit(self) -> Optional[Suit]:
        """The suit established by the first numbered card played, or None."""
        if not self.played_cards:
            return None
        first = min(self.played_cards, key=lambda c: c.play_order)
        if first.card.card_type == CardType.NUMBERED:
            return first.card.suit
        return None

    def legal_cards(self, hand: list[Card]) -> list[Card]:
        """Cards from *hand* that are legal to play given the current trick state.

        Rules (spec §4):
        - No led suit → any card is legal.
        - Special cards are always legal (they carry no suit).
        - Numbered cards of led suit are legal; if the player holds any, they MUST
          play one (or a special). Off-suit including Black are only allowed when
          the player is void in the led suit.
        """
        if not self.played_cards:
            return list(hand)

        if _C_AVAILABLE:
            hand_data   = [_card_to_c_tuple(c) for c in hand]
            played_data = [_pc_to_led_c_tuple(pc) for pc in self.played_cards]
            mask = _c_legal_cards_mask(hand_data, played_data)
            return [c for c, m in zip(hand, mask) if m]

        # Pure-Python fallback
        suit = self.led_suit
        if suit is None:
            return list(hand)

        specials = [c for c in hand if c.is_special]
        suited = [c for c in hand if c.card_type == CardType.NUMBERED and c.suit == suit]

        if suited:
            return suited + specials
        return list(hand)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> TrickResult:
        """Determine the trick winner following the spec §5 algorithm exactly."""
        if not self.played_cards:
            raise ValueError("Cannot resolve an empty trick")

        if _C_AVAILABLE:
            cards_data = [_pc_to_c_tuple(pc) for pc in self.played_cards]
            winner_player, bonus = _c_resolve_trick(cards_data)
            winner_pc = next(
                pc for pc in self.played_cards if pc.player_index == winner_player
            )
            return TrickResult(
                winner_player_index=winner_player,
                winner_played_card=winner_pc,
                bonus_points=bonus,
            )

        cards = self.played_cards
        has_sk = any(c.effective_type == CardType.SKULL_KING for c in cards)
        has_mermaid = any(c.effective_type == CardType.MERMAID for c in cards)
        has_pirate = any(c.effective_type == CardType.PIRATE for c in cards)

        # Step 1: SK + Mermaid, no Pirate → Mermaid wins (spec §3 interaction matrix)
        if has_sk and has_mermaid and not has_pirate:
            winner = self._first_of(CardType.MERMAID)

        # Step 2: SK present, and either no Mermaid or a Pirate is also present.
        # Pirate beats Mermaid → Mermaid loses her ability to counter SK → SK wins.
        elif has_sk:
            winner = self._first_of(CardType.SKULL_KING)

        # Step 3: Pirate(s), no SK
        elif has_pirate:
            winner = self._first_of(CardType.PIRATE)

        # Step 4: Mermaid only (no SK, no Pirate)
        elif has_mermaid:
            winner = self._first_of(CardType.MERMAID)

        # Step 5: Only numbered cards and/or Escapes
        else:
            winner = self._resolve_numbered()

        bonus = self._compute_bonuses(winner)
        return TrickResult(
            winner_player_index=winner.player_index,
            winner_played_card=winner,
            bonus_points=bonus,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _first_of(self, card_type: CardType) -> PlayedCard:
        """Return the earliest-played card with the given effective type."""
        matches = [c for c in self.played_cards if c.effective_type == card_type]
        return min(matches, key=lambda c: c.play_order)

    def _resolve_numbered(self) -> PlayedCard:
        """Resolve a trick that contains no SK / Pirate / Mermaid specials."""
        cards = self.played_cards
        non_escapes = [c for c in cards if c.effective_type != CardType.ESCAPE]

        # All Escapes → trick leader wins (spec §9.1)
        if not non_escapes:
            return min(cards, key=lambda c: c.play_order)

        led_suit = self.led_suit

        if led_suit is None:
            # Escape led (spec §9.2): highest Black wins; otherwise the first
            # non-Escape colored card establishes an informal led suit.
            black = [c for c in non_escapes if c.card.suit == TRUMP_SUIT]
            if black:
                return max(black, key=lambda c: c.card.numeric_value)
            first_colored = min(non_escapes, key=lambda c: c.play_order)
            led_suit = first_colored.card.suit

        # Black (trump) beats all colored numbered cards (spec §4.3)
        black = [
            c for c in cards
            if c.card.card_type == CardType.NUMBERED and c.card.suit == TRUMP_SUIT
        ]
        if black:
            return max(black, key=lambda c: c.card.numeric_value)

        # Highest card of led suit wins (spec §5 step 5b)
        led = [
            c for c in cards
            if c.card.card_type == CardType.NUMBERED and c.card.suit == led_suit
        ]
        return max(led, key=lambda c: c.card.numeric_value)

    def _compute_bonuses(self, winner: PlayedCard) -> int:
        """Compute bonus points earned by the trick winner (spec §6.2 and §6.3).

        Bonuses are stored here; whether they are actually applied to the player's
        score depends on whether the player hit their bid — enforced in RoundScore.
        """
        bonus = 0
        cards = self.played_cards
        wtype = winner.effective_type

        # Black 14 bonus: +20, but ONLY if a special card won (spec CONFIRMED-05)
        if wtype in _BONUS_ELIGIBLE_WINNERS:
            if any(
                c.card.card_type == CardType.NUMBERED
                and c.card.suit == TRUMP_SUIT
                and c.card.value == 14
                for c in cards
            ):
                bonus += 20

        # Skull King captures Pirates: +30 per Pirate (spec §6.2)
        if wtype == CardType.SKULL_KING:
            pirate_count = sum(1 for c in cards if c.effective_type == CardType.PIRATE)
            bonus += 30 * pirate_count

        # Mermaid captures Skull King: +40 (spec §6.2)
        if wtype == CardType.MERMAID:
            if any(c.card.card_type == CardType.SKULL_KING for c in cards):
                bonus += 40

        return bonus
