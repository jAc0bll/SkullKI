from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from skull_king.cards import Card, CardType, Deck, TigressMode, MAX_PLAYERS, NUM_ROUNDS
from skull_king.game_state import FrozenPlayerState, GamePhase, GameState
from skull_king.player import PlayerState
from skull_king.trick import PlayedCard, Trick


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a player action violates game rules."""


class Validator:
    @staticmethod
    def validate_bid(
        player_index: int,
        bid: int,
        round_number: int,
        phase: GamePhase,
        bids_placed: frozenset[int],
        n_players: int,
    ) -> None:
        if phase != GamePhase.BIDDING:
            raise ValidationError(
                f"Cannot bid: game is in {phase.value} phase, not BIDDING"
            )
        if not (0 <= player_index < n_players):
            raise ValidationError(
                f"Player index {player_index} out of range [0, {n_players - 1}]"
            )
        if player_index in bids_placed:
            raise ValidationError(
                f"Player {player_index} has already bid this round"
            )
        if not (0 <= bid <= round_number):
            raise ValidationError(
                f"Bid {bid} out of range [0, {round_number}] for round {round_number}"
            )

    @staticmethod
    def validate_play(
        player_index: int,
        card: Card,
        tigress_mode: Optional[TigressMode],
        hand: list[Card],
        current_trick: Trick,
        expected_player: int,
        phase: GamePhase,
    ) -> None:
        if phase != GamePhase.PLAYING:
            raise ValidationError(
                f"Cannot play a card: game is in {phase.value} phase, not PLAYING"
            )
        if player_index != expected_player:
            raise ValidationError(
                f"It is player {expected_player}'s turn, not player {player_index}'s"
            )
        if card not in hand:
            raise ValidationError(
                f"{card!r} is not in player {player_index}'s hand"
            )
        if card.card_type == CardType.TIGRESS and tigress_mode is None:
            raise ValidationError(
                "Tigress must be played with a declared mode (PIRATE or ESCAPE)"
            )
        if card.card_type != CardType.TIGRESS and tigress_mode is not None:
            raise ValidationError(
                f"{card!r} is not the Tigress card; tigress_mode must be None"
            )
        legal = current_trick.legal_cards(hand)
        if card not in legal:
            led = current_trick.led_suit
            raise ValidationError(
                f"{card!r} is not a legal play "
                f"(led suit is {led.value if led else 'none'}; you must follow suit)"
            )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class GameEngine:
    """Full 10-round Skull King game engine.

    Usage::

        engine = GameEngine(n_players=3, seed=42)
        state = engine.start()

        # BIDDING phase — all players bid in any order
        state = engine.place_bid(0, 2)
        state = engine.place_bid(1, 1)
        state = engine.place_bid(2, 0)

        # PLAYING phase — current_player_index tells you whose turn it is
        while state.phase == GamePhase.PLAYING:
            p = state.current_player_index
            hand = list(state.player_states[p].hand)
            legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
            state = engine.play_card(p, legal[0])

        # Repeat for each round until state.phase == GamePhase.GAME_OVER
    """

    n_players: int
    seed: int = 0

    # Internal mutable state — never expose these directly.
    _players: list[PlayerState] = field(init=False)
    _round: int = field(init=False, default=1)
    _trick_in_round: int = field(init=False, default=1)
    _trick_leader: int = field(init=False, default=0)
    _play_count: int = field(init=False, default=0)
    _current_trick: Trick = field(init=False)
    _completed_tricks: list = field(init=False, default_factory=list)
    _phase: GamePhase = field(init=False, default=GamePhase.BIDDING)
    _bids_placed: set[int] = field(init=False)
    _rng: random.Random = field(init=False)
    _started: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not (2 <= self.n_players <= MAX_PLAYERS):
            raise ValueError(
                f"n_players must be between 2 and {MAX_PLAYERS}, got {self.n_players}"
            )
        self._players = [PlayerState(i) for i in range(self.n_players)]
        self._current_trick = Trick()
        self._bids_placed = set()
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> GameState:
        """Deal round 1 cards and enter the bidding phase. Must be called exactly once."""
        if self._started:
            raise RuntimeError("Game has already started; cannot call start() again")
        self._started = True
        self._deal_round()
        return self.get_state()

    def place_bid(self, player_index: int, bid: int) -> GameState:
        """Record a bid for *player_index*. Legal during BIDDING phase only."""
        Validator.validate_bid(
            player_index, bid, self._round, self._phase,
            frozenset(self._bids_placed), self.n_players,
        )
        self._players[player_index].place_bid(bid, self._round)
        self._bids_placed.add(player_index)

        if len(self._bids_placed) == self.n_players:
            self._phase = GamePhase.PLAYING

        return self.get_state()

    def play_card(
        self,
        player_index: int,
        card: Card,
        tigress_mode: Optional[TigressMode] = None,
    ) -> GameState:
        """Play *card* for *player_index*. Legal during PLAYING phase only."""
        expected = self._current_player_index()
        Validator.validate_play(
            player_index, card, tigress_mode,
            self._players[player_index].hand,
            self._current_trick, expected, self._phase,
        )

        pc = PlayedCard(
            card=card,
            player_index=player_index,
            play_order=self._play_count + 1,
            tigress_mode=tigress_mode,
        )
        self._current_trick.add_card(pc)
        self._players[player_index].hand.remove(card)
        self._play_count += 1

        if self._play_count == self.n_players:
            self._resolve_trick()

        return self.get_state()

    def get_state(self) -> GameState:
        """Return an immutable snapshot of the current game state."""
        return GameState(
            round_number=self._round,
            trick_number=self._trick_in_round,
            phase=self._phase,
            n_players=self.n_players,
            current_player_index=self._current_player_index(),
            trick_leader_index=self._trick_leader,
            player_states=tuple(self._freeze(p) for p in self._players),
            current_trick_cards=tuple(self._current_trick.played_cards),
        )

    @property
    def completed_tricks_this_round(self) -> list:
        """Tricks already resolved this round (not the current trick)."""
        return list(self._completed_tricks)

    # ------------------------------------------------------------------
    # Testing support
    # ------------------------------------------------------------------

    def _inject_hands(self, hands: list[list[Card]]) -> None:
        """Replace current hands with known cards. For deterministic testing only.

        Call during BIDDING phase after start() and before any bids are placed.
        """
        if len(hands) != self.n_players:
            raise ValueError(
                f"Expected {self.n_players} hands, got {len(hands)}"
            )
        for player, hand in zip(self._players, hands):
            player.set_hand(list(hand))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_player_index(self) -> int:
        if self._phase == GamePhase.BIDDING:
            unbid = [i for i in range(self.n_players) if i not in self._bids_placed]
            return unbid[0] if unbid else 0
        if self._phase == GamePhase.PLAYING:
            return (self._trick_leader + self._play_count) % self.n_players
        return 0  # GAME_OVER sentinel

    def _deal_round(self) -> None:
        deck = Deck()
        deck.shuffle(seed=self._rng.randint(0, 2**31 - 1))
        hands, _ = deck.deal(self.n_players, self._round)
        for player, hand in zip(self._players, hands):
            player.set_hand(hand)

    def _resolve_trick(self) -> None:
        result = self._current_trick.resolve()
        winner = result.winner_player_index
        self._players[winner].record_trick_win(result.bonus_points)
        self._completed_tricks.append(self._current_trick)
        self._trick_leader = winner
        self._trick_in_round += 1
        self._play_count = 0
        self._current_trick = Trick()

        if self._trick_in_round > self._round:
            self._end_round()

    def _end_round(self) -> None:
        for p in self._players:
            p.finalize_round(self._round)

        if self._round == NUM_ROUNDS:
            self._phase = GamePhase.GAME_OVER
        else:
            self._round += 1
            self._trick_in_round = 1
            self._bids_placed = set()
            self._completed_tricks = []
            for p in self._players:
                p.reset_for_round()
            self._deal_round()
            self._phase = GamePhase.BIDDING

    def _freeze(self, p: PlayerState) -> FrozenPlayerState:
        return FrozenPlayerState(
            player_index=p.player_index,
            hand=tuple(p.hand),
            bid=p.bid,
            tricks_won_this_round=p.tricks_won_this_round,
            accumulated_bonus=p.accumulated_bonus,
            total_score=p.total_score,
        )
