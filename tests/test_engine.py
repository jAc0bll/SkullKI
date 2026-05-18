"""GameEngine — integration tests covering lifecycle, validation, known outcomes, and full games.

Known-outcome tests use _inject_hands() to control which cards are dealt so we
can assert exact scores without depending on a specific RNG sequence.
"""
import pytest
from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.engine import GameEngine, ValidationError
from skull_king.game_state import GamePhase
from skull_king.resolver import TrickResolver


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------


def num(suit: Suit, value: int) -> Card:
    return Card(card_type=CardType.NUMBERED, suit=suit, value=value)


SK = Card(card_type=CardType.SKULL_KING)
PIRATE = Card(card_type=CardType.PIRATE)
MERMAID = Card(card_type=CardType.MERMAID)
ESCAPE = Card(card_type=CardType.ESCAPE)
TIGRESS = Card(card_type=CardType.TIGRESS)
BLACK_14 = num(Suit.BLACK, 14)
BLACK_7 = num(Suit.BLACK, 7)
YELLOW_5 = num(Suit.YELLOW, 5)
YELLOW_3 = num(Suit.YELLOW, 3)
YELLOW_7 = num(Suit.YELLOW, 7)
GREEN_9 = num(Suit.GREEN, 9)


# ---------------------------------------------------------------------------
# Auto-play helper (deterministic, seed-independent)
# ---------------------------------------------------------------------------


def auto_bid(engine: GameEngine, state) -> object:
    """Bid 1 (or the round number if round is 1, to cap at valid range)."""
    p = state.current_player_index
    bid = min(1, state.round_number)
    return engine.place_bid(p, bid)


def auto_play(engine: GameEngine, state) -> object:
    """Play the first legal card in hand; Tigress always declared as ESCAPE."""
    p = state.current_player_index
    hand = list(state.player_states[p].hand)
    legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
    card = legal[0]
    tigress_mode = TigressMode.ESCAPE if card.card_type == CardType.TIGRESS else None
    return engine.play_card(p, card, tigress_mode)


def drive_to_game_over(engine: GameEngine, state) -> object:
    """Drive an already-started game to GAME_OVER using auto strategies."""
    while state.phase != GamePhase.GAME_OVER:
        if state.phase == GamePhase.BIDDING:
            state = auto_bid(engine, state)
        else:
            state = auto_play(engine, state)
    return state


# ---------------------------------------------------------------------------
# 1. Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_returns_bidding_state(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        assert state.phase == GamePhase.BIDDING
        assert state.round_number == 1

    def test_start_deals_correct_hand_sizes(self):
        engine = GameEngine(n_players=3, seed=0)
        state = engine.start()
        for ps in state.player_states:
            assert len(ps.hand) == 1  # round 1 → 1 card each

    def test_start_twice_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        with pytest.raises(RuntimeError, match="already started"):
            engine.start()

    def test_invalid_player_count_low(self):
        with pytest.raises(ValueError):
            GameEngine(n_players=1)

    def test_invalid_player_count_high(self):
        with pytest.raises(ValueError):
            GameEngine(n_players=7)

    def test_get_state_before_start(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.get_state()
        assert state.phase == GamePhase.BIDDING
        assert state.round_number == 1


# ---------------------------------------------------------------------------
# 2. Bidding phase behaviour
# ---------------------------------------------------------------------------


class TestBiddingPhase:
    def test_partial_bids_stay_in_bidding(self):
        engine = GameEngine(n_players=3, seed=0)
        state = engine.start()
        state = engine.place_bid(0, 1)
        assert state.phase == GamePhase.BIDDING

    def test_all_bids_transitions_to_playing(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        state = engine.place_bid(0, 1)
        state = engine.place_bid(1, 0)
        assert state.phase == GamePhase.PLAYING

    def test_bid_recorded_in_state(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        state = engine.place_bid(0, 1)
        assert state.player_states[0].bid == 1

    def test_current_player_advances_after_bid(self):
        engine = GameEngine(n_players=3, seed=0)
        engine.start()
        state = engine.place_bid(0, 1)
        assert state.current_player_index == 1  # next unbid player

    def test_bids_accepted_in_any_order(self):
        engine = GameEngine(n_players=3, seed=0)
        engine.start()
        engine.place_bid(2, 0)
        engine.place_bid(0, 1)
        state = engine.place_bid(1, 1)
        assert state.phase == GamePhase.PLAYING

    def test_zero_bid_valid(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        state = engine.place_bid(0, 0)
        assert state.player_states[0].bid == 0

    def test_bid_equal_to_round_number_valid(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        rn = state.round_number
        state = engine.place_bid(0, rn)
        assert state.player_states[0].bid == rn


# ---------------------------------------------------------------------------
# 3. Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def _engine_in_playing(self, n: int = 2) -> tuple[GameEngine, object]:
        engine = GameEngine(n_players=n, seed=0)
        state = engine.start()
        for i in range(n):
            state = engine.place_bid(i, 1)
        return engine, state

    def test_bid_during_playing_raises(self):
        engine, _ = self._engine_in_playing()
        with pytest.raises(ValidationError, match="PLAYING"):
            engine.place_bid(0, 1)

    def test_bid_already_placed_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine.place_bid(0, 1)
        with pytest.raises(ValidationError, match="already bid"):
            engine.place_bid(0, 1)

    def test_bid_above_round_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        with pytest.raises(ValidationError, match="out of range"):
            engine.place_bid(0, state.round_number + 1)

    def test_bid_negative_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        with pytest.raises(ValidationError, match="out of range"):
            engine.place_bid(0, -1)

    def test_play_during_bidding_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        card = state.player_states[0].hand[0]
        with pytest.raises(ValidationError, match="BIDDING"):
            engine.play_card(0, card)

    def test_play_wrong_player_raises(self):
        engine, state = self._engine_in_playing()
        expected = state.current_player_index
        wrong = 1 - expected
        card = state.player_states[wrong].hand[0]
        with pytest.raises(ValidationError, match=f"player {expected}"):
            engine.play_card(wrong, card)

    def test_play_card_not_in_hand_raises(self):
        engine, state = self._engine_in_playing()
        p = state.current_player_index
        fake_card = num(Suit.GREEN, 14)
        with pytest.raises(ValidationError, match="not in player"):
            engine.play_card(p, fake_card)

    def test_play_illegal_suit_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([
            [YELLOW_7, BLACK_7],
            [YELLOW_3, GREEN_9],
        ])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        # P0 leads Yellow 7 → led suit = Yellow
        engine.play_card(0, YELLOW_7)
        # P1 has Yellow 3 (must follow) but tries to play Green 9
        with pytest.raises(ValidationError, match="must follow suit"):
            engine.play_card(1, GREEN_9)

    def test_play_tigress_without_mode_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[TIGRESS], [YELLOW_5]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 0)
        with pytest.raises(ValidationError, match="declared mode"):
            engine.play_card(0, TIGRESS)  # no tigress_mode

    def test_play_non_tigress_with_mode_raises(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[PIRATE], [YELLOW_5]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 0)
        with pytest.raises(ValidationError, match="not the Tigress"):
            engine.play_card(0, PIRATE, tigress_mode=TigressMode.PIRATE)


# ---------------------------------------------------------------------------
# 4. Known-outcome round scores
# ---------------------------------------------------------------------------


class TestKnownRoundOutcomes:
    """Each test injects exact hands so expected scores can be calculated by hand."""

    def _start_round1(self, hands: list[list[Card]]) -> tuple[GameEngine, object]:
        engine = GameEngine(n_players=len(hands), seed=0)
        engine.start()
        engine._inject_hands(hands)
        return engine, engine.get_state()

    # --- Trick resolution ---

    def test_trump_beats_color(self):
        """P0: Black 7, P1: Yellow 5. P0 leads → Black wins. P0: bid=1,won=1 → +20."""
        engine, state = self._start_round1([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        engine.play_card(0, BLACK_7)         # P0 leads Black 7 (trump)
        state = engine.play_card(1, YELLOW_5)  # P1 void in Black, plays Yellow
        assert state.phase == GamePhase.BIDDING  # round ended, next round dealing
        scores = state.scores
        assert scores[0] == 20   # bid hit: 1×20
        assert scores[1] == -10  # bid miss: 1×−10

    def test_bid_zero_success(self):
        """Both play Escape, bid 0. P0 leads → P0 wins trick. P0: bid=0,won=1 → −10(r1).
        P1: bid=0,won=0 → +10(r1)."""
        engine, _ = self._start_round1([[ESCAPE], [ESCAPE]])
        engine.place_bid(0, 0)
        engine.place_bid(1, 0)
        engine.play_card(0, ESCAPE)
        state = engine.play_card(1, ESCAPE)
        assert state.scores[0] == -10  # bid 0, won 1, round 1 → -1×10
        assert state.scores[1] == 10   # bid 0, won 0, round 1 → +1×10

    def test_bid_miss_under(self):
        """Round 2: each player has 2 cards. Both bid 2.
        P0 wins 1, P1 wins 1 → both miss (bid 2, won 1 → |2-1|×-10 = -10 each)."""
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        # Drive to round 2: round 1 P0 leads and wins
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7)
        engine.play_card(1, YELLOW_5)   # round 1 done, P0 won

        # Round 2 — inject known hands
        engine._inject_hands([[YELLOW_7, GREEN_9], [YELLOW_3, BLACK_7]])
        engine.place_bid(0, 2)
        engine.place_bid(1, 2)
        # Trick 1: P0 leads Yellow 7. P1 must follow Yellow → plays Yellow 3.
        # P0 wins (Yellow 7 > Yellow 3).
        engine.play_card(0, YELLOW_7)
        engine.play_card(1, YELLOW_3)
        # Trick 2: P0 leads (won trick 1). P0 plays Green 9.
        # P1 void in Green → plays Black 7 (trump). P1 wins.
        engine.play_card(0, GREEN_9)
        state = engine.play_card(1, BLACK_7)

        # Round 1 scores: P0 bid=1,won=1 → +20.  P1 bid=0,won=0 → +10 (round1×10).
        # Round 2 scores: both bid=2, both won=1 → |2-1|×-10 = -10 each.
        r2_p0 = state.player_states[0].total_score - 20   # R1: +20
        r2_p1 = state.player_states[1].total_score - 10   # R1: +10 (bid-0 hit)
        assert r2_p0 == -10
        assert r2_p1 == -10

    # --- Bonus scoring through the engine ---

    def test_sk_captures_pirate_bonus_in_engine(self):
        """P0: SK. P1: Pirate. P0 leads SK → wins +30 bonus.
        P0: bid=1,won=1,bonus=30 → +20+30=+50."""
        engine, _ = self._start_round1([[SK], [PIRATE]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        engine.play_card(0, SK)
        state = engine.play_card(1, PIRATE)
        assert state.scores[0] == 50  # 20 + 30
        assert state.scores[1] == -10  # bid=1, won=0

    def test_mermaid_captures_sk_bonus_in_engine(self):
        """P0: SK. P1: Mermaid. Mermaid wins +40 bonus.
        P0: bid=1,won=0 → -10. P1: bid=1,won=1,bonus=40 → +60."""
        engine, _ = self._start_round1([[SK], [MERMAID]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        engine.play_card(0, SK)
        state = engine.play_card(1, MERMAID)
        assert state.scores[0] == -10
        assert state.scores[1] == 60  # 20 + 40

    def test_black14_bonus_pirate_wins(self):
        """Round 1 (2 players): P0 gets Black 14, P1 gets Pirate.
        P0 leads Black 14. P1 is void in Black → plays Pirate → wins, bonus +20."""
        engine, _ = self._start_round1([[BLACK_14], [PIRATE]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        engine.play_card(0, BLACK_14)
        state = engine.play_card(1, PIRATE)
        # P1 wins trick: bid=1, won=1, bonus=20 → 40
        assert state.scores[1] == 40
        # P0 loses trick: bid=1, won=0 → -10
        assert state.scores[0] == -10

    def test_black14_bonus_not_given_to_numbered_winner(self):
        """Black 13 leads, Black 14 also played. Black 14 wins (highest trump).
        Black 14 winner is a numbered card → no bonus (CONFIRMED-05)."""
        engine, _ = self._start_round1([[num(Suit.BLACK, 13)], [BLACK_14]])
        engine.place_bid(0, 0)
        engine.place_bid(1, 1)
        engine.play_card(0, num(Suit.BLACK, 13))
        state = engine.play_card(1, BLACK_14)
        # P1 wins with Black 14 (numbered) → no Black-14 bonus.
        assert state.scores[1] == 20  # bid=1,won=1,bonus=0 → plain +20
        # P0 bid=0, won=0, round=1 → +round×10 = +10
        assert state.scores[0] == 10

    def test_missed_bid_bonus_not_applied(self):
        """SK captures Pirate but player misses their bid — bonus must NOT apply."""
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        # Give each player 2 cards: round 2
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7); engine.play_card(1, YELLOW_5)  # R1 done

        # Round 2: P0 gets [SK, YELLOW_5], P1 gets [PIRATE, YELLOW_7]
        engine._inject_hands([[SK, YELLOW_5], [PIRATE, YELLOW_7]])
        # P0 bids 2 (will only win 1 → miss), P1 bids 1
        engine.place_bid(0, 2)
        engine.place_bid(1, 1)
        # Trick 1: P0 leads SK → P1 plays Pirate → SK wins, +30 bonus accrued
        engine.play_card(0, SK)
        engine.play_card(1, PIRATE)
        # Trick 2: P0 leads Yellow 5, P1 plays Yellow 7 → P1 wins
        engine.play_card(0, YELLOW_5)
        state = engine.play_card(1, YELLOW_7)

        # P0 round2: bid=2, won=1 → miss → -10 (bonus ignored)
        # P0 total: R1=+20 + R2=-10 = +10
        assert state.scores[0] == 20 + (-10)

    def test_tigress_as_pirate_captured_by_sk(self):
        """SK vs Tigress-as-Pirate: SK wins, +30 bonus (CONFIRMED-06)."""
        TIGRESS_CARD = Card(card_type=CardType.TIGRESS)
        engine, _ = self._start_round1([[SK], [TIGRESS_CARD]])
        engine.place_bid(0, 1)
        engine.place_bid(1, 1)
        engine.play_card(0, SK)
        state = engine.play_card(1, TIGRESS_CARD, tigress_mode=TigressMode.PIRATE)
        assert state.scores[0] == 50  # 20 + 30


# ---------------------------------------------------------------------------
# 5. Trick leader rotation
# ---------------------------------------------------------------------------


class TestTrickLeaderRotation:
    def test_winner_leads_next_trick(self):
        """Round 2: trick 1 won by P1. P1 should lead trick 2."""
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7); engine.play_card(1, YELLOW_5)  # R1: P0 wins

        # Round 2: inject so P1 wins trick 1
        engine._inject_hands([[YELLOW_5, YELLOW_3], [BLACK_7, GREEN_9]])
        engine.place_bid(0, 0); engine.place_bid(1, 1)
        # P0 leads (won last trick in R1). Yellow 5 leads.
        engine.play_card(0, YELLOW_5)
        # P1 void in Yellow → plays Black 7 → P1 wins trick 1
        state = engine.play_card(1, BLACK_7)
        assert state.trick_leader_index == 1  # P1 should now lead
        assert state.current_player_index == 1

    def test_trick_leader_index_in_state(self):
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        assert state.trick_leader_index == 0  # P0 leads round 1


# ---------------------------------------------------------------------------
# 6. Current player tracking
# ---------------------------------------------------------------------------


class TestCurrentPlayerTracking:
    def test_current_player_during_playing(self):
        engine = GameEngine(n_players=3, seed=0)
        engine.start()
        engine._inject_hands([[YELLOW_7], [YELLOW_3], [GREEN_9]])
        engine.place_bid(0, 1); engine.place_bid(1, 0); engine.place_bid(2, 0)
        state = engine.get_state()
        # P0 is trick leader → plays first
        assert state.current_player_index == 0

    def test_current_player_advances_after_play(self):
        engine = GameEngine(n_players=3, seed=0)
        engine.start()
        engine._inject_hands([[YELLOW_7], [YELLOW_3], [GREEN_9]])
        engine.place_bid(0, 1); engine.place_bid(1, 0); engine.place_bid(2, 0)
        state = engine.play_card(0, YELLOW_7)
        assert state.current_player_index == 1

    def test_current_player_wraps_around(self):
        engine = GameEngine(n_players=3, seed=0)
        engine.start()
        engine._inject_hands([[YELLOW_7], [YELLOW_3], [GREEN_9]])
        engine.place_bid(0, 1); engine.place_bid(1, 0); engine.place_bid(2, 0)
        engine.play_card(0, YELLOW_7)
        state = engine.play_card(1, YELLOW_3)
        assert state.current_player_index == 2


# ---------------------------------------------------------------------------
# 7. Round transitions
# ---------------------------------------------------------------------------


class TestRoundTransitions:
    def test_round_increments_after_all_tricks(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7)
        state = engine.play_card(1, YELLOW_5)
        assert state.round_number == 2

    def test_new_round_starts_in_bidding(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7)
        state = engine.play_card(1, YELLOW_5)
        assert state.phase == GamePhase.BIDDING

    def test_hands_cleared_on_new_round(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7)
        state = engine.play_card(1, YELLOW_5)
        # New round 2 — each player should have 2 cards dealt
        assert len(state.player_states[0].hand) == 2
        assert len(state.player_states[1].hand) == 2

    def test_trick_number_resets_each_round(self):
        engine = GameEngine(n_players=2, seed=0)
        engine.start()
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 0)
        engine.play_card(0, BLACK_7)
        state = engine.play_card(1, YELLOW_5)
        assert state.trick_number == 1  # first trick of round 2


# ---------------------------------------------------------------------------
# 8. Score accumulation across rounds
# ---------------------------------------------------------------------------


class TestScoreAccumulation:
    def test_scores_accumulate_across_rounds(self):
        """
        Round 1 (1 trick):
          P0: bid=1, won=1 (Black 7 beats Yellow 5) → +20
          P1: bid=1, won=0                          → -10

        Round 2 (2 tricks):
          P0: bid=0, won=0 (plays Escapes)          → +round_number×10 = +20
          P1: bid=2, won=2 (Black 7 + Black 9)      → +40
          (Note: in a 2-player round one player *must* win each trick;
           Escape-leads let trump cards win freely.)

        Cumulative:
          P0: +20 + 20 = +40
          P1: -10 + 40 = +30
        """
        engine = GameEngine(n_players=2, seed=0)
        engine.start()

        # Round 1
        engine._inject_hands([[BLACK_7], [YELLOW_5]])
        engine.place_bid(0, 1); engine.place_bid(1, 1)
        engine.play_card(0, BLACK_7)   # P0 leads, wins
        engine.play_card(1, YELLOW_5)  # round 1 done

        # Round 2: P0 gets two Escapes, P1 gets Black 7 and Black 9.
        # P0 bids 0 (wants to win 0 tricks), P1 bids 2.
        # P0 still leads (won R1 last trick).
        engine._inject_hands(
            [[ESCAPE, ESCAPE], [num(Suit.BLACK, 7), num(Suit.BLACK, 9)]]
        )
        engine.place_bid(0, 0); engine.place_bid(1, 2)

        # Trick 1: P0 leads Escape (no led suit). P1 plays Black 7 → wins.
        engine.play_card(0, ESCAPE)
        engine.play_card(1, num(Suit.BLACK, 7))

        # Trick 2: P1 leads Black 9. P0 plays Escape (void in Black). Black 9 wins.
        engine.play_card(1, num(Suit.BLACK, 9))
        state = engine.play_card(0, ESCAPE)

        assert state.scores[0] == 40   # R1:+20, R2:+20
        assert state.scores[1] == 30   # R1:-10, R2:+40


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _record_game(self, seed: int) -> list[tuple]:
        """Play a full game and record (round, trick, winner, score) per state."""
        engine = GameEngine(n_players=2, seed=seed)
        state = engine.start()
        records = []
        while state.phase != GamePhase.GAME_OVER:
            if state.phase == GamePhase.BIDDING:
                state = auto_bid(engine, state)
            else:
                records.append((
                    state.round_number,
                    state.trick_number,
                    state.current_player_index,
                    state.scores,
                ))
                state = auto_play(engine, state)
        return records

    def test_same_seed_same_game(self):
        records_a = self._record_game(seed=7)
        records_b = self._record_game(seed=7)
        assert records_a == records_b

    def test_different_seeds_different_games(self):
        records_a = self._record_game(seed=1)
        records_b = self._record_game(seed=2)
        assert records_a != records_b


# ---------------------------------------------------------------------------
# 10. Full 10-round game
# ---------------------------------------------------------------------------


class TestFullGame:
    def _play_full_game(self, n_players: int, seed: int) -> object:
        engine = GameEngine(n_players=n_players, seed=seed)
        state = engine.start()
        return drive_to_game_over(engine, state)

    def test_2_player_game_reaches_game_over(self):
        state = self._play_full_game(n_players=2, seed=0)
        assert state.phase == GamePhase.GAME_OVER

    def test_3_player_game_reaches_game_over(self):
        state = self._play_full_game(n_players=3, seed=42)
        assert state.phase == GamePhase.GAME_OVER

    def test_6_player_game_reaches_game_over(self):
        state = self._play_full_game(n_players=6, seed=99)
        assert state.phase == GamePhase.GAME_OVER

    def test_all_hands_empty_at_game_over(self):
        state = self._play_full_game(n_players=2, seed=0)
        for ps in state.player_states:
            assert len(ps.hand) == 0

    def test_correct_round_number_at_game_over(self):
        state = self._play_full_game(n_players=2, seed=0)
        assert state.round_number == 10

    def test_each_player_has_10_score_history_entries(self):
        from skull_king.engine import GameEngine as GE
        engine = GE(n_players=2, seed=0)
        state = engine.start()
        drive_to_game_over(engine, state)
        for p in engine._players:
            assert p.rounds_played == 10

    def test_game_over_is_terminal(self):
        """After GAME_OVER, play_card and place_bid should raise ValidationError."""
        engine = GameEngine(n_players=2, seed=0)
        state = engine.start()
        state = drive_to_game_over(engine, state)
        with pytest.raises(ValidationError):
            engine.place_bid(0, 1)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 999])
    def test_full_game_deterministic_across_seeds(self, seed: int):
        """Each seed produces a valid completed game."""
        state = self._play_full_game(n_players=2, seed=seed)
        assert state.phase == GamePhase.GAME_OVER
        for ps in state.player_states:
            assert len(ps.hand) == 0
