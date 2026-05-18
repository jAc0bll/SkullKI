"""Tests for GameState: immutability, serialization, and partial observation."""
import json
import pytest
from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.game_state import (
    FrozenPlayerState,
    GamePhase,
    GameState,
    _card_to_dict,
    _card_from_dict,
    _played_card_to_dict,
    _played_card_from_dict,
)
from skull_king.trick import PlayedCard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_player(index: int, score: int = 0, bid: int | None = None) -> FrozenPlayerState:
    return FrozenPlayerState(
        player_index=index,
        hand=(),
        bid=bid,
        tricks_won_this_round=0,
        accumulated_bonus=0,
        total_score=score,
    )


def make_state(
    n_players: int = 2,
    round_number: int = 3,
    trick_number: int = 1,
    phase: GamePhase = GamePhase.PLAYING,
) -> GameState:
    return GameState(
        round_number=round_number,
        trick_number=trick_number,
        phase=phase,
        n_players=n_players,
        current_player_index=0,
        trick_leader_index=0,
        player_states=tuple(make_player(i) for i in range(n_players)),
        current_trick_cards=(),
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_game_state_frozen(self):
        gs = make_state()
        with pytest.raises(Exception):
            gs.round_number = 99  # type: ignore[misc]

    def test_frozen_player_state_frozen(self):
        ps = make_player(0)
        with pytest.raises(Exception):
            ps.bid = 5  # type: ignore[misc]

    def test_played_card_frozen(self):
        pc = PlayedCard(
            card=Card(card_type=CardType.ESCAPE),
            player_index=0,
            play_order=1,
        )
        with pytest.raises(Exception):
            pc.play_order = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_scores(self):
        gs = GameState(
            round_number=1, trick_number=1, phase=GamePhase.PLAYING,
            n_players=3, current_player_index=0, trick_leader_index=0,
            player_states=(make_player(0, 100), make_player(1, -20), make_player(2, 50)),
            current_trick_cards=(),
        )
        assert gs.scores == (100, -20, 50)

    def test_bids(self):
        gs = GameState(
            round_number=2, trick_number=1, phase=GamePhase.PLAYING,
            n_players=2, current_player_index=0, trick_leader_index=0,
            player_states=(make_player(0, bid=2), make_player(1, bid=None)),
            current_trick_cards=(),
        )
        assert gs.bids == (2, None)

    def test_current_player(self):
        gs = make_state(n_players=3)
        # default current_player_index=0
        assert gs.current_player.player_index == 0


# ---------------------------------------------------------------------------
# Serialization — card helpers
# ---------------------------------------------------------------------------


class TestCardSerialization:
    def test_numbered_card_round_trip(self):
        c = Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=7)
        assert _card_from_dict(_card_to_dict(c)) == c

    def test_special_card_round_trip(self):
        for ct in (CardType.ESCAPE, CardType.PIRATE, CardType.MERMAID,
                   CardType.SKULL_KING, CardType.TIGRESS):
            c = Card(card_type=ct)
            assert _card_from_dict(_card_to_dict(c)) == c


class TestPlayedCardSerialization:
    def test_regular_card_round_trip(self):
        pc = PlayedCard(
            card=Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=14),
            player_index=2,
            play_order=3,
        )
        assert _played_card_from_dict(_played_card_to_dict(pc)) == pc

    def test_tigress_round_trip(self):
        pc = PlayedCard(
            card=Card(card_type=CardType.TIGRESS),
            player_index=1,
            play_order=2,
            tigress_mode=TigressMode.PIRATE,
        )
        restored = _played_card_from_dict(_played_card_to_dict(pc))
        assert restored == pc
        assert restored.tigress_mode == TigressMode.PIRATE


# ---------------------------------------------------------------------------
# Full GameState serialization round-trip
# ---------------------------------------------------------------------------


class TestGameStateSerialization:
    def _full_state(self) -> GameState:
        pc = PlayedCard(
            card=Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=9),
            player_index=1,
            play_order=1,
        )
        return GameState(
            round_number=5,
            trick_number=3,
            phase=GamePhase.PLAYING,
            n_players=3,
            current_player_index=2,
            trick_leader_index=1,
            player_states=(
                FrozenPlayerState(
                    player_index=0,
                    hand=(Card(card_type=CardType.NUMBERED, suit=Suit.GREEN, value=4),),
                    bid=2,
                    tricks_won_this_round=1,
                    accumulated_bonus=30,
                    total_score=80,
                ),
                make_player(1, score=60, bid=3),
                make_player(2, score=40),
            ),
            current_trick_cards=(pc,),
        )

    def test_to_dict_is_json_serializable(self):
        gs = self._full_state()
        d = gs.to_dict()
        json.dumps(d)  # must not raise

    def test_to_json_and_back(self):
        gs = self._full_state()
        restored = GameState.from_dict(json.loads(gs.to_json()))
        assert restored == gs

    def test_round_trip_preserves_phase(self):
        for phase in GamePhase:
            gs = make_state(phase=phase)
            restored = GameState.from_dict(gs.to_dict())
            assert restored.phase == phase

    def test_round_trip_preserves_trick_cards(self):
        pc = PlayedCard(
            card=Card(card_type=CardType.TIGRESS),
            player_index=0,
            play_order=1,
            tigress_mode=TigressMode.ESCAPE,
        )
        gs = GameState(
            round_number=1, trick_number=1, phase=GamePhase.PLAYING,
            n_players=2, current_player_index=1, trick_leader_index=0,
            player_states=(make_player(0), make_player(1)),
            current_trick_cards=(pc,),
        )
        restored = GameState.from_dict(gs.to_dict())
        assert restored.current_trick_cards[0].tigress_mode == TigressMode.ESCAPE


# ---------------------------------------------------------------------------
# Partial observation (hidden hands)
# ---------------------------------------------------------------------------


class TestObservationForPlayer:
    def _state_with_hands(self) -> GameState:
        hand0 = (Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=7),)
        hand1 = (Card(card_type=CardType.PIRATE),)
        return GameState(
            round_number=2, trick_number=1, phase=GamePhase.PLAYING,
            n_players=2, current_player_index=0, trick_leader_index=0,
            player_states=(
                FrozenPlayerState(0, hand0, bid=1, tricks_won_this_round=0,
                                  accumulated_bonus=0, total_score=0),
                FrozenPlayerState(1, hand1, bid=2, tricks_won_this_round=0,
                                  accumulated_bonus=0, total_score=0),
            ),
            current_trick_cards=(),
        )

    def test_own_hand_visible(self):
        gs = self._state_with_hands()
        obs = gs.observation_for(player_index=0)
        own_hand = obs["player_states"][0]["hand"]
        assert len(own_hand) == 1

    def test_opponent_hand_hidden(self):
        gs = self._state_with_hands()
        obs = gs.observation_for(player_index=0)
        opponent_hand = obs["player_states"][1]["hand"]
        assert opponent_hand == []

    def test_bids_still_visible(self):
        gs = self._state_with_hands()
        obs = gs.observation_for(player_index=0)
        assert obs["player_states"][1]["bid"] == 2

    def test_original_state_not_mutated(self):
        """observation_for must not modify the original GameState."""
        gs = self._state_with_hands()
        gs.observation_for(player_index=0)
        # Original opponent hand still present
        assert len(gs.player_states[1].hand) == 1
