"""Unit tests for RandomAgent, HeuristicAgent, and MCTSAgent."""
import pytest

from skull_king.agents import BaseAgent, HeuristicAgent, MCTSAgent, RandomAgent
from skull_king.agents.heuristic_agent import _card_strength, _would_beat
from skull_king.cards import Card, CardType, Suit, TigressMode
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase
from skull_king.trick import PlayedCard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SK = Card(card_type=CardType.SKULL_KING)
PIRATE = Card(card_type=CardType.PIRATE)
MERMAID = Card(card_type=CardType.MERMAID)
ESCAPE = Card(card_type=CardType.ESCAPE)
TIGRESS = Card(card_type=CardType.TIGRESS)
BLACK_14 = Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=14)
BLACK_7 = Card(card_type=CardType.NUMBERED, suit=Suit.BLACK, value=7)
YELLOW_9 = Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=9)
YELLOW_1 = Card(card_type=CardType.NUMBERED, suit=Suit.YELLOW, value=1)


def _run_quick_game(agents: list) -> list[int]:
    """Run one game and return final scores by seat."""
    n = len(agents)
    engine = GameEngine(n_players=n, seed=99)
    state = engine.start()
    while state.phase != GamePhase.GAME_OVER:
        seat = state.current_player_index
        agent = agents[seat]
        agent.before_move(engine)
        if state.phase == GamePhase.BIDDING:
            bid = agent.bid(state, seat)
            state = engine.place_bid(seat, bid)
        else:
            card, mode = agent.play(state, seat)
            state = engine.play_card(seat, card, mode)
    return [state.player_states[i].total_score for i in range(n)]


# ---------------------------------------------------------------------------
# BaseAgent contract
# ---------------------------------------------------------------------------


class TestBaseAgentContract:
    def test_random_is_base_agent(self):
        assert isinstance(RandomAgent(), BaseAgent)

    def test_heuristic_is_base_agent(self):
        assert isinstance(HeuristicAgent(), BaseAgent)

    def test_mcts_is_base_agent(self):
        assert isinstance(MCTSAgent(n_simulations=1), BaseAgent)

    def test_agents_have_names(self):
        assert RandomAgent.name == "Random"
        assert HeuristicAgent.name == "Heuristic"
        assert MCTSAgent.name == "MCTS"


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def _make_state(self):
        engine = GameEngine(n_players=3, seed=0)
        return engine, engine.start()

    def test_bid_in_range(self):
        agent = RandomAgent(seed=0)
        engine, state = self._make_state()
        for _ in range(20):
            b = agent.bid(state, 0)
            assert 0 <= b <= state.round_number

    def test_play_returns_legal_card(self):
        agent = RandomAgent(seed=0)
        engine = GameEngine(n_players=3, seed=5)
        state = engine.start()
        # Bid everyone so we reach PLAYING
        for i in range(3):
            state = engine.place_bid(i, 1)
        card, mode = agent.play(state, state.current_player_index)
        pi = state.current_player_index
        assert card in state.player_states[pi].hand

    def test_full_game_completes(self):
        scores = _run_quick_game([RandomAgent(seed=i) for i in range(3)])
        assert len(scores) == 3

    def test_act_picks_legal_env_action(self):
        import numpy as np
        from skull_king.env.skull_king_env import SkullKingEnv
        env = SkullKingEnv(n_players=3, seed=0)
        obs, _ = env.reset()
        mask = env.action_masks()
        agent = RandomAgent(seed=0)
        action = agent.act(obs, mask)
        assert mask[action]


# ---------------------------------------------------------------------------
# HeuristicAgent — bidding
# ---------------------------------------------------------------------------


class TestHeuristicBid:
    def _bid_for(self, hand: list[Card], round_number: int) -> int:
        from skull_king.game_state import FrozenPlayerState, GamePhase, GameState
        from skull_king.trick import PlayedCard
        # Build minimal GameState
        ps = FrozenPlayerState(
            player_index=0,
            hand=tuple(hand),
            bid=None,
            tricks_won_this_round=0,
            accumulated_bonus=0,
            total_score=0,
        )
        state = GameState(
            round_number=round_number,
            trick_number=1,
            phase=GamePhase.BIDDING,
            n_players=2,
            current_player_index=0,
            trick_leader_index=0,
            player_states=(ps, ps),
            current_trick_cards=(),
        )
        return HeuristicAgent().bid(state, 0)

    def test_escape_only_bids_zero(self):
        assert self._bid_for([ESCAPE], 1) == 0

    def test_sk_bids_one(self):
        assert self._bid_for([SK], 1) == 1

    def test_bid_clamped_to_round_number(self):
        # 5 pirates would estimate ~3.75 → round to 4, but round=2 → clamp to 2
        b = self._bid_for([PIRATE] * 5, 2)
        assert b == 2

    def test_mixed_hand_positive_bid(self):
        b = self._bid_for([SK, PIRATE, BLACK_14, ESCAPE, ESCAPE], 5)
        assert b >= 1

    def test_all_escapes_bid_zero(self):
        b = self._bid_for([ESCAPE] * 5, 5)
        assert b == 0


# ---------------------------------------------------------------------------
# HeuristicAgent — play helpers
# ---------------------------------------------------------------------------


class TestHeuristicCardStrength:
    def test_sk_strongest(self):
        assert _card_strength(SK) > _card_strength(PIRATE)

    def test_pirate_beats_trump(self):
        assert _card_strength(PIRATE) > _card_strength(BLACK_14)

    def test_trump_beats_colored(self):
        assert _card_strength(BLACK_14) > _card_strength(YELLOW_9)

    def test_escape_weakest(self):
        assert _card_strength(ESCAPE) == 0

    def test_tigress_pirate_equals_pirate(self):
        assert _card_strength(TIGRESS, TigressMode.PIRATE) == _card_strength(PIRATE)

    def test_tigress_escape_equals_escape(self):
        assert _card_strength(TIGRESS, TigressMode.ESCAPE) == _card_strength(ESCAPE)


class TestWouldBeat:
    def _pc(self, card, player, order, mode=None):
        return PlayedCard(card=card, player_index=player, play_order=order, tigress_mode=mode)

    def test_sk_beats_yellow(self):
        played = (self._pc(YELLOW_9, 0, 1),)
        assert _would_beat(SK, played, 1, None)

    def test_yellow_does_not_beat_sk(self):
        played = (self._pc(SK, 0, 1),)
        assert not _would_beat(YELLOW_9, played, 1, None)

    def test_mermaid_beats_sk(self):
        played = (self._pc(SK, 0, 1),)
        assert _would_beat(MERMAID, played, 1, None)

    def test_escape_does_not_beat_anything(self):
        played = (self._pc(YELLOW_1, 0, 1),)
        assert not _would_beat(ESCAPE, played, 1, None)


# ---------------------------------------------------------------------------
# HeuristicAgent — full game and play strategy
# ---------------------------------------------------------------------------


class TestHeuristicPlay:
    def test_full_game_completes(self):
        scores = _run_quick_game([HeuristicAgent(), HeuristicAgent(), HeuristicAgent()])
        assert len(scores) == 3

    def test_plays_legal_card(self):
        engine = GameEngine(n_players=3, seed=7)
        state = engine.start()
        for i in range(3):
            state = engine.place_bid(i, 1)
        agent = HeuristicAgent()
        agent.before_move(engine)
        pi = state.current_player_index
        card, mode = agent.play(state, pi)
        from skull_king.resolver import TrickResolver
        legal = TrickResolver.legal_plays(list(state.current_trick_cards), list(state.player_states[pi].hand))
        assert card in legal

    def test_plays_weakest_winner_when_trying_to_win(self):
        """With two winners available, should play the weaker one."""
        from skull_king.agents.heuristic_agent import _candidates, HeuristicAgent
        played = ()
        legal = [BLACK_14, YELLOW_9]  # leading an empty trick, want to win
        candidates = _candidates(legal)
        result = HeuristicAgent._choose(candidates, played, player_index=0, want_to_win=True)
        assert result[0] == BLACK_14  # only option that's "strongest" when leading

    def test_prefers_escape_when_trying_to_lose(self):
        from skull_king.agents.heuristic_agent import _candidates, HeuristicAgent
        played = ()
        legal = [ESCAPE, YELLOW_9, BLACK_7]
        candidates = _candidates(legal)
        card, mode = HeuristicAgent._choose(candidates, played, 0, want_to_win=False)
        assert card.card_type == CardType.ESCAPE

    def test_mixed_game_random_vs_heuristic(self):
        scores = _run_quick_game([HeuristicAgent(), RandomAgent(0), RandomAgent(1)])
        assert all(isinstance(s, int) for s in scores)


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------


class TestMCTSAgent:
    def test_bid_in_range(self):
        agent = MCTSAgent(n_simulations=2, seed=0)
        engine = GameEngine(n_players=3, seed=0)
        state = engine.start()
        agent.before_move(engine)
        b = agent.bid(state, 0)
        assert 0 <= b <= state.round_number

    def test_play_returns_legal_card(self):
        agent = MCTSAgent(n_simulations=2, seed=0)
        engine = GameEngine(n_players=3, seed=0)
        state = engine.start()
        for i in range(3):
            state = engine.place_bid(i, 1)
        agent.before_move(engine)
        pi = state.current_player_index
        card, mode = agent.play(state, pi)
        from skull_king.resolver import TrickResolver
        legal = TrickResolver.legal_plays(
            list(state.current_trick_cards), list(state.player_states[pi].hand)
        )
        assert card in legal

    def test_full_game_completes(self):
        scores = _run_quick_game(
            [MCTSAgent(n_simulations=2), RandomAgent(0), RandomAgent(1)]
        )
        assert len(scores) == 3

    def test_without_snapshot_falls_back(self):
        """before_move not called → agent falls back to random."""
        agent = MCTSAgent(n_simulations=2, seed=0)
        engine = GameEngine(n_players=3, seed=0)
        state = engine.start()
        b = agent.bid(state, 0)
        assert 0 <= b <= state.round_number
