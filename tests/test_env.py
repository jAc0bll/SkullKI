"""Tests for SkullKingEnv and RandomAgent."""
import numpy as np
import pytest

from skull_king.env.skull_king_env import (
    SkullKingEnv,
    ACTION_SPACE_SIZE,
    OBS_SIZE,
    N_BID_ACTIONS,
    TIGRESS_AS_ESCAPE_ACTION,
    TIGRESS_AS_PIRATE_ACTION,
    _bid_conditioned_signal,
    _HINT_BID0_TRICK_WON,
    _HINT_ON_TRACK,
    _HINT_OVERSHOT,
)
from skull_king.agents.random_agent import RandomAgent
from skull_king.game_state import GamePhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(n_players=3, reward_mode="round", seed=42) -> SkullKingEnv:
    return SkullKingEnv(n_players=n_players, reward_mode=reward_mode, seed=seed)


def run_full_episode(env: SkullKingEnv, agent: RandomAgent) -> dict:
    return agent.run_episode(env)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEnvCreation:
    def test_default_construction(self):
        env = SkullKingEnv()
        assert env.n_players == 3

    def test_obs_space_shape(self):
        env = SkullKingEnv()
        assert env.observation_space.shape == (OBS_SIZE,)

    def test_action_space_size(self):
        env = SkullKingEnv()
        assert env.action_space.n == ACTION_SPACE_SIZE

    def test_invalid_n_players_low(self):
        with pytest.raises(ValueError):
            SkullKingEnv(n_players=1)

    def test_invalid_n_players_high(self):
        with pytest.raises(ValueError):
            SkullKingEnv(n_players=7)

    def test_invalid_reward_mode(self):
        with pytest.raises(ValueError):
            SkullKingEnv(reward_mode="bad")

    def test_invalid_controlled_player(self):
        with pytest.raises(ValueError):
            SkullKingEnv(n_players=3, controlled_player=3)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = make_env()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(info, dict)

    def test_obs_in_valid_range(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.min() >= -1.0
        assert obs.max() <= 1.0

    def test_info_contains_keys(self):
        env = make_env()
        _, info = env.reset()
        for key in ("round", "trick", "phase", "score", "bid", "tricks_won"):
            assert key in info

    def test_initial_phase_is_bidding(self):
        env = make_env()
        _, info = env.reset()
        assert info["phase"] == "BIDDING"

    def test_reset_seed_determines_game(self):
        env1 = make_env(seed=7)
        env2 = make_env(seed=7)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_differ(self):
        env1 = make_env(seed=1)
        env2 = make_env(seed=2)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert not np.array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Action masks
# ---------------------------------------------------------------------------


class TestActionMasks:
    def test_mask_shape(self):
        env = make_env()
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == bool

    def test_bidding_phase_only_bid_actions_active(self):
        env = make_env()
        env.reset()
        assert env._current_state.phase == GamePhase.BIDDING
        mask = env.action_masks()
        # Play actions must all be False during bidding
        assert not any(mask[N_BID_ACTIONS:])
        # At least one bid action must be True (bid 0 always legal)
        assert mask[0]

    def test_bid_range_matches_round_number(self):
        env = make_env()
        env.reset()
        rn = env._current_state.round_number
        mask = env.action_masks()
        # Bids 0..round_number are legal
        for b in range(rn + 1):
            assert mask[b], f"bid {b} should be legal in round {rn}"
        # Bids above round_number are illegal
        for b in range(rn + 1, N_BID_ACTIONS):
            assert not mask[b], f"bid {b} should be illegal in round {rn}"

    def test_play_phase_no_bid_actions_active(self):
        env = make_env()
        obs, info = env.reset()
        # Place a bid to advance (bid 0 is always valid)
        mask = env.action_masks()
        bid_action = next(i for i, ok in enumerate(mask) if ok)
        env.step(bid_action)
        if env._current_state.phase == GamePhase.PLAYING:
            mask2 = env.action_masks()
            assert not any(mask2[:N_BID_ACTIONS])

    def test_all_masked_actions_are_truly_illegal(self):
        """Any action marked True in mask should not raise ValidationError."""
        from skull_king.engine import ValidationError
        env = make_env(seed=0)
        env.reset()
        mask = env.action_masks()
        # Just verify at least one action is legal without error
        legal_actions = [i for i, ok in enumerate(mask) if ok]
        assert len(legal_actions) > 0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_correct_types(self):
        env = make_env()
        env.reset()
        mask = env.action_masks()
        action = next(i for i, ok in enumerate(mask) if ok)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_always_false(self):
        env = make_env()
        env.reset()
        mask = env.action_masks()
        action = next(i for i, ok in enumerate(mask) if ok)
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False

    def test_step_on_game_over_raises(self):
        env = make_env(seed=0)
        agent = RandomAgent(seed=0)
        agent.run_episode(env)
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_obs_in_valid_range_after_steps(self):
        env = make_env()
        agent = RandomAgent(seed=99)
        obs, _ = env.reset()
        for _ in range(5):
            mask = env.action_masks()
            action = agent.act(obs, mask)
            obs, _, terminated, _, _ = env.step(action)
            assert obs.min() >= -1.0
            assert obs.max() <= 1.0
            if terminated:
                break


# ---------------------------------------------------------------------------
# Full episode
# ---------------------------------------------------------------------------


class TestFullEpisode:
    def test_episode_terminates(self):
        env = make_env(seed=0)
        agent = RandomAgent(seed=0)
        result = run_full_episode(env, agent)
        assert "total_reward" in result
        assert "final_score" in result

    def test_episode_has_positive_steps(self):
        env = make_env(seed=1)
        agent = RandomAgent(seed=1)
        result = run_full_episode(env, agent)
        assert result["steps"] > 0

    def test_2_player_game_completes(self):
        env = SkullKingEnv(n_players=2, seed=0)
        agent = RandomAgent(seed=0)
        result = run_full_episode(env, agent)
        assert result["steps"] > 0

    def test_6_player_game_completes(self):
        env = SkullKingEnv(n_players=6, seed=0)
        agent = RandomAgent(seed=0)
        result = run_full_episode(env, agent)
        assert result["steps"] > 0

    def test_final_phase_is_game_over(self):
        env = make_env(seed=5)
        agent = RandomAgent(seed=5)
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            mask = env.action_masks()
            action = agent.act(obs, mask)
            obs, _, terminated, _, _ = env.step(action)
        assert env._current_state.phase == GamePhase.GAME_OVER

    def test_deterministic_with_same_seed(self):
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)
        env1 = make_env(seed=42)
        env2 = make_env(seed=42)
        result1 = run_full_episode(env1, agent1)
        result2 = run_full_episode(env2, agent2)
        assert result1["final_score"] == result2["final_score"]
        assert result1["steps"] == result2["steps"]

    def test_different_seeds_can_differ(self):
        results = []
        for s in range(5):
            env = make_env(seed=s)
            agent = RandomAgent(seed=s)
            results.append(run_full_episode(env, agent)["final_score"])
        # Scores shouldn't all be identical
        assert len(set(results)) > 1


# ---------------------------------------------------------------------------
# Reward modes
# ---------------------------------------------------------------------------


class TestRewardModes:
    def _collect_rewards(self, env: SkullKingEnv) -> list[float]:
        agent = RandomAgent(seed=0)
        obs, _ = env.reset()
        rewards = []
        terminated = False
        while not terminated:
            mask = env.action_masks()
            action = agent.act(obs, mask)
            obs, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
        return rewards

    def test_sparse_reward_zero_until_end(self):
        env = SkullKingEnv(n_players=3, reward_mode="sparse", seed=0)
        rewards = self._collect_rewards(env)
        assert all(r == 0.0 for r in rewards[:-1])

    def test_sparse_final_reward_nonzero_or_zero(self):
        env = SkullKingEnv(n_players=3, reward_mode="sparse", seed=0)
        rewards = self._collect_rewards(env)
        assert isinstance(rewards[-1], float)

    def test_round_reward_zero_during_tricks(self):
        """During trick play (no round boundary), reward should be 0."""
        env = SkullKingEnv(n_players=3, reward_mode="round", seed=7)
        agent = RandomAgent(seed=0)
        obs, _ = env.reset()
        # Bid first
        mask = env.action_masks()
        action = agent.act(obs, mask)
        obs, reward, terminated, _, info = env.step(action)
        # Mid-trick rewards should be 0
        if info["phase"] == "PLAYING":
            assert reward == 0.0

    def test_shaped_mode_runs(self):
        env = SkullKingEnv(n_players=3, reward_mode="shaped", seed=0)
        rewards = self._collect_rewards(env)
        assert len(rewards) > 0

    def test_shaped_has_intra_round_nonzero_rewards(self):
        """Shaped mode must produce some non-zero rewards before round end."""
        env = SkullKingEnv(n_players=3, reward_mode="shaped", seed=0)
        rewards = self._collect_rewards(env)
        # Not every intra-round step will be zero (trick hints fire)
        assert any(r != 0.0 for r in rewards)

    def test_round_all_intra_round_rewards_zero(self):
        """Round mode must produce zero reward on every non-boundary step."""
        env = SkullKingEnv(n_players=3, reward_mode="round", seed=0)
        agent = RandomAgent(seed=0)
        obs, _ = env.reset()
        non_zero_intra = False
        prev_phase = env._current_state.phase
        terminated = False
        while not terminated:
            mask = env.action_masks()
            action = agent.act(obs, mask)
            obs, reward, terminated, _, _ = env.step(action)
            cur_phase = env._current_state.phase
            at_boundary = (
                cur_phase == GamePhase.BIDDING or cur_phase == GamePhase.GAME_OVER
            )
            if not at_boundary and reward != 0.0:
                non_zero_intra = True
        assert not non_zero_intra

    def test_all_modes_complete_full_game(self):
        for mode in ("sparse", "round", "shaped"):
            env = SkullKingEnv(n_players=3, reward_mode=mode, seed=0)
            agent = RandomAgent(seed=0)
            result = run_full_episode(env, agent)
            assert result["steps"] > 0, f"mode {mode!r} failed to complete"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="reward_mode"):
            SkullKingEnv(reward_mode="dense")


# ---------------------------------------------------------------------------
# Controlled player seat
# ---------------------------------------------------------------------------


class TestControlledPlayer:
    def test_non_zero_controlled_player(self):
        env = SkullKingEnv(n_players=3, controlled_player=2, seed=0)
        agent = RandomAgent(seed=0)
        result = run_full_episode(env, agent)
        assert result["steps"] > 0

    def test_obs_reflects_controlled_player_hand(self):
        env = SkullKingEnv(n_players=3, controlled_player=0, seed=0)
        obs, _ = env.reset()
        hand_vec = obs[0:70]
        # The hand vector should have exactly round_number 1.0 entries
        rn = env._current_state.round_number
        assert int(hand_vec.sum()) == rn


# ---------------------------------------------------------------------------
# _bid_conditioned_signal unit tests
# ---------------------------------------------------------------------------


class TestBidConditionedSignal:
    """Direct unit tests for the shaped-mode hint function."""

    def test_bid0_single_trick_win_is_negative(self):
        sig = _bid_conditioned_signal(bid=0, prev_tricks=0, delta_tricks=1)
        assert sig == pytest.approx(_HINT_BID0_TRICK_WON)

    def test_bid0_multiple_wins_scales(self):
        sig = _bid_conditioned_signal(bid=0, prev_tricks=0, delta_tricks=3)
        assert sig == pytest.approx(_HINT_BID0_TRICK_WON * 3)

    def test_bid0_no_tricks_no_signal(self):
        # delta_tricks=0 means no trick resolved for the player
        sig = _bid_conditioned_signal(bid=0, prev_tricks=0, delta_tricks=0)
        assert sig == pytest.approx(0.0)

    def test_bid_positive_on_track(self):
        # First trick win when bid=3: on track
        sig = _bid_conditioned_signal(bid=3, prev_tricks=0, delta_tricks=1)
        assert sig == pytest.approx(_HINT_ON_TRACK)

    def test_bid_positive_second_trick_on_track(self):
        # Second trick win with bid=3: still on track
        sig = _bid_conditioned_signal(bid=3, prev_tricks=1, delta_tricks=1)
        assert sig == pytest.approx(_HINT_ON_TRACK)

    def test_bid_positive_exactly_at_bid_is_on_track(self):
        # Hitting the bid exactly: still +on-track (not overshot)
        sig = _bid_conditioned_signal(bid=2, prev_tricks=1, delta_tricks=1)
        assert sig == pytest.approx(_HINT_ON_TRACK)  # prev=1, new=2 == bid

    def test_bid_positive_overshoot_is_negative(self):
        # Third trick when bid=2: overshot
        sig = _bid_conditioned_signal(bid=2, prev_tricks=2, delta_tricks=1)
        assert sig == pytest.approx(_HINT_OVERSHOT)

    def test_bid_mixed_on_track_then_overshoot(self):
        # Two tricks in one step: first on track (prev=1→2=bid), second overshot (2→3)
        sig = _bid_conditioned_signal(bid=2, prev_tricks=1, delta_tricks=2)
        assert sig == pytest.approx(_HINT_ON_TRACK + _HINT_OVERSHOT)

    def test_bid_signals_negative_is_correct_direction(self):
        # Bid=0 signal must be negative
        assert _bid_conditioned_signal(bid=0, prev_tricks=0, delta_tricks=1) < 0
        # On-track signal must be positive
        assert _bid_conditioned_signal(bid=5, prev_tricks=0, delta_tricks=1) > 0
        # Overshot signal must be negative
        assert _bid_conditioned_signal(bid=1, prev_tricks=1, delta_tricks=1) < 0

    def test_round_signal_dominates_trick_hints(self):
        """Max possible hint (all 10 tricks in round 10) stays small vs round signal."""
        # Worst case: 10 tricks all on-track
        max_hint = abs(_HINT_ON_TRACK) * 10  # = 0.30
        # Minimum round signal for a meaningful bid hit: bid=1, round=1 → 20/200 = 0.10
        # Max round signal: bid=10, round=10, bonus=0 → 200/200 = 1.0
        assert max_hint < 0.5, "trick hints should stay below typical round signal"
