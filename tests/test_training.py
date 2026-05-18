"""Smoke tests for the training infrastructure.

All tests use tiny step counts so they run in a few seconds.
They verify correctness of wiring, not convergence.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np
import pytest

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.env.skull_king_env import SkullKingEnv
from skull_king.game_state import GamePhase
from skull_king.training.self_play_env import SB3AgentWrapper, SelfPlaySkullKingEnv
from skull_king.training.train import TrainingConfig, load_config


# ---------------------------------------------------------------------------
# SelfPlaySkullKingEnv
# ---------------------------------------------------------------------------


class TestSelfPlayEnv:
    def test_is_skull_king_env(self):
        env = SelfPlaySkullKingEnv(n_players=3)
        assert isinstance(env, SkullKingEnv)

    def test_reset_and_step_without_opponent(self):
        env = SelfPlaySkullKingEnv(n_players=3, seed=0)
        obs, _ = env.reset()
        mask = env.action_masks()
        action = int(np.argmax(mask))  # first legal action
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == obs.shape

    def test_set_opponent_and_run(self):
        """Setting a frozen model as opponent should not crash."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO

        env = SelfPlaySkullKingEnv(n_players=3, seed=0)
        vec = DummyVecEnv([lambda: SelfPlaySkullKingEnv(n_players=3, seed=0)])
        model = MaskablePPO("MlpPolicy", vec, verbose=0)

        env.set_opponent(model)
        obs, _ = env.reset()

        terminated = False
        steps = 0
        while not terminated and steps < 30:
            mask = env.action_masks()
            action, _ = model.predict(obs[np.newaxis], action_masks=mask[np.newaxis])
            obs, _, terminated, _, _ = env.step(int(action[0]))
            steps += 1

        assert steps > 0

    def test_full_episode_with_self_play(self):
        """Full episode with self-play opponent must reach GAME_OVER."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO

        vec = DummyVecEnv([lambda: SelfPlaySkullKingEnv(n_players=3, seed=5)])
        model = MaskablePPO("MlpPolicy", vec, verbose=0)

        env = SelfPlaySkullKingEnv(n_players=3, seed=5)
        env.set_opponent(model)
        obs, _ = env.reset()

        terminated = False
        while not terminated:
            mask = env.action_masks()
            action, _ = model.predict(obs[np.newaxis], action_masks=mask[np.newaxis])
            obs, _, terminated, _, _ = env.step(int(action[0]))

        assert env._current_state.phase == GamePhase.GAME_OVER

    def test_opponent_fallback_is_random(self):
        """Without set_opponent(), env auto-plays others randomly."""
        env = SelfPlaySkullKingEnv(n_players=3, seed=0)
        obs, _ = env.reset()
        assert env._opponent is None
        mask = env.action_masks()
        action = next(i for i, ok in enumerate(mask) if ok)
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == obs.shape


# ---------------------------------------------------------------------------
# SB3AgentWrapper
# ---------------------------------------------------------------------------


class TestSB3AgentWrapper:
    def _make_model_and_wrapper(self, n_players=3):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO
        from skull_king.engine import GameEngine

        vec = DummyVecEnv([lambda: SkullKingEnv(n_players=n_players, seed=0)])
        model = MaskablePPO("MlpPolicy", vec, verbose=0)
        wrapper = SB3AgentWrapper(model, n_players=n_players)
        engine = GameEngine(n_players=n_players, seed=0)
        engine.start()
        wrapper.before_move(engine)
        return wrapper, engine

    def test_bid_in_range(self):
        from skull_king.engine import GameEngine
        wrapper, engine = self._make_model_and_wrapper(3)
        state = engine.get_state()
        b = wrapper.bid(state, 0)
        assert 0 <= b <= state.round_number

    def test_play_returns_legal_card(self):
        from skull_king.engine import GameEngine
        from skull_king.resolver import TrickResolver
        wrapper, engine = self._make_model_and_wrapper(3)
        state = engine.get_state()
        # Bid all players first
        for i in range(3):
            engine.place_bid(i, 1)
        state = engine.get_state()
        wrapper.before_move(engine)
        pi = state.current_player_index
        card, mode = wrapper.play(state, pi)
        legal = TrickResolver.legal_plays(
            list(state.current_trick_cards), list(state.player_states[pi].hand)
        )
        assert card in legal

    def test_wrapper_plays_full_tournament_game(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO
        from skull_king.tournament.runner import TournamentRunner

        vec = DummyVecEnv([lambda: SkullKingEnv(n_players=3, seed=0)])
        model = MaskablePPO("MlpPolicy", vec, verbose=0)
        wrapper = SB3AgentWrapper(model, n_players=3)

        agents = [wrapper, RandomAgent(0), RandomAgent(1)]
        result = TournamentRunner(seed=0).run(agents, n_games=2)
        assert result.n_games == 2


# ---------------------------------------------------------------------------
# TrainingConfig / YAML loading
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.n_players == 4
        assert cfg.total_timesteps == 100_000
        assert cfg.reward_mode == "round"
        assert cfg.net_arch == [256, 256]

    def test_load_yaml(self, tmp_path):
        yml = textwrap.dedent("""\
            n_players: 3
            total_timesteps: 5_000
            n_envs: 2
            learning_rate: 1.0e-4
            net_arch: [128, 128]
        """)
        p = tmp_path / "cfg.yaml"
        p.write_text(yml)
        cfg = load_config(str(p))
        assert cfg.n_players == 3
        assert cfg.total_timesteps == 5_000
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.net_arch == [128, 128]

    def test_load_real_config(self):
        cfg = load_config("training_config.yaml")
        assert cfg.n_players >= 2
        assert cfg.total_timesteps > 0
        assert cfg.reward_mode in ("sparse", "round", "shaped")


# ---------------------------------------------------------------------------
# End-to-end: tiny 2 000-step training run
# ---------------------------------------------------------------------------


class TestMinimalTraining:
    def test_train_2000_steps(self, tmp_path):
        """Full pipeline smoke test: 2 000 steps with evaluation."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from sb3_contrib import MaskablePPO
        from skull_king.training.callbacks import (
            SelfPlayCallback, StepCheckpointCallback, TournamentEvalCallback,
        )
        from stable_baselines3.common.callbacks import CallbackList

        n_players = 3
        vec_env = VecMonitor(
            DummyVecEnv([
                lambda: SelfPlaySkullKingEnv(n_players=n_players, seed=i)
                for i in range(2)
            ])
        )
        model = MaskablePPO(
            "MlpPolicy", vec_env, verbose=0,
            n_steps=64, batch_size=32, n_epochs=2,
            policy_kwargs={"net_arch": [64, 64]},
        )
        callbacks = CallbackList([
            SelfPlayCallback(update_freq=500, verbose=0),
            TournamentEvalCallback(
                n_players=n_players,
                eval_freq=1_000,
                n_games=3,
                n_eval_episodes=2,
                verbose=0,
            ),
            StepCheckpointCallback(
                save_freq=1_000,
                save_path=str(tmp_path),
                name_prefix="test_model",
                verbose=0,
            ),
        ])

        model.learn(total_timesteps=2_000, callback=callbacks)

        # Checkpoint was saved
        files = list(tmp_path.glob("*.zip"))
        assert len(files) >= 1

    def test_model_predict_legal_action(self):
        """Trained (untrained) model always outputs a valid action given a mask."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO

        env = SkullKingEnv(n_players=3, seed=0)
        vec = DummyVecEnv([lambda: SkullKingEnv(n_players=3, seed=0)])
        model = MaskablePPO("MlpPolicy", vec, verbose=0)

        obs, _ = env.reset()
        for _ in range(10):
            mask = env.action_masks()
            action, _ = model.predict(obs[np.newaxis], action_masks=mask[np.newaxis])
            assert mask[int(action[0])], f"Illegal action {action[0]} chosen"
            obs, _, terminated, _, _ = env.step(int(action[0]))
            if terminated:
                obs, _ = env.reset()
