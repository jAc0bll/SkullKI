"""Training callbacks: self-play opponent refresh, tournament evaluation, logging."""
from __future__ import annotations

import io
import os
import time
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.env.skull_king_env import SkullKingEnv
from skull_king.training.self_play_env import SB3AgentWrapper
from skull_king.tournament.runner import TournamentRunner


# ---------------------------------------------------------------------------
# Self-play: freeze current policy as opponent every N steps
# ---------------------------------------------------------------------------


class SelfPlayCallback(BaseCallback):
    """Periodically replaces the frozen opponent in every training env with a
    deep copy of the current learning policy.

    Until the first update (step 0) the opponents are random agents (the env
    default), giving the agent a warm-start opponent mix.

    Parameters
    ----------
    update_freq:
        Number of *training steps* (``num_timesteps``) between updates.
    verbose:
        0 = silent, 1 = log each update.
    """

    def __init__(self, update_freq: int = 5_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._update_freq = update_freq
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self._update_freq:
            # deepcopy fails in Python 3.14 on mappingproxy inside VecMonitor;
            # use SB3's own serialisation which only saves policy weights.
            buf = io.BytesIO()
            self.model.save(buf)
            buf.seek(0)
            frozen = type(self.model).load(buf)
            self.training_env.env_method("set_opponent", frozen)
            self._last_update = self.num_timesteps
            if self.verbose >= 1:
                print(f"[SelfPlay] opponent updated at step {self.num_timesteps}")
        return True


# ---------------------------------------------------------------------------
# Tournament evaluation with TensorBoard logging
# ---------------------------------------------------------------------------


class TournamentEvalCallback(BaseCallback):
    """Runs periodic tournament evaluation and logs to TensorBoard.

    Logs the following metrics (all prefixed with ``eval/``):
        win_rate_vs_random      — win rate in 4-player game (agent + 3 random)
        win_rate_vs_heuristic   — win rate in 4-player game (agent + 3 heuristic)
        avg_score_vs_random     — mean final score vs random
        avg_score_vs_heuristic  — mean final score vs heuristic
        bid_accuracy            — fraction of rounds where bid was exactly met
        mean_episode_reward     — mean reward from a fresh SkullKingEnv episode run

    Parameters
    ----------
    n_players:
        Number of players in evaluation games (should match training env).
    eval_freq:
        Number of training steps between evaluations.
    n_games:
        Tournament games per opponent type.
    n_eval_episodes:
        Episodes for mean-episode-reward calculation.
    eval_seed:
        Fixed seed for reproducible evaluations.
    """

    def __init__(
        self,
        n_players: int = 4,
        eval_freq: int = 10_000,
        n_games: int = 20,
        n_eval_episodes: int = 10,
        eval_seed: int = 999,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._n_players = n_players
        self._eval_freq = eval_freq
        self._n_games = n_games
        self._n_eval_episodes = n_eval_episodes
        self._eval_seed = eval_seed
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self._eval_freq:
            self._last_eval = self.num_timesteps
            t0 = time.time()
            self._run_eval()
            elapsed = time.time() - t0
            if self.verbose >= 1:
                print(f"[Eval] completed in {elapsed:.1f}s at step {self.num_timesteps}")
        return True

    def _run_eval(self) -> None:
        n = self._n_players
        ppo_agent = SB3AgentWrapper(self.model, n_players=n, name="PPO")
        runner = TournamentRunner(seed=self._eval_seed)

        # ── vs random ────────────────────────────────────────────────────────
        agents_r = [ppo_agent] + [RandomAgent(i) for i in range(n - 1)]
        result_r = runner.run(agents_r, n_games=self._n_games)
        wr_random = result_r.win_rates().get("PPO", 0.0)
        avg_r = result_r.avg_scores().get("PPO", 0.0)

        # ── vs heuristic ─────────────────────────────────────────────────────
        agents_h = [ppo_agent] + [HeuristicAgent() for _ in range(n - 1)]
        result_h = runner.run(agents_h, n_games=self._n_games)
        wr_heuristic = result_h.win_rates().get("PPO", 0.0)
        avg_h = result_h.avg_scores().get("PPO", 0.0)

        # ── episode reward in raw env (masked eval loop) ──────────────────────
        mean_reward = self._eval_mean_reward(n, self._n_eval_episodes)

        # ── bid accuracy via raw env episodes ─────────────────────────────────
        bid_accuracy = self._measure_bid_accuracy(n, n_episodes=self._n_eval_episodes)

        # ── TensorBoard logging ───────────────────────────────────────────────
        self.logger.record("eval/win_rate_vs_random", wr_random)
        self.logger.record("eval/win_rate_vs_heuristic", wr_heuristic)
        self.logger.record("eval/avg_score_vs_random", avg_r)
        self.logger.record("eval/avg_score_vs_heuristic", avg_h)
        self.logger.record("eval/mean_episode_reward", float(mean_reward))
        self.logger.record("eval/bid_accuracy", bid_accuracy)
        self.logger.dump(self.num_timesteps)

    def _eval_mean_reward(self, n_players: int, n_episodes: int) -> float:
        """Mean episode reward using proper action masks (evaluate_policy lacks mask support)."""
        env = SkullKingEnv(n_players=n_players, seed=self._eval_seed)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = self.model.predict(
                    obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True
                )
                obs, reward, terminated, truncated, _ = env.step(int(action[0]))
                ep_reward += float(reward)
                done = terminated or truncated
            rewards.append(ep_reward)
        return float(np.mean(rewards))

    def _measure_bid_accuracy(self, n_players: int, n_episodes: int) -> float:
        """Fraction of rounds where the agent hit their bid.

        Uses per-round score delta as proxy: positive delta means bid was hit
        (a miss always gives -10 × round_number; a hit always gives ≥ +10).
        Direct tricks_won inspection fails because _advance_others resets
        tricks_won_this_round before step() returns at round boundaries.
        """
        env = SkullKingEnv(n_players=n_players, seed=self._eval_seed)

        bid_hits = 0
        bid_total = 0

        for _ in range(n_episodes):
            obs, info = env.reset()
            terminated = False
            prev_score = 0
            prev_round = info["round"]

            while not terminated:
                mask = env.action_masks()
                action, _ = self.model.predict(
                    obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True
                )
                obs, _, terminated, _, info = env.step(int(action[0]))

                cur_round = info["round"]
                cur_score = info["score"]

                if cur_round > prev_round or terminated:
                    if cur_score - prev_score > 0:
                        bid_hits += 1
                    bid_total += 1
                    prev_score = cur_score
                    prev_round = cur_round

        return bid_hits / bid_total if bid_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Curriculum: anneal heuristic_mix from high → low over training
# ---------------------------------------------------------------------------


class CurriculumCallback(BaseCallback):
    """Linearly anneals the heuristic_mix in every training env.

    Phase 1 (early): high heuristic_mix (e.g. 0.8) — agent always faces
    near-optimal bidding and play from opponents, so it cannot exploit
    random-policy mistakes and must learn real strategy.

    Phase 2 (late): low heuristic_mix (e.g. 0.1) — agent refines against
    its own frozen policies, discovering strategies that beat self-play.

    Parameters
    ----------
    total_timesteps:
        Must match ``model.learn(total_timesteps=...)``.
    start_mix:
        heuristic_mix at step 0 (recommended: 0.8).
    end_mix:
        heuristic_mix at final step (recommended: 0.05–0.15).
    update_interval:
        Steps between mix updates (default: 10 000).
    """

    def __init__(
        self,
        total_timesteps: int,
        start_mix: float = 0.8,
        end_mix: float = 0.1,
        update_interval: int = 10_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._total = total_timesteps
        self._start = start_mix
        self._end = end_mix
        self._interval = update_interval
        self._last_update = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self._interval:
            progress = min(self.num_timesteps / self._total, 1.0)
            mix = self._start + (self._end - self._start) * progress
            self.training_env.env_method("set_heuristic_mix", mix)
            self._last_update = self.num_timesteps
            if self.verbose >= 1:
                print(
                    f"[Curriculum] heuristic_mix → {mix:.3f}"
                    f"  ({progress * 100:.0f}% of training)"
                )
        return True


# ---------------------------------------------------------------------------
# Checkpoint on a per-step schedule
# ---------------------------------------------------------------------------


class StepCheckpointCallback(BaseCallback):
    """Save the model every *save_freq* steps.

    Parameters
    ----------
    save_freq:
        Steps between checkpoints.
    save_path:
        Directory where models are saved.
    name_prefix:
        Filename prefix; saved as ``{prefix}_{step}.zip``.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_path = save_path
        self._name_prefix = name_prefix
        self._last_save = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self._save_freq:
            path = os.path.join(
                self._save_path,
                f"{self._name_prefix}_{self.num_timesteps}",
            )
            self.model.save(path)
            self._last_save = self.num_timesteps
            if self.verbose >= 1:
                print(f"[Checkpoint] saved → {path}.zip")
        return True
