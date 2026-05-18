"""Skull King RL training entry point.

Usage
-----
    python -m skull_king.training.train                        # uses training_config.yaml
    python -m skull_king.training.train --config my_cfg.yaml  # custom config

The script:
  1. Loads a YAML config (see ``training_config.yaml`` in the project root).
  2. Creates N vectorised ``SelfPlaySkullKingEnv`` instances.
  3. Trains ``MaskablePPO`` with self-play, evaluation, and checkpointing.
  4. Saves the final model to ``{model_dir}/final.zip``.
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

# Suppress harmless Python 3.14 RuntimeWarning when SubprocVecEnv spawns subprocesses
warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

import numpy as np
import yaml
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from sb3_contrib import MaskablePPO

from skull_king.training.callbacks import (
    CurriculumCallback,
    SelfPlayCallback,
    StepCheckpointCallback,
    TournamentEvalCallback,
)
from skull_king.training.self_play_env import SelfPlaySkullKingEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    # env
    n_players: int = 4
    controlled_player: int = 0
    reward_mode: str = "round"
    env_seed: int = 42
    heuristic_mix: float = 0.3

    # training loop
    total_timesteps: int = 100_000
    n_envs: int = 4
    self_play_update_freq: int = 5_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    n_eval_games: int = 20
    checkpoint_freq: int = 25_000

    # curriculum
    curriculum_start_mix: float = 0.8   # initial heuristic_mix (overrides heuristic_mix)
    curriculum_end_mix: float = 0.10    # final heuristic_mix after full training

    # output
    log_dir: str = "logs/skull_king"
    model_dir: str = "models/skull_king"
    run_name: str = "ppo_selfplay"

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # network
    net_arch: list = field(default_factory=lambda: [256, 256])


def load_config(path: str) -> TrainingConfig:
    """Load a YAML config file and return a ``TrainingConfig``."""
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    cfg = TrainingConfig()

    def _apply(section_dict: dict[str, Any], prefix: str = "") -> None:
        for k, v in section_dict.items():
            attr = f"{prefix}{k}" if prefix else k
            if hasattr(cfg, attr):
                setattr(cfg, attr, v)

    for section, values in raw.items():
        if isinstance(values, dict):
            _apply(values)
        else:
            if hasattr(cfg, section):
                setattr(cfg, section, values)

    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config_path: str = "training_config.yaml") -> MaskablePPO:
    """Run a full training experiment; return the final model."""
    cfg = load_config(config_path)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    tb_log = os.path.join(cfg.log_dir, cfg.run_name)
    print(f"\n{'='*60}")
    print(f" Skull King PPO Training")
    print(f" run:        {cfg.run_name}")
    print(f" players:    {cfg.n_players}   reward: {cfg.reward_mode}")
    print(f" steps:      {cfg.total_timesteps:,}   envs: {cfg.n_envs}")
    print(f" TensorBoard: tensorboard --logdir {cfg.log_dir}")
    print(f"{'='*60}\n")

    # ── Vectorised training envs ──────────────────────────────────────────
    # SubprocVecEnv runs each env in its own OS process — true parallel game
    # simulation across CPU cores.  Falls back to DummyVecEnv (sequential)
    # if subprocess spawning fails (e.g. some Windows + Python version combos).
    def _make_env(idx: int):
        def _inner():
            # Each env runs in its own subprocess. PyTorch defaults to using
            # multiple threads per process (via OpenMP/MKL). With 34+ subprocesses
            # this causes severe thread oversubscription — set to 1 thread so each
            # process uses exactly one core for opponent neural net inference.
            import torch
            torch.set_num_threads(1)
            return SelfPlaySkullKingEnv(
                n_players=cfg.n_players,
                controlled_player=cfg.controlled_player,
                reward_mode=cfg.reward_mode,
                seed=cfg.env_seed + idx,
                heuristic_mix=cfg.curriculum_start_mix,
            )
        return _inner

    env_fns = [_make_env(i) for i in range(cfg.n_envs)]
    try:
        vec_env = SubprocVecEnv(env_fns)
        print(" Using SubprocVecEnv (parallel, one process per env)")
    except Exception as exc:
        print(f" SubprocVecEnv failed ({exc}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # ── Model ─────────────────────────────────────────────────────────────
    policy_kwargs = {"net_arch": cfg.net_arch}

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tb_log,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=cfg.env_seed,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = CallbackList([
        CurriculumCallback(
            total_timesteps=cfg.total_timesteps,
            start_mix=cfg.curriculum_start_mix,
            end_mix=cfg.curriculum_end_mix,
            verbose=1,
        ),
        SelfPlayCallback(
            update_freq=cfg.self_play_update_freq,
            verbose=1,
        ),
        TournamentEvalCallback(
            n_players=cfg.n_players,
            eval_freq=cfg.eval_freq,
            n_games=cfg.n_eval_games,
            n_eval_episodes=cfg.n_eval_episodes,
            verbose=1,
        ),
        StepCheckpointCallback(
            save_freq=cfg.checkpoint_freq,
            save_path=cfg.model_dir,
            name_prefix=cfg.run_name,
            verbose=1,
        ),
    ])

    # ── Learn ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(cfg.model_dir, "final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved → {final_path}.zip")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skull King PPO agent")
    parser.add_argument(
        "--config",
        default="training_config.yaml",
        help="Path to YAML config file (default: training_config.yaml)",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
