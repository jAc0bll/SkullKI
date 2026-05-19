"""NFSP training entry point for Skull King.

Usage
-----
    python -m skull_king.training.nfsp_train                         # nfsp_config.yaml
    python -m skull_king.training.nfsp_train --config my_cfg.yaml
    python -m skull_king.training.nfsp_train --from-model models/skull_king/final

Saves at the end:
    {model_dir}/nfsp_br_final.zip   — best-response PPO network
    {model_dir}/nfsp_sl_final.pt    — average-strategy SL network
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

import torch
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE
from skull_king.training.callbacks import (
    CurriculumCallback,
    SelfPlayCallback,
    StepCheckpointCallback,
    TournamentEvalCallback,
)
from skull_king.training.nfsp_callbacks import NFSPCallback
from skull_king.training.nfsp_env import NFSPSelfPlayEnv
from skull_king.training.nfsp_reservoir import ReservoirBuffer
from skull_king.training.nfsp_sl import AverageStrategyNet, SLTrainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NFSPConfig:
    # env
    n_players: int = 4
    controlled_player: int = 0
    reward_mode: str = "shaped"
    env_seed: int = 42

    # curriculum
    curriculum_start_mix: float = 0.80
    curriculum_end_mix: float = 0.08

    # training loop
    total_timesteps: int = 10_000_000
    n_envs: int = 12
    self_play_update_freq: int = 25_000
    eval_freq: int = 100_000
    n_eval_episodes: int = 20
    n_eval_games: int = 40
    checkpoint_freq: int = 500_000

    # output
    log_dir: str = "logs/skull_king"
    model_dir: str = "models/skull_king"
    run_name: str = "nfsp_v1"

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list = field(default_factory=lambda: [512, 512])

    # NFSP-specific
    eta: float = 0.1                    # prob of BR mode for each opponent
    reservoir_capacity: int = 2_000_000
    sl_lr: float = 1e-3
    sl_batch_size: int = 512
    sl_n_updates: int = 8               # SL gradient steps per rollout
    sl_hidden: list = field(default_factory=lambda: [256, 256])
    min_buffer_size: int = 5_000        # reservoir fill before first SL update


def load_config(path: str) -> NFSPConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    cfg = NFSPConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config_path: str = "nfsp_config.yaml", from_model: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    tb_log = os.path.join(cfg.log_dir, cfg.run_name)

    print(f"\n{'='*64}")
    print(f"  Skull King NFSP Training")
    print(f"  run:         {cfg.run_name}")
    print(f"  players:     {cfg.n_players}   reward: {cfg.reward_mode}")
    print(f"  steps:       {cfg.total_timesteps:,}   envs: {cfg.n_envs}")
    print(f"  eta:         {cfg.eta}   reservoir: {cfg.reservoir_capacity:,}")
    print(f"  TensorBoard: tensorboard --logdir {cfg.log_dir}")
    print(f"{'='*64}\n")

    sl_hidden = tuple(cfg.sl_hidden)

    # ── Vectorised training envs ──────────────────────────────────────────
    def _make_env(idx: int):
        def _inner():
            import torch as _torch
            _torch.set_num_threads(1)
            return NFSPSelfPlayEnv(
                eta=cfg.eta,
                sl_hidden=sl_hidden,
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
        print("  Using SubprocVecEnv (parallel)")
    except Exception as exc:
        print(f"  SubprocVecEnv failed ({exc}), using DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = {"net_arch": list(cfg.net_arch)}

    if from_model:
        print(f"  Warm-starting BR from: {from_model}")
        model = MaskablePPO.load(
            from_model,
            env=vec_env,
            tensorboard_log=tb_log,
            verbose=1,
        )
        model.learning_rate = cfg.learning_rate
        model.ent_coef = cfg.ent_coef
    else:
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

    # ── SL network + trainer (main process only) ──────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sl_net = AverageStrategyNet(
        obs_size=OBS_SIZE,
        action_size=ACTION_SPACE_SIZE,
        hidden=sl_hidden,
    ).to(device)
    sl_net.eval()

    reservoir = ReservoirBuffer(
        capacity=cfg.reservoir_capacity,
        obs_size=OBS_SIZE,
        action_size=ACTION_SPACE_SIZE,
        seed=cfg.env_seed,
    )
    sl_trainer = SLTrainer(
        net=sl_net,
        lr=cfg.sl_lr,
        batch_size=cfg.sl_batch_size,
        device=device,
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
        NFSPCallback(
            reservoir=reservoir,
            sl_trainer=sl_trainer,
            sl_n_updates=cfg.sl_n_updates,
            min_buffer_size=cfg.min_buffer_size,
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

    # ── Save ──────────────────────────────────────────────────────────────
    br_path = os.path.join(cfg.model_dir, "nfsp_br_final")
    sl_path = os.path.join(cfg.model_dir, "nfsp_sl_final.pt")
    model.save(br_path)
    torch.save(sl_net.state_dict(), sl_path)
    print(f"\nTraining complete.")
    print(f"  BR model → {br_path}.zip")
    print(f"  SL model → {sl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skull King NFSP agent")
    parser.add_argument("--config", default="nfsp_config.yaml")
    parser.add_argument(
        "--from-model",
        default=None,
        help="Warm-start BR from an existing model zip (e.g. models/skull_king/final)",
    )
    args = parser.parse_args()
    train(args.config, from_model=args.from_model)


if __name__ == "__main__":
    main()
