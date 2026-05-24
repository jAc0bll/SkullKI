"""Entry point for ReBeL training.

Usage:
    python -m skull_king.training.rebel.train --config configs/rebel/rebel_v1.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RebelConfig:
    # Game
    n_players: int = 4
    seed: int = 42

    # Training loop
    n_iterations: int = 500
    games_per_iter: int = 20

    # Subgame solver
    n_subgame_samples: int = 16       # K determinizations per decision node
    n_cfr_iters_per_subgame: int = 50  # CFR iters within each determinization

    # Networks
    value_hidden: list = field(default_factory=lambda: [512, 512, 256])
    policy_hidden: list = field(default_factory=lambda: [512, 512, 256])

    # Optimization
    value_lr: float = 1e-3
    policy_lr: float = 1e-3
    batch_size: int = 512
    train_steps: int = 100
    buffer_capacity: int = 200_000

    # Logging / persistence
    eval_every: int = 50
    checkpoint_every: int = 100
    model_dir: str = "models/rebel"
    run_name: str = "rebel_v1"


def _load_config(path: str) -> RebelConfig:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    cfg = RebelConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from skull_king.training.rebel.trainer import RebelTrainer
    trainer = RebelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
