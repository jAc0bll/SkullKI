"""DouZero training entry point.

Single GPU:
    python -m skull_king.training.douzero.train --config configs/douzero/douzero_v1.yaml

Multi-GPU (4 GPUs example):
    torchrun --nproc_per_node=4 --master_port=29500 \\
        -m skull_king.training.douzero.train --config configs/douzero/douzero_v1.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class DouZeroConfig:
    # ── Game ────────────────────────────────────────────────────────
    n_players: int = 4
    seed: int = 42

    # ── Training loop ───────────────────────────────────────────────
    n_iterations: int = 5000
    collect_steps: int = 32768          # decisions per iter (per rank)
    n_envs: int = 2048                  # parallel game envs (per rank)

    # ── Exploration ─────────────────────────────────────────────────
    epsilon_start: float = 0.10
    epsilon_end:   float = 0.01
    epsilon_decay: int   = 2500          # iters to anneal over

    # ── Curriculum (linear schedule across n_iterations) ────────────
    # Defaults: heavy heuristic + random early, more self + league later.
    self_start:      float = 0.30
    self_end:        float = 0.70
    league_start:    float = 0.10
    league_end:      float = 0.25
    heuristic_start: float = 0.40
    heuristic_end:   float = 0.05
    random_start:    float = 0.20
    random_end:      float = 0.00
    league_capacity: int = 8
    league_snapshot_every: int = 200    # iters

    # ── Q-network ───────────────────────────────────────────────────
    hidden: list = field(default_factory=lambda: [1024, 1024, 512, 256])
    compile_nets: bool = True           # torch.compile (CUDA only)

    # ── Optimization ────────────────────────────────────────────────
    lr: float = 5.0e-4
    batch_size: int = 4096
    train_steps: int = 250
    buffer_capacity: int = 2_000_000
    grad_clip: float = 1.0

    # ── Logging / persistence ───────────────────────────────────────
    eval_every: int = 50
    checkpoint_every: int = 250
    model_dir: str = "models/douzero"
    run_name: str = "douzero_v1"

    # ── Resume ──────────────────────────────────────────────────────
    resume_from: str = ""               # base path, no .pt suffix
    start_iter: int = 1


def _load_config(path: str) -> DouZeroConfig:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    cfg = DouZeroConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from skull_king.training.douzero.trainer import DouZeroTrainer
    DouZeroTrainer(cfg).train()


if __name__ == "__main__":
    main()
