"""AlphaZero training entry point.

Single GPU:
    python -m skull_king.training.alphazero.train --config configs/alphazero/alphazero_v1.yaml

Multi-GPU (4× RTX 4090):
    torchrun --nproc_per_node=4 --master_port=29500 \\
        -m skull_king.training.alphazero.train --config configs/alphazero/alphazero_v1.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class AlphaZeroConfig:
    # ── Game ────────────────────────────────────────────────────────
    n_players: int = 4
    seed: int = 42

    # ── Training loop (per rank) ───────────────────────────────────
    n_iterations: int = 3000
    collect_decisions: int = 2048    # seat-0 decisions per iter per rank
    n_envs: int = 128                # parallel self-play games per rank

    # ── MCTS ────────────────────────────────────────────────────────
    n_simulations: int = 80          # MCTS simulations per move (training)
    eval_simulations: int = 100      # MCTS simulations per move (eval)
    c_puct: float = 2.0              # PUCT exploration constant
    dirichlet_alpha: float = 0.3     # root prior noise (AlphaZero default)
    dirichlet_eps: float = 0.25
    temperature_initial: float = 1.0  # τ for the first temperature_drop_iter iters
    temperature_final: float = 0.1    # τ afterwards (near-greedy)
    temperature_drop_iter: int = 500

    # ── Network ─────────────────────────────────────────────────────
    hidden: list = field(default_factory=lambda: [1024, 1024, 512])
    value_hidden: int = 128

    # ── Optimization ────────────────────────────────────────────────
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 4096
    train_steps: int = 80
    buffer_capacity: int = 1_000_000
    value_loss_weight: float = 1.0
    grad_clip: float = 1.0

    # ── Logging / persistence ───────────────────────────────────────
    eval_every: int = 25            # fast policy-only eval cadence
    mcts_eval_every: int = 200      # slow MCTS-augmented eval cadence
    eval_games_fast: int = 100      # n games for policy-only eval (cheap)
    eval_games_mcts: int = 30       # n games for MCTS-augmented eval (expensive)
    checkpoint_every: int = 100
    model_dir: str = "models/alphazero"
    run_name: str = "alphazero_v1"

    # ── Resume ──────────────────────────────────────────────────────
    resume_from: str = ""
    start_iter: int = 1


def _load_config(path: str) -> AlphaZeroConfig:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    cfg = AlphaZeroConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_config(args.config)
    from skull_king.training.alphazero.trainer import AlphaZeroTrainer
    AlphaZeroTrainer(cfg).train()


if __name__ == "__main__":
    main()
