"""NFSP training entry point.

Usage:
    python -m skull_king.training.nfsp.train --config configs/nfsp/nfsp_v1.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class NfspConfig:
    # Game
    n_players: int = 4
    seed: int = 42

    # Collection
    n_iterations: int = 2000
    collect_steps: int = 16_384   # decisions per iter
    n_envs: int = 512             # parallel game environments

    # NFSP
    eta: float = 0.1              # prob of using BR policy (rest uses avg)
    epsilon_start: float = 0.06   # ε-greedy exploration for BR
    epsilon_end: float = 0.001
    epsilon_decay: int = 1000     # iters to anneal over
    epsilon: float = 0.06         # current value, updated during training

    # Networks
    hidden: list = field(default_factory=lambda: [512, 512, 256])
    compile_nets: bool = True     # torch.compile both nets (CUDA only)
    eval_br: bool = True          # also evaluate the BR (Q-net) policy each eval

    # Q-net (RL / best response)
    rl_lr: float = 1e-3
    rl_batch_size: int = 4096
    rl_train_steps: int = 300
    rl_buffer_capacity: int = 1_000_000

    # Avg-net (SL / average strategy)
    sl_lr: float = 1e-3
    sl_batch_size: int = 4096
    sl_train_steps: int = 300
    sl_buffer_capacity: int = 2_000_000

    # Logging / persistence
    eval_every: int = 50
    checkpoint_every: int = 200
    model_dir: str = "models/nfsp"
    run_name: str = "nfsp_v1"

    resume_from: str = ""
    start_iter: int = 1


def _load_config(path: str) -> NfspConfig:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    cfg = NfspConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from skull_king.training.nfsp.trainer import NfspTrainer
    NfspTrainer(cfg).train()


if __name__ == "__main__":
    main()
