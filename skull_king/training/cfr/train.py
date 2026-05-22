"""Deep CFR training entry point.

Usage
-----
    python -m skull_king.training.cfr.train                         # cfr_config.yaml
    python -m skull_king.training.cfr.train --config cfr_config_v4_pc.yaml

Output
------
    models/skull_king/cfr_final_adv.pt    — advantage network (training only)
    models/skull_king/cfr_final_strat.pt  — strategy network  (use for play)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import yaml

from skull_king.training.cfr.trainer import DeepCFRTrainer


@dataclass
class CFRConfig:
    # Game
    n_players: int = 4
    env_seed: int = 42

    # CFR loop
    n_cfr_iterations: int = 500
    traversals_per_player: int = 200   # per player per iteration
    num_workers: int = 8               # parallel workers (set to 1 for debug)

    # Network architecture (shared between adv and strat nets)
    net_hidden: list = field(default_factory=lambda: [512, 512])

    # Advantage network training
    adv_lr: float = 1e-3
    adv_batch_size: int = 1024
    adv_train_steps: int = 500         # gradient steps per iteration (fixed budget)
    adv_buffer_capacity: int = 2_000_000

    # Strategy network training
    strat_lr: float = 1e-3
    strat_batch_size: int = 1024
    strat_train_steps: int = 500       # gradient steps per iteration (fixed budget)
    strat_buffer_capacity: int = 2_000_000

    # Opponent pool: fraction of non-traverser decisions played by HeuristicAgent.
    # 0.0 = pure self-play (old behaviour); 0.4 = 40% heuristic opponents.
    heuristic_frac: float = 0.0

    # Logging / output
    eval_every_n_iters: int = 50
    checkpoint_every_n_iters: int = 100
    model_dir: str = "models/skull_king"
    run_name: str = "cfr_v1"

    # --- backward compat: old epoch-based keys are silently ignored ---
    adv_train_epochs: int = 0
    strat_train_epochs: int = 0


def load_config(path: str) -> CFRConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = CFRConfig()
    known = {f.name for f in __import__("dataclasses").fields(cfg)}
    unknown = [k for k in raw if k not in known]
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {unknown}.  "
            f"Valid keys: {sorted(known)}"
        )
    for k, v in raw.items():
        setattr(cfg, k, v)
    # Backward compat: if old epoch keys used but new step keys not set,
    # convert epochs → rough step count (epochs × ~500 steps/epoch).
    if cfg.adv_train_steps == 500 and cfg.adv_train_epochs > 0:
        cfg.adv_train_steps = cfg.adv_train_epochs * 500
    if cfg.strat_train_steps == 500 and cfg.strat_train_epochs > 0:
        cfg.strat_train_steps = cfg.strat_train_epochs * 500
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skull King Deep CFR agent")
    parser.add_argument("--config", default="cfr_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = DeepCFRTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
