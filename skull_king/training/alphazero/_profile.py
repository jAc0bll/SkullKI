"""Detailed profile of one AlphaZero collect+train iter.

Run on the server with:
    cd /workspace/SkullKI && python -m skull_king.training.alphazero._profile
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
import torch

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.training.alphazero.buffers import AZReplayBuffer
from skull_king.training.alphazero.networks import AlphaZeroNet
from skull_king.training.alphazero.runner import AlphaZeroRunner
from skull_king.training.rebel.public_belief_state import pbs_encoding_size


def main() -> None:
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    net = AlphaZeroNet(4, hidden=(1024, 1024, 512), value_hidden=128).to(device)
    net.eval()
    pbs_size = pbs_encoding_size(4)

    for n_envs, n_sims in [(128, 25), (256, 25), (512, 25), (512, 15), (1024, 15)]:
        runner = AlphaZeroRunner(n_envs=n_envs, n_players=4, device=device, seed=42)
        buf = AZReplayBuffer(capacity=100000, pbs_size=pbs_size, action_size=ACTION_SPACE_SIZE)

        # Warmup — finish first iter through compilation overhead
        runner.collect(net, buf, n_simulations=n_sims, c_puct=2.0,
                       dirichlet_alpha=0.3, dirichlet_eps=0.25,
                       temperature=1.0, max_decisions=200)
        torch.cuda.synchronize()

        # Timed run
        t0 = time.time()
        n = runner.collect(net, buf, n_simulations=n_sims, c_puct=2.0,
                           dirichlet_alpha=0.3, dirichlet_eps=0.25,
                           temperature=1.0, max_decisions=2048)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"n_envs={n_envs:4d}  n_sims={n_sims:2d}  -> collected {n} dec in {elapsed:.1f}s "
              f"({n/elapsed:.0f} dec/s)")

    # Now a detailed cProfile of one configuration
    print("\n=== cProfile of n_envs=512, n_sims=15, 2048 decisions ===")
    runner = AlphaZeroRunner(n_envs=512, n_players=4, device=device, seed=42)
    buf = AZReplayBuffer(capacity=100000, pbs_size=pbs_size, action_size=ACTION_SPACE_SIZE)
    runner.collect(net, buf, n_simulations=15, c_puct=2.0,
                   dirichlet_alpha=0.3, dirichlet_eps=0.25,
                   temperature=1.0, max_decisions=200)  # warmup
    torch.cuda.synchronize()

    profiler = cProfile.Profile()
    profiler.enable()
    runner.collect(net, buf, n_simulations=15, c_puct=2.0,
                   dirichlet_alpha=0.3, dirichlet_eps=0.25,
                   temperature=1.0, max_decisions=2048)
    torch.cuda.synchronize()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
