"""DouZero-style training for Skull King.

Reference: Zha et al. 2021, "DouZero: Mastering DouDizhu with Self-Play Deep
Reinforcement Learning" (ICML 2021).

Differences from the paper:
  - Skull King is 4-player with symmetric roles → one shared Q-network
    (DouZero used three role-specific Q-nets for Landlord/Peasant).
  - We add a curriculum mixing Heuristic + Random + League opponents into
    self-play to learn exploitative play (Skull King has a strong baseline
    heuristic; pure self-play converges to Nash and underexploits).
  - Q(s) → q-values vector (small action space); the paper used Q(s, a)
    because Dou Dizhu has a combinatorial action space.

Algorithm:
  Actors run vectorized self-play games with mixed opponents.
  Trajectories are stored with terminal MC returns per acting player.
  Q-network trained with MSE: minimize (Q(s, a) - G_t)^2 over the buffer.
  ε-greedy collection policy with annealed ε.
  Periodically snapshot Q-net into a frozen league for opponent diversity.

Multi-GPU:
  Launch with `torchrun --nproc_per_node=N -m skull_king.training.douzero.train`.
  Each rank runs its own VectorizedRunner + replay buffer; gradients are
  synced across ranks via DistributedDataParallel.
"""
