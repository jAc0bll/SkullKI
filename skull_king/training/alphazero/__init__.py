"""AlphaZero-style training for Skull King.

References:
  - Silver et al. 2017, "Mastering the game of Go without human knowledge"
  - Silver et al. 2018, "A general reinforcement learning algorithm that
    masters chess, shogi, and Go through self-play"

Algorithm:
  - Single shared network with policy head + value head.
  - Self-play with batched MCTS produces (state, π_target, z_target) data:
      π_target  = root-action visit-count distribution from the search
      z_target  = terminal (or n-step) seat-0 outcome, scaled to [-1, +1]
  - Network trained with: cross-entropy(π, π_target) + MSE(V, z) + L2 reg.

Skull-King-specific adaptations:
  - Imperfect information handled by "implicit opponents": between two of
    the agent's decisions, opponent moves are sampled from the same shared
    policy network (treating opponents as rational players using the same
    policy). This is essentially Single-Determinization MCTS without an
    explicit sampling step.
  - Reward shaped per round (return-to-go), scaled to roughly [-1, +1].

Multi-GPU:
  Launch with `torchrun --nproc_per_node=4` for 4× GPU training. Each rank
  runs its own self-play + replay buffer; gradients are synced via
  DistributedDataParallel. Rank 0 evaluates and checkpoints.
"""
