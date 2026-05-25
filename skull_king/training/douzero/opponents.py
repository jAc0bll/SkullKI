"""Opponent pool for curriculum self-play training.

Each game has four seats. Seat 0 is the learning agent (the live Q-net,
the only seat whose transitions get pushed to the replay buffer). Seats
1-3 are sampled per-game from a mixed distribution:

  - SELF      : the same live Q-net (acts ε-greedy, no buffer push)
  - LEAGUE    : one of K frozen past Q-net snapshots
  - HEURISTIC : the rule-based HeuristicAgent
  - RANDOM    : uniformly random legal action

The mixture probabilities follow a curriculum: heavy heuristic + random
early to bootstrap useful state coverage, more self/league later for
strategic refinement. Decay is linear in the training-iteration fraction.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch


# Opponent-type integer codes (kept dense for fast numpy grouping)
OPP_SELF      = 0
OPP_LEAGUE_0  = 1   # +k for league index k
OPP_HEURISTIC = 200
OPP_RANDOM    = 201


@dataclass
class CurriculumSchedule:
    """Linear schedule for opponent-mix probabilities."""
    self_start: float = 0.30
    self_end:   float = 0.70
    league_start: float = 0.10
    league_end:   float = 0.25
    heuristic_start: float = 0.40
    heuristic_end:   float = 0.05
    random_start: float = 0.20
    random_end:   float = 0.00

    def probs(self, t: int, total_t: int) -> tuple[float, float, float, float]:
        """Return (p_self, p_league, p_heuristic, p_random) at iteration t.

        Probabilities are renormalized to sum to 1.0 in case of rounding.
        """
        frac = min(1.0, max(0.0, t / max(1, total_t)))
        p_self = self.self_start + (self.self_end - self.self_start) * frac
        p_leag = self.league_start + (self.league_end - self.league_start) * frac
        p_heur = self.heuristic_start + (self.heuristic_end - self.heuristic_start) * frac
        p_rand = self.random_start + (self.random_end - self.random_start) * frac
        s = p_self + p_leag + p_heur + p_rand
        return p_self / s, p_leag / s, p_heur / s, p_rand / s


class LeaguePool:
    """Fixed-capacity FIFO pool of frozen Q-network snapshots on GPU.

    Snapshots are kept as separate nn.Module instances in eval mode. They
    are kept on the same device as the live network for fast inference.
    """

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._nets: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._nets)

    def snapshot(self, live_net: torch.nn.Module) -> None:
        """Add a deep copy of the live Q-net to the pool, eval mode."""
        if isinstance(live_net, torch.nn.parallel.DistributedDataParallel):
            live_net = live_net.module
        # Snapshot the unwrapped module to avoid DDP wrapping in stored copies
        snap = deepcopy(live_net)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self._nets.append(snap)

    def get(self, idx: int) -> torch.nn.Module:
        return self._nets[idx]


def assign_opponents(
    n_envs: int,
    n_players: int,
    schedule: CurriculumSchedule,
    league_size: int,
    t: int,
    total_t: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-env opponent assignment.

    Returns
    -------
    opp : int8 array of shape [n_envs, n_players]
        opp[i, 0] is always OPP_SELF (learning seat).
        opp[i, j>0] is one of OPP_{SELF,LEAGUE_0+k,HEURISTIC,RANDOM}.
    """
    p_self, p_leag, p_heur, p_rand = schedule.probs(t, total_t)

    # Without any past snapshots we have no league — push that mass into self.
    if league_size <= 0:
        p_self += p_leag
        p_leag = 0.0

    opp = np.empty((n_envs, n_players), dtype=np.int16)
    opp[:, 0] = OPP_SELF

    # Sample categorically for seats 1..n-1 across all envs at once
    other_seats = (n_players - 1) * n_envs
    cats = rng.choice(
        4,
        size=other_seats,
        p=[p_self, p_leag, p_heur, p_rand],
    )
    # Mat materialize per-cat
    codes = np.empty(other_seats, dtype=np.int16)
    codes[cats == 0] = OPP_SELF
    if league_size > 0:
        league_pick = rng.integers(0, league_size, size=int((cats == 1).sum()))
        codes[cats == 1] = OPP_LEAGUE_0 + league_pick.astype(np.int16)
    codes[cats == 2] = OPP_HEURISTIC
    codes[cats == 3] = OPP_RANDOM

    opp[:, 1:] = codes.reshape(n_envs, n_players - 1)
    return opp
