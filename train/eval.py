"""Python-side tournament for evaluating Python agents (e.g. NNAgent) against
the C++ baselines. Seat-rotating like the C++ sk_tournament but supports
mixing C++ agents and Python agents in the same lineup.

Usage:
    python train/eval.py --games 80 --agents nn,heuristic,heuristic,heuristic \
                         --ckpt train/checkpoints/bc_small.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
sys.path.insert(0, str(REPO_ROOT / "train"))

import skullking as sk  # noqa: E402
from nn_agent import NNAgent  # noqa: E402


def make_agent(name: str, ckpt: str | None, ismcts_sims: int):
    if name == "random":     return sk.RandomAgent()
    if name == "heuristic":  return sk.HeuristicAgent()
    if name == "ismcts":
        cfg = sk.ISMCTSConfig()
        cfg.num_simulations = ismcts_sims
        return sk.ISMCTSAgent(cfg)
    if name == "nn":
        if not ckpt:
            raise SystemExit("--ckpt PATH required when 'nn' is in --agents")
        return NNAgent(ckpt_path=ckpt)
    raise SystemExit(f"Unknown agent: {name}")


def play_one(seating, agent_pool, rng):
    state = sk.initial_state(start_player=0)
    sk.deal_round(state, rng)
    while not sk.is_terminal(state):
        agent = agent_pool[seating[state.current_player]]
        a = agent.select_action(state, rng)
        sk.step(state, a, rng)
    return state.scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",   type=int, default=80)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--agents",  type=str, default="nn,heuristic,heuristic,heuristic")
    ap.add_argument("--ckpt",    type=str, default="")
    ap.add_argument("--ismcts-sims", type=int, default=400)
    args = ap.parse_args()

    names = args.agents.split(",")
    if len(names) != sk.N_PLAYERS:
        raise SystemExit(f"--agents needs exactly {sk.N_PLAYERS} entries")

    # Build one instance per UNIQUE name (re-use across rotations).
    agent_pool: dict[str, object] = {}
    for n in names:
        if n not in agent_pool:
            agent_pool[n] = make_agent(n, args.ckpt or None, args.ismcts_sims)

    rng = sk.Rng(seed=args.seed)
    rotations = sk.N_PLAYERS
    games_per_rot = max(1, args.games // rotations)

    stats = {n: dict(sum_=0.0, sum_sq=0.0, wins=0, games=0) for n in agent_pool}

    t0 = time.perf_counter()
    for rot in range(rotations):
        seating = [names[(p + rot) % sk.N_PLAYERS] for p in range(sk.N_PLAYERS)]
        for g in range(games_per_rot):
            scores = play_one(seating, agent_pool, rng)
            winner = int(np.argmax(scores))
            for p in range(sk.N_PLAYERS):
                st = stats[seating[p]]
                st["sum_"] += scores[p]
                st["sum_sq"] += scores[p] * scores[p]
                st["games"] += 1
                if p == winner: st["wins"] += 1
    elapsed = time.perf_counter() - t0

    total_games = games_per_rot * rotations
    print(f"\n=== {games_per_rot} games per rotation × {rotations} rotations = {total_games} games ===")
    print(f"Agent line-up: {args.agents}")
    print(f"Elapsed: {elapsed:.1f}s ({total_games/elapsed:.1f} games/sec)\n")
    print(f"{'agent':<12} {'games':>7} {'avgScore':>10} {'stdScore':>10} {'wins':>6} {'winRate%':>10}")
    print("-" * 63)
    for n, st in stats.items():
        mean = st["sum_"] / st["games"]
        var  = max(0.0, st["sum_sq"]/st["games"] - mean*mean)
        sd   = math.sqrt(var)
        wr   = 100.0 * st["wins"] / st["games"]
        print(f"{n:<12} {st['games']:>7} {mean:>10.2f} {sd:>10.2f} {st['wins']:>6} {wr:>9.1f}%")


if __name__ == "__main__":
    main()
