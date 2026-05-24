"""Evaluate all cfr_v6_heuristic checkpoints + reference models."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.cfr.agent import CFRAgent
from skull_king.training.cfr.networks import StrategyNet

N_PLAYERS = 4
N_GAMES = 200
MODEL_DIR = "models/skull_king"
BEWERTUNG_DIR = "bewertung"

CHECKPOINTS = [
    (f"cfr_v6_iter{i}", os.path.join(MODEL_DIR, f"cfr_v6_heuristic_iter{i}_strat.pt"))
    for i in range(250, 2001, 250)
]

# Reference: v2_server (best historical model) and v3_pc for comparison
COMPARE = [
    ("cfr_final (v2_server)", os.path.join(BEWERTUNG_DIR, "cfr_final_strat.pt")),
    (f"cfr_v3_pc_iter1000", os.path.join(MODEL_DIR, "cfr_v3_pc_iter1000_strat.pt")),
]


def load_agent(path: str, name: str) -> CFRAgent | None:
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    net = StrategyNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return CFRAgent(net, n_players=N_PLAYERS, name=name)


def eval_agent(agent: CFRAgent) -> dict:
    runner = TournamentRunner(seed=42)
    r_r = runner.run([agent] + [RandomAgent(i) for i in range(N_PLAYERS - 1)], n_games=N_GAMES)
    r_h = runner.run([agent] + [HeuristicAgent() for _ in range(N_PLAYERS - 1)], n_games=N_GAMES)
    return {
        "wr_random": r_r.win_rates().get(agent.name, 0.0),
        "wr_heuristic": r_h.win_rates().get(agent.name, 0.0),
        "avg_random": r_r.avg_scores().get(agent.name, 0.0),
        "avg_heuristic": r_h.avg_scores().get(agent.name, 0.0),
    }


def print_table(rows: list[tuple[str, dict]]) -> None:
    header = f"{'Model':<30}  {'WR vs Rand':>10}  {'WR vs Heur':>10}  {'Avg vs Rand':>11}  {'Avg vs Heur':>11}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<30}  {m['wr_random']:>10.1%}  {m['wr_heuristic']:>10.1%}"
            f"  {m['avg_random']:>+11.1f}  {m['avg_heuristic']:>+11.1f}"
        )


print("\n" + "=" * 80)
print("  cfr_v6_heuristic  (2000 iter, all 5 fixes)")
print("=" * 80)
rows_v6: list[tuple[str, dict]] = []
for name, path in CHECKPOINTS:
    agent = load_agent(path, name)
    if agent is None:
        continue
    m = eval_agent(agent)
    print(f"  {name}: WR_rand={m['wr_random']:.1%}  WR_heur={m['wr_heuristic']:.1%}"
          f"  avg_rand={m['avg_random']:+.1f}  avg_heur={m['avg_heuristic']:+.1f}")
    rows_v6.append((name, m))

print("\n" + "=" * 80)
print("  Reference models (v2_server, v3_pc)")
print("=" * 80)
rows_ref: list[tuple[str, dict]] = []
for name, path in COMPARE:
    agent = load_agent(path, name)
    if agent is None:
        continue
    m = eval_agent(agent)
    print(f"  {name}: WR_rand={m['wr_random']:.1%}  WR_heur={m['wr_heuristic']:.1%}"
          f"  avg_rand={m['avg_random']:+.1f}  avg_heur={m['avg_heuristic']:+.1f}")
    rows_ref.append((name, m))

print("\n\n" + "=" * 80)
print("  SUMMARY TABLE — cfr_v6_heuristic")
print("=" * 80)
print_table(rows_v6)

if rows_ref:
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE — Reference models")
    print("=" * 80)
    print_table(rows_ref)
