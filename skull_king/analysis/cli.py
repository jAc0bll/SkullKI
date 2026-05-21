"""CLI entry point for generating GTO analysis reports."""
from __future__ import annotations

import argparse
import json
import os

from skull_king.analysis.explorer import StrategyExplorer, hand_strength, hand_features
from skull_king.analysis.tables import generate_bid_table, generate_bid_summary
from skull_king.cards import build_deck, CardType
import random


DEFAULT_MODEL = "models/skull_king/cfr_final_strat.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GTO strategy analysis")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--output", default="gto_tables.json")
    parser.add_argument("--round", type=int, default=None, help="Analyze specific round only")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    explorer = StrategyExplorer(args.model)

    print("Generating bid table...")
    bid_table = generate_bid_table(explorer, n_samples_per_round=args.samples)
    bid_summary = generate_bid_summary(bid_table)

    # Print human-readable summary
    print("\n=== BID STRATEGY BY ROUND ===")
    for rn, stats in sorted(bid_summary.items()):
        dist = stats["bid_distribution"]
        top_bid = max(dist, key=dist.get)
        print(f"Round {rn:2d}: avg_bid={stats['avg_expected_bid']:.2f}  "
              f"top_bid={top_bid}  strength={stats['avg_hand_strength']:.2f}")
        for b, p in sorted(dist.items()):
            if p > 0.03:
                bar = "█" * int(p * 30)
                print(f"         bid {b}: {p:5.1%}  {bar}")

    # Save JSON
    out = {"bid_summary": {str(k): v for k, v in bid_summary.items()}}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
