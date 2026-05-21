"""Pre-compute strategy tables for GTO analysis."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from skull_king.analysis.explorer import StrategyExplorer, hand_features, hand_strength
from skull_king.cards import (
    Card, CardType, Suit, TRUMP_SUIT, build_deck, NUM_ROUNDS
)


def _deal_random_hand(round_num: int, rng: random.Random) -> list[Card]:
    """Deal a random hand of round_num cards."""
    deck = build_deck()
    rng.shuffle(deck)
    return deck[:round_num]


def generate_bid_table(
    explorer: StrategyExplorer,
    n_samples_per_round: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Sample random hands per round and return bid strategy data."""
    rng = random.Random(seed)
    rows = []
    for round_num in range(1, NUM_ROUNDS + 1):
        for _ in range(n_samples_per_round):
            hand = _deal_random_hand(round_num, rng)
            result = explorer.query_bid(hand, round_num)
            feat = hand_features(hand)
            row = {
                "round": round_num,
                "recommended_bid": result.recommended_bid,
                "expected_bid": sum(b * p for b, p in result.probabilities.items()),
                **feat,
                **{f"p_bid_{b}": result.probabilities.get(b, 0.0)
                   for b in range(round_num + 1)},
            }
            rows.append(row)
    return rows


def generate_bid_summary(bid_table: list[dict]) -> dict:
    """Aggregate bid table into summary statistics per round."""
    summary = {}
    for round_num in range(1, NUM_ROUNDS + 1):
        rows = [r for r in bid_table if r["round"] == round_num]
        if not rows:
            continue
        avg_bid = np.mean([r["expected_bid"] for r in rows])
        avg_strength = np.mean([r["strength"] for r in rows])
        # Bid distribution
        max_bid = round_num
        dist = {}
        for b in range(max_bid + 1):
            key = f"p_bid_{b}"
            dist[b] = float(np.mean([r.get(key, 0.0) for r in rows]))
        summary[round_num] = {
            "avg_expected_bid": float(avg_bid),
            "avg_hand_strength": float(avg_strength),
            "bid_distribution": dist,
            "n_samples": len(rows),
        }
    return summary


def generate_special_bid_table(
    explorer: StrategyExplorer,
    round_num: int = 5,
    n_samples: int = 300,
    seed: int = 99,
) -> list[dict]:
    """Show how having Skull King / Pirates / Mermaids changes bidding."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        hand = _deal_random_hand(round_num, rng)
        result = explorer.query_bid(hand, round_num)
        feat = hand_features(hand)
        rows.append({
            **feat,
            "recommended_bid": result.recommended_bid,
            "expected_bid": sum(b * p for b, p in result.probabilities.items()),
            "round": round_num,
        })
    return rows
