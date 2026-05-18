"""Analyse a trained model -- extract human-readable strategy insights.

For each game the script logs every bid and card-play decision with context
(hand strength, bid status, whether the played card would win the trick, etc.)
and produces a report you can use to understand what the AI learned and how
to play like it yourself.

Usage
-----
    python -m skull_king.training.analyze
    python -m skull_king.training.analyze --model models/skull_king/ppo_selfplay_v1_100000
    python -m skull_king.training.analyze --games 300 --seed 42
"""
from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.cards import Card, CardType, TigressMode, TRUMP_SUIT
from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import SkullKingEnv
from skull_king.game_state import GamePhase
from skull_king.resolver import TrickResolver
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.self_play_env import SB3AgentWrapper
from skull_king.trick import PlayedCard

_N_PLAYERS = 4
_PPO_SEAT = 0


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def _hand_strength(hand: list[Card]) -> float:
    """Expected-win weight sum (same formula as HeuristicAgent)."""
    total = 0.0
    for c in hand:
        if c.card_type == CardType.NUMBERED:
            total += (0.15 + (c.value / 14) * 0.60) if c.suit == TRUMP_SUIT else (c.value / 14) * 0.15
        elif c.card_type == CardType.SKULL_KING:
            total += 0.95
        elif c.card_type == CardType.PIRATE:
            total += 0.75
        elif c.card_type == CardType.TIGRESS:
            total += 0.45
        elif c.card_type == CardType.MERMAID:
            total += 0.25
    return total


def _card_strength(card: Card, mode: Optional[TigressMode] = None) -> float:
    """0-10 strength: SK=10, Pirate/T-Pirate=8, Mermaid=5, trump=1-6, colour=0-1, escape=0."""
    if card.card_type == CardType.SKULL_KING:
        return 10.0
    if card.card_type == CardType.PIRATE:
        return 8.0
    if card.card_type == CardType.TIGRESS:
        return 8.0 if mode == TigressMode.PIRATE else 0.0
    if card.card_type == CardType.MERMAID:
        return 5.0
    if card.card_type == CardType.ESCAPE:
        return 0.0
    # Numbered
    return (1.0 + (card.value / 14) * 5.0) if card.suit == TRUMP_SUIT else (card.value / 14)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect(model: MaskablePPO, n_games: int, seed: int) -> tuple[list, list, list]:
    """Play n_games games vs random opponents; return (bids, plays, round_outcomes)."""
    util_env = SkullKingEnv(n_players=_N_PLAYERS)
    rng = random.Random(seed)

    bids: list[dict] = []           # one entry per PPO bid decision
    plays: list[dict] = []          # one entry per PPO card-play decision
    round_outcomes: list[dict] = [] # one entry per completed round

    for game_i in range(n_games):
        engine = GameEngine(n_players=_N_PLAYERS, seed=seed + game_i)
        state = engine.start()
        ppo_prev_score = 0
        round_bids: dict[int, int] = {}  # round -> bid placed by PPO

        while state.phase != GamePhase.GAME_OVER:
            seat = state.current_player_index
            cur_round = state.round_number

            # -- Bidding ------------------------------------------------------
            if state.phase == GamePhase.BIDDING:
                if seat == _PPO_SEAT:
                    completed = engine.completed_tricks_this_round
                    obs = util_env._build_observation_for(state, seat, completed)
                    mask = util_env._action_masks_for(state, seat)
                    act, _ = model.predict(obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True)
                    bid = max(0, min(int(act[0]), state.round_number))

                    hand = list(state.player_states[seat].hand)
                    bids.append({
                        "round": cur_round,
                        "hand_strength": _hand_strength(hand),
                        "bid": bid,
                        "bid_rate": bid / cur_round,
                    })
                    round_bids[cur_round] = bid
                    state = engine.place_bid(seat, bid)
                else:
                    state = engine.place_bid(seat, rng.randint(0, state.round_number))

            # -- Playing ------------------------------------------------------
            else:
                if seat == _PPO_SEAT:
                    ps = state.player_states[seat]
                    bid_now = ps.bid if ps.bid is not None else 0
                    tricks_now = ps.tricks_won_this_round
                    want_win = tricks_now < bid_now

                    completed = engine.completed_tricks_this_round
                    obs = util_env._build_observation_for(state, seat, completed)
                    mask = util_env._action_masks_for(state, seat)
                    act, _ = model.predict(obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True)
                    card, mode = util_env._decode_play_action(int(act[0]))

                    played_so_far = list(state.current_trick_cards)
                    if played_so_far:
                        candidate = PlayedCard(
                            card=card,
                            player_index=seat,
                            play_order=len(played_so_far) + 1,
                            tigress_mode=mode,
                        )
                        would_win: Optional[bool] = (
                            TrickResolver.resolve(played_so_far + [candidate]).winner_player_index == seat
                        )
                    else:
                        would_win = None  # leading -- no one to beat yet

                    is_escape = card.card_type == CardType.ESCAPE or (
                        card.card_type == CardType.TIGRESS and mode == TigressMode.ESCAPE
                    )
                    is_special = card.card_type in (CardType.SKULL_KING, CardType.PIRATE, CardType.MERMAID) or (
                        card.card_type == CardType.TIGRESS and mode == TigressMode.PIRATE
                    )

                    plays.append({
                        "round": cur_round,
                        "want_win": want_win,
                        "bid": bid_now,
                        "card_type": card.card_type.value,
                        "card_strength": _card_strength(card, mode),
                        "would_win": would_win,
                        "is_leading": not played_so_far,
                        "is_escape": is_escape,
                        "is_special": is_special,
                    })
                    state = engine.play_card(seat, card, mode)
                else:
                    hand_l = list(state.player_states[seat].hand)
                    legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand_l)
                    c = rng.choice(legal)
                    m: Optional[TigressMode] = None
                    if c.card_type == CardType.TIGRESS:
                        m = rng.choice([TigressMode.PIRATE, TigressMode.ESCAPE])
                    state = engine.play_card(seat, c, m)

            # -- Detect round end ----------------------------------------------
            # score_delta > 0 <-> bid hit (missed bids give -10*round, hits give >=+10)
            is_over = state.phase == GamePhase.GAME_OVER
            if (state.round_number != cur_round or is_over) and cur_round in round_bids:
                new_score = state.player_states[_PPO_SEAT].total_score
                delta = new_score - ppo_prev_score
                round_outcomes.append({
                    "round": cur_round,
                    "bid": round_bids[cur_round],
                    "score_delta": delta,
                    "hit": delta > 0,
                })
                ppo_prev_score = new_score
                del round_bids[cur_round]

    return bids, plays, round_outcomes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(values: list) -> str:
    return f"{np.mean(values):.0%}" if values else "n/a"


def report(model: MaskablePPO, model_label: str, n_games: int, seed: int) -> None:
    W = 62
    print("=" * W)
    print(" Skull King -- Model Strategy Analysis")
    print(f" Model : {model_label}")
    print(f" Games : {n_games} analysis games + 100 tournament games")
    print("=" * W)

    # -- Tournament benchmark ------------------------------------------------
    print("\nTOURNAMENT BENCHMARK  (100 games each)")
    ppo = SB3AgentWrapper(model, n_players=_N_PLAYERS, name="PPO")
    runner = TournamentRunner(seed=seed)
    r_r = runner.run([ppo] + [RandomAgent(i) for i in range(_N_PLAYERS - 1)], n_games=100)
    r_h = runner.run([ppo] + [HeuristicAgent() for _ in range(_N_PLAYERS - 1)], n_games=100)
    wr_r = r_r.win_rates()["PPO"]
    wr_h = r_h.win_rates()["PPO"]
    avg_r = r_r.avg_scores()["PPO"]
    avg_h = r_h.avg_scores()["PPO"]
    print(f"  vs Random:    win={wr_r:5.1%}  avg_score={avg_r:+.0f}")
    print(f"  vs Heuristic: win={wr_h:5.1%}  avg_score={avg_h:+.0f}")
    print(f"  (Random baseline: win=25.0%  avg_score~=-100 vs heuristic)")

    # -- Collect -------------------------------------------------------------
    print(f"\nCollecting {n_games} games of play data...", end=" ", flush=True)
    bids, plays, round_outcomes = _collect(model, n_games, seed)
    print(f"{len(bids)} bids  |  {len(plays)} plays  |  {len(round_outcomes)} rounds")

    # -- Bid calibration -----------------------------------------------------
    print("\nBIDDING: HAND STRENGTH -> BID")
    print("  Hand strength = sum of card weights (SK=0.95, Pirate=0.75, Mermaid=0.25,")
    print("                  Black14=0.75, Black1=0.15, colour cards <=0.15, escape=0)")
    buckets = [(0.0, 1.0, "very weak"), (1.0, 2.0, "weak"), (2.0, 3.0, "medium"),
               (3.0, 4.5, "strong"), (4.5, 99.0, "very strong")]
    print(f"  {'Strength':>14}  {'bid/round':>10}  {'avg bid':>8}  n")
    for lo, hi, label in buckets:
        sub = [b for b in bids if lo <= b["hand_strength"] < hi]
        if not sub:
            continue
        print(f"  {lo:.1f}-{'inf' if hi > 50 else f'{hi:.1f}'} ({label:<11})  "
              f"{np.mean([b['bid_rate'] for b in sub]):>9.1%}  "
              f"{np.mean([b['bid'] for b in sub]):>8.1f}  {len(sub)}")

    all_hs = np.array([b["hand_strength"] for b in bids])
    all_br = np.array([b["bid_rate"] for b in bids])
    bid_rule: Optional[tuple[float, float]] = None
    if len(all_hs) >= 20:
        slope, intercept = np.polyfit(all_hs, all_br, 1)
        if slope > 0.005:
            pts_per_trick = 1.0 / slope
            bid_rule = (pts_per_trick, intercept)
            print(f"\n  Linear fit: bid_rate ~= {slope:.3f}*strength + {intercept:.3f}")
            print(f"  -> Rule: bid 1 trick per ~{pts_per_trick:.1f} strength points")

    # -- Bid accuracy by round ---------------------------------------------
    print("\nBID ACCURACY BY ROUND")
    per_round: dict[int, list[bool]] = {}
    for ro in round_outcomes:
        per_round.setdefault(ro["round"], []).append(ro["hit"])
    line1, line2 = "  ", "  "
    for r in range(1, 11):
        hits = per_round.get(r, [])
        acc = f"R{r}={np.mean(hits):.0%}" if hits else f"R{r}=n/a"
        if r <= 5:
            line1 += f"{acc:<10}"
        else:
            line2 += f"{acc:<10}"
    print(line1.rstrip())
    print(line2.rstrip())
    overall_acc = np.mean([ro["hit"] for ro in round_outcomes]) if round_outcomes else 0.0
    print(f"  Overall: {overall_acc:.1%}   (random baseline ~= 20%)")

    # -- Play style --------------------------------------------------------
    print("\nPLAY STYLE")
    behind = [p for p in plays if p["want_win"]]
    ahead  = [p for p in plays if not p["want_win"]]
    for subset, label in [(behind, "When NEEDING tricks (bid > tricks won)"),
                          (ahead,  "When AVOIDING tricks (bid met / bid=0)")]:
        if not subset:
            continue
        avg_str = np.mean([p["card_strength"] for p in subset])
        esc_pct  = np.mean([p["is_escape"] for p in subset])
        spec_pct = np.mean([p["is_special"] for p in subset])
        following = [p for p in subset if not p["is_leading"] and p["would_win"] is not None]
        win_pct = np.mean([p["would_win"] for p in following]) if following else float("nan")
        print(f"  {label}")
        print(f"    avg card strength : {avg_str:.1f}/10")
        print(f"    plays a winner    : {win_pct:.0%}"
              f" (of following plays)" if not np.isnan(win_pct) else "    plays a winner: n/a")
        print(f"    escape rate       : {esc_pct:.0%}"
              f"  |  special rate: {spec_pct:.0%}")

    # -- Special card timing -----------------------------------------------
    print("\nSPECIAL CARD TIMING  (avg round used; played_when_behind=needing tricks)")
    specials = [
        ("[SK]  Skull King", "SKULL_KING"),
        ("[P]   Pirate    ", "PIRATE"),
        ("[M]   Mermaid   ", "MERMAID"),
    ]
    for label, ctype in specials:
        sub = [p for p in plays if p["card_type"] == ctype]
        if not sub:
            print(f"  {label}: not played")
            continue
        print(f"  {label}: avg_round={np.mean([p['round'] for p in sub]):.1f}  "
              f"played_when_behind={_pct([p['want_win'] for p in sub])}  (n={len(sub)})")

    escapes = [p for p in plays if p["is_escape"]]
    if escapes:
        bid0 = [p for p in escapes if p["bid"] == 0]
        print(f"  [E]   Escape    : avg_round={np.mean([p['round'] for p in escapes]):.1f}  "
              f"used_when_bid=0: {len(bid0)/len(escapes):.0%}  (n={len(escapes)})")

    # -- Human takeaways ---------------------------------------------------
    print("\nHUMAN STRATEGY TAKEAWAYS")
    n = 1

    if bid_rule is not None:
        pts, intercept = bid_rule
        base = max(0.0, intercept)
        print(f"  {n}. Bidding: bid 1 trick per ~{pts:.1f} strength pts (base {base:.2f})")
        print(f"     e.g. SK + 2 Pirates (strength~=2.45): bid ~{round(2.45/pts + base)}")
        n += 1

    if behind and ahead:
        b_str = np.mean([p["card_strength"] for p in behind])
        a_str = np.mean([p["card_strength"] for p in ahead])
        b_esc = np.mean([p["is_escape"] for p in behind])
        a_esc = np.mean([p["is_escape"] for p in ahead])
        print(f"  {n}. Need tricks : play strong cards (avg {b_str:.1f}/10), escape only {b_esc:.0%}")
        n += 1
        print(f"  {n}. Avoid tricks: play weak cards (avg {a_str:.1f}/10), escape when possible ({a_esc:.0%})")
        n += 1

    if overall_acc > 0:
        verdict = "good" if overall_acc >= 0.45 else "developing" if overall_acc >= 0.30 else "weak"
        print(f"  {n}. Bid accuracy {overall_acc:.0%} ({verdict}) -- random baseline is ~20%")
        n += 1

    if wr_r >= 0.50:
        print(f"  {n}. Beats random reliably ({wr_r:.0%}) -- safe to use as an opponent")
        n += 1
    if wr_h >= 0.25:
        print(f"  {n}. Competitive vs Heuristic ({wr_h:.0%}) -- model has learned real strategy")

    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a trained Skull King model")
    parser.add_argument(
        "--model",
        default="models/skull_king/ppo_selfplay_v1_100000",
        help="Path to model zip (suffix optional)",
    )
    parser.add_argument("--games", type=int, default=200, help="Analysis games (default 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = MaskablePPO.load(args.model)
    report(model, args.model, n_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
