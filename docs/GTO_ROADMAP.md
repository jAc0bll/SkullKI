# Solving Skull King — GTO roadmap

Goal: an equilibrium ("GTO") strategy for every situation of base-game
Skull King, first for **4 players**, later a **2-player mode** where the
strategy is provably GTO.

## What "GTO" means here

| Players | Solution concept | Guarantee | How we measure it |
|---|---|---|---|
| 2 | Nash equilibrium of a 2-player zero-sum game | Unexploitable: no opponent strategy wins in expectation | Exploitability (exact for small rounds) |
| 4 | Nash equilibrium of a 4-player game | No *single* player gains by deviating; no guarantee against several players deviating together. Same situation as 6-max poker solvers | **NashConv** = Σ over players of best-response gain (0 ⇔ Nash) |

CFR on games with more than 2 players provably converges only to a coarse
correlated equilibrium, so for 4 players we always **measure** NashConv
instead of assuming it.

## Decomposition: rounds are independent games

The deck is reshuffled every round. Rounds only interact through the score
vector (and the rotating start player). So:

1. **Points objective** (this phase): each round is a separate game; seat 1
   = the round's start player. Utility = own round points − mean of the
   others' (`Utility::Relative`, zero-sum), or own points (`Absolute`).
2. **Win objective** (later): a value function `V(scores, round)` (like ICM
   in poker tournaments) turns round points into change in win probability;
   rounds are then re-solved with that utility. This matters most in the last
   rounds.

## Rules (base game, German Ravensburger/Schmidt rules in `rules.txt`)

Engine decisions that affect the solution:

- **Bidding is simultaneous.** `observe()` hides other players' bids until
  all are in; infoset keys during bidding contain only seat + own hand.
- 14s score for whoever holds them at round end (+10 colored, +20 black),
  no matter who played them; bonuses only when the bid is hit.
- **Mermaid bonus (+20) only for a Mermaid captured by a Pirate** (incl.
  Tigress-as-Pirate). A Mermaid that wins a trick earns nothing for herself
  or a second Mermaid. (Fixed — the engine used to award every captured
  Mermaid, including the winner's own.)
- Pirate + Skull King + Mermaid in one trick → first Mermaid wins, +40 only.
- Escape-led trick: the first colored card sets the suit to follow.
- Publicly revealed voids (`GameState::voidSuits`) are tracked; the
  determinizer only deals hands consistent with them.

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Engine fixes: simultaneous bidding, voids, mermaid bonus, allocation-free legal moves | ✅ |
| 1 | Exact solver for one round: full-width Discounted CFR + exact best response / NashConv (`solver/`, `sk_solve_round`) | ✅ round 1 |
| 2a | Sampled CFR (external-sampling MCCFR) measured against the exact round 1 | ✅ |
| 2b | Abstraction framework: suit isomorphism (lossless), feature abstraction (lossy), pruning — round 1 solved in 6 s at NashConv 0.0001 | ✅ |
| 2c | Rounds 4–10 (4p): tables saturate (≥ 90 % unique keys) → neural networks on the same features (Deep CFR family with sampled traverser actions), validated against round 1 | next |
| 3 | Decision-time subgame solving for a concrete spot | planned |
| 4 | Win-probability value function, re-solve with win utility | planned |
| 2p | 2-player engine mode, exact solutions for small rounds, true GTO | planned |

### Round 1, 4 players (exact)

- 12,588,740 deals (kind level), 3,351,556 infosets (≈ 3.35M of them are
  Tigress declarations; 244 bidding infosets).
- One DCFR iteration ≈ 3.5 s, one exact NashConv evaluation ≈ 25 s on a
  12-core i7-12700K.
- **Solved:** NashConv 31.3 → < 0.0001 points/round after 300 iterations.
  The equilibrium bid is pure and the same for every seat: bid 1 with
  Black 5–14, Mermaid, Pirate, Tigress, Skull King; otherwise 0. Leading
  costs ~1.3 relative points. Details: `results/round1_4p/README.md`.

### Why round 2+ needs a different method

Round 2 with 4 players has on the order of 10¹³ deals and 10¹⁰+ infosets
(e.g. seat 4 in trick 1: 1,891 hands × 81 bid profiles × ~2·10⁵ histories).
Full-width tabular CFR is out of reach, as is an exact best response for
4 players. Plan:

1. **Sampled CFR** (external-sampling MCCFR) on the same `RoundState` /
   infoset keys, measured on round 1 (`results/mccfr_vs_exact/`): seats 1–3
   are near-exact after 19 s, but seat 4's ~3.3M rare Tigress infosets are
   mostly never sampled (NashConv plateaus ~0.15). An ES traversal also
   grows ~6× per round (31M nodes in round 10). Hence: sample the
   traverser's actions too, and generalise with networks.
2. **Function approximation** for rounds ≥ 3 (Deep CFR / ESCHER-style regret
   networks, or regularised self-play such as MMD / R-NaD). Pick the method
   empirically: whichever gives the lowest NashConv on rounds 1–2 (2p: 1–3),
   where exact values exist, gets scaled to round 10.
3. Approximate exploitability for big rounds via local best response or a
   trained exploiter agent.

## 2-player mode (true GTO)

- Make the player count a runtime/template parameter (`N_PLAYERS = 4` is a
  compile-time constant today).
- **Decided: pure heads-up, no Graybeard.** (The official 2-player rules add
  Graybeard, a ghost hand played second in every trick; we do not model it.)
- 2p rounds 1–2 are small enough for exact solving with the existing
  full-width solver (round 2 heads-up: ~5·10⁶ deals),
  round 3 with sampling.
