# Abstraction & pruning — milestone M1 (round 1, 4 players)

Same budget as the MCCFR baseline in `../mccfr_vs_exact/` (200 batches ×
65,536 deals = 52M traversals); NashConv is exact.

| run | solve time | NashConv | seat 4 gain | infosets | nodes / traversal |
|---|---|---|---|---|---|
| MCCFR, exact keys (baseline) | 19.2 s | 0.153 | 0.149 | 564k | 15.7 |
| MCCFR, suit isomorphism | 14.1 s | **0.055** | 0.052 | 247k | 15.7 |
| MCCFR, suit isomorphism + pruning 0.95 | 8.3 s | 0.058 | 0.053 | 139k | **10.1** |

- **Suit isomorphism** (lossless: Yellow/Green/Purple are interchangeable)
  pools the samples of equivalent situations. That cuts NashConv ~3× at
  equal cost, and it is still falling at the end of the run instead of
  plateauing.
- **Regret-based pruning** (Pluribus: skip currently-unplayed traverser
  actions 95 % of the time) cuts work per traversal by 36 % here at
  practically no quality cost. Its real payoff comes in the big rounds,
  once strategies are sparse.
- What remains is seat 4's rare Tigress situations. Lossless reductions
  cannot pool them further; this needs the lossy feature abstraction (M2),
  whose cost is again measured here.

Commands:

```
sk_solve_round --method mccfr --abstraction suit-iso --round 1 --iters 200 --eval-every 40 --deals-per-batch 65536
sk_solve_round --method mccfr --abstraction suit-iso --prune 0.95 --prune-after 20 --round 1 --iters 200 --eval-every 40 --deals-per-batch 65536
```
