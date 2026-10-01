# Abstraction & pruning — milestone M1 (round 1, 4 players)

Same budget as the MCCFR baseline in `../mccfr_vs_exact/` (200 batches ×
65,536 deals = 52M traversals); NashConv is exact.

| run | solve time | NashConv | seat 4 gain | infosets | nodes / traversal |
|---|---|---|---|---|---|
| MCCFR, exact keys (baseline) | 19.2 s | 0.153 | 0.149 | 564k | 15.7 |
| MCCFR, suit isomorphism | 14.1 s | **0.055** | 0.052 | 247k | 15.7 |
| MCCFR, suit isomorphism + pruning 0.95 | 8.3 s | 0.058 | 0.053 | 139k | **10.1** |
| MCCFR, **feature abstraction** (M2) | **6.0 s** | **0.0001** | 0.0001 | **3.4k** | 15.7 |

- **Suit isomorphism** (lossless: Yellow/Green/Purple are interchangeable)
  pools the samples of equivalent situations. That cuts NashConv ~3× at
  equal cost, and it is still falling at the end of the run instead of
  plateauing.
- **Regret-based pruning** (Pluribus: skip currently-unplayed traverser
  actions 95 % of the time) cuts work per traversal by 36 % here at
  practically no quality cost. Its real payoff comes in the big rounds,
  once strategies are sparse.
- What remains is seat 4's rare Tigress situations. Lossless reductions
  cannot pool them further.

## M2: feature abstraction

`--abstraction features` (solver/src/feature_abstraction.cpp) describes a
decision by strategic features instead of card identities: every
player's bid status, what currently wins the trick and what bonus rides on
it, unseen specials/trumps, public voids, and per-card descriptors (would it
win now, unseen higher cards, suit relation, suit length, 14). Cards with
equal descriptors become one action.

**Round 1: exact-quality solution in 6 s with 3,360 infosets** (exact
DCFR needs ~150 full iterations, ~8 min, over 3.35M infosets for the same
NashConv). The features capture what matters and pool the rare Tigress
situations.

## Why tables stop scaling: key saturation

`sk_bench_traversal --key-stats <abstraction>` samples decision points
from random play and counts distinct table keys
(`key_saturation_*.txt`):

| round | features, 1e6 samples | suit-iso, 1e6 samples |
|---|---|---|
| 1 | 1,777 | 7,177 |
| 3 | 389,430 | 668,599 |
| 5 | 678,011 | 986,859 |
| 10 | 914,201 | 1,000,000 |

From round ~4 on almost every situation is unique even under the feature
abstraction. A table cannot pool anything there unless the abstraction is
made so coarse that it loses quality. The features themselves are right
(round 1); what does not scale is *looking them up in a table*. Next
step: feed the same information to a neural network, which generalises
between similar but non-identical situations (Deep CFR family, with
sampled traverser actions to avoid the ~6x-per-round traversal blow-up).

Commands:

```
sk_solve_round --method mccfr --abstraction suit-iso --round 1 --iters 200 --eval-every 40 --deals-per-batch 65536
sk_solve_round --method mccfr --abstraction suit-iso --prune 0.95 --prune-after 20 --round 1 --iters 200 --eval-every 40 --deals-per-batch 65536
```
