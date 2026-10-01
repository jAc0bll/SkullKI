# Round 1, 4 players — exact equilibrium

Produced with

```
sk_solve_round --round 1 --iters 300 --eval-every 50 --out bidding_strategy.csv
```

(full-width Discounted CFR, α=1.5 β=0 γ=2, utility = own points − mean of
the others', 12,588,740 deals, 3,351,556 infosets, ~17 min on an i7-12700K).

## Convergence (exact NashConv, points per round)

| iteration | 1 | 50 | 100 | 150 | 300 |
|---|---|---|---|---|---|
| NashConv | 31.28 | 0.0037 | 0.0005 | 0.0001 | < 0.0001 |

NashConv is the sum, over the 4 players, of what each could gain by
switching to a best response while the others keep the solution. Below
0.0001 points per round, no single player can exploit this strategy by any
meaningful margin.

## Equilibrium bidding (seat 1 leads the trick)

The equilibrium is pure and **identical for all four seats**:

| card | bid |
|---|---|
| Yellow / Green / Purple 1–14 | 0 |
| Black 1–4 | 0 |
| Black 5–14 | 1 |
| Escape | 0 |
| Mermaid, Pirate, Tigress, Skull King | 1 |

Sanity check by hand: bidding 1 scores +20/−10 and bidding 0 scores
+10/−10, so bidding 1 pays off once P(win) > 0.4. Black 5 loses to 18 of the
other 69 cards (Black 6–14, 5 Pirates, Tigress, Skull King, 2 Mermaids):
P(no opponent holds one) = C(51,3)/C(69,3) = 0.397. Black 4 (19 beaters)
gives 0.374. The threshold sits exactly between Black 4 and Black 5. A
colored 14 wins only 29 % of the time, so it bids 0.

Green/Purple strategies equal Yellow to 1e-20 (suit symmetry, a free check
of the solver).

## Value of each seat (relative points per round, under the equilibrium)

| seat 1 (leads) | seat 2 | seat 3 | seat 4 |
|---|---|---|---|
| −1.31 | +0.50 | +0.51 | +0.30 |

Leading round 1 is a disadvantage of about 1.3 points relative to the field.

## Tigress declaration (when holding Tigress you always bid 1)

Probability of declaring Pirate, weighted by how often each situation occurs:
seat 1 100 %, seat 2 93 %, seat 3 88 %, seat 4 86 %. The remainder are
situations where Pirate cannot win anyway (e.g. Skull King already played),
where declaring Escape avoids giving the Skull King player a +30 bonus.
The "own bid 0" lines in `solve_log.txt` are off the equilibrium path
(Tigress never bids 0), so their 0.5 is meaningless. The tool now labels
them as such.

## Files

- `solve_log.txt` — solver output.
- `bidding_strategy.csv` — equilibrium bid probabilities for every
  (seat, card) bidding infoset.
