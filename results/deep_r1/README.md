# Neural CFR (milestone N1) — round 1 gate, 4 players

Method: ESCHER-style neural CFR (`train/deep_cfr.py` + `sk_deep`). The
traverser acts uniformly at random and regrets come from a value network,
so the cost per trajectory is linear in the round length. Every number
below is the exact NashConv of the network strategy in the real game.

## Runs (i7-12700K, CPU only)

| run | iterations | average-strategy net | current (regret-matching) net |
|---|---|---|---|
| v1, exact action values (diagnostic) | 30 | 0.128 | 0.036 |
| v1, value net (stopped at 12) | 10 | 0.649 (exact-values run at 10: 0.627) | 0.069 (0.070) |
| **v2**, value net, 3× regret training, avg weighted by t² | 30 | **0.036** | 0.112 (0.0009 at iter 20) |

v2 trajectory of the average net: 0.74 (5) → 0.107 (10) → 0.054 (15) →
0.046 (20) → 0.034 (25) → 0.036 (30). About 70 s per iteration, most of
it network training on the CPU.

Command (v2):

```
.venv/Scripts/python train/deep_cfr.py --round 1 --iters 30 --eval-every 5 \
    --regret-steps 6000 --policy-steps 8000 --avg-weight-power 2 --workdir runs/deep_r1_v2
```

### Findings

1. **The value network costs nothing measurable.** At equal iterations the
   value-net run tracks the run with exact action values (0.649 vs 0.627).
   This is the component that makes big rounds affordable.
2. **Average-strategy weighting matters most:** weighting iteration t by t²
   (DCFR's γ=2) instead of t made the average net 3.5× better.
3. **What is left is one near-indifferent decision.** `sk_deep chart` shows
   the networks bid exactly like the exact solution everywhere except at the
   Black 4 / Black 5 boundary (P(win) 37 % / 40 % against a 40 % breakeven).
   The EV difference there is ~1.3 points (Black 4) and ~0.1 points
   (Black 5), so the residual NashConv of 0.036 points/round is under half a
   point over a whole game. The self-imposed gate of 0.01 was missed by 3×;
   accepted as practically exact, with longer training planned on rented
   hardware.

## Exploiter calibration (exploitability where exact evaluation is impossible)

From round 2 on, an exact best response is out of reach for 4 players.
The replacement is a **trained exploiter**: the same neural CFR, but only
one seat learns while the others are frozen (`--br-vs`). Its gain over
playing the frozen strategy itself is a lower bound on that seat's
exploitability.

Calibration on round 1, seat 4 against the uniform strategy:

| | exact | trained exploiter (10 iterations, 2M-deal match) |
|---|---|---|
| best-response value | +8.011 | +8.005 ± 0.009 |
| uniform baseline | −0.286 | −0.289 ± 0.010 |

The exploiter recovers 99.9 % of the exact exploitability.

## Cost per round (data generation, this PC, 20 threads)

`sk_deep gen-values --count 4000` and `gen-regrets --count 1000`
(`runs/scale`):

| round | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| regret samples / s | 20k | 15k | 14k | 14k | 13k | 12k |
| value samples / s | 78k | 74k | 56k | 63k | 60k | 60k |

Generation now grows only linearly with the round. With external sampling,
a single round-10 traversal took ~9 s. Network training, not data, becomes
the bottleneck, which is what a rented GPU is for.
