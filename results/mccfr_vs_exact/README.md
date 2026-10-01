# Sampling (MCCFR) vs. exact (DCFR) — round 1, 4 players

Question: how close does sampled CFR get to the exact solution, and what
does it cost in larger rounds? Round 1 is the calibration case because its
exact NashConv is computable.

## Runs (i7-12700K, 20 threads, solve time excludes evaluation)

```
sk_solve_round --method mccfr --round 1 --iters 200 --eval-every 20 --deals-per-batch 65536
sk_solve_round --method dcfr  --round 1 --iters 60  --eval-every 10
sk_bench_traversal --max-round 10 --probes 20000 --measure-up-to 4
```

| method | solve time | NashConv | gains seat 1 / 2 / 3 / 4 | infosets seen |
|---|---|---|---|---|
| MCCFR, 5.2M traversals | 0.5 s | 0.316 | 0.010 / 0.013 / 0.086 / 0.208 | 117k |
| MCCFR, 26M traversals | 6.4 s | 0.185 | 0.001 / 0.002 / 0.007 / 0.175 | 367k |
| MCCFR, 52M traversals | 19 s | 0.153 | 0.0003 / 0.0006 / 0.003 / 0.149 | 564k of 3.35M |
| DCFR (exact), 10 iterations | 26 s | 0.334 | | all |
| DCFR (exact), 30 iterations | 70 s | 0.016 | | all |
| DCFR (exact), 60 iterations | 138 s | 0.002 | | all |

### Finding 1: sampling learns common situations fast, rare ones never

MCCFR is very fast for seats 1–3: after 19 s no seat 1–3 deviation gains
more than 0.003 points. Almost the whole remaining NashConv is **seat 4**.
Seat 4 has ~3.3M Tigress-declaration infosets (16 bid profiles × every
ordered set of 3 cards before it). Each one is so rare that only 17 % of
all infosets were ever sampled; the unseen ones keep a uniform
strategy, and the exact best response exploits exactly those.

A tabular method has to see a situation to learn it. "With bid 1, Skull
King already in the trick → declare Escape" is one rule, but the table
must learn it separately for each of the ~200k card combinations. That is
the argument for **generalisation** (neural networks or abstraction):
learn the rule once and apply it to unseen combinations.

### Finding 2: an external-sampling traversal grows ~6× per round

External sampling explores every action of the traversing player. Knuth
estimate of nodes per traversal and the extrapolated time on the whole
machine (`traversal_cost.txt`):

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| nodes / traversal | 16 | 52 | 193 | 801 | 3.7k | 20k | 111k | 671k | 4.3M | 31M |
| time / traversal | 0.2 µs | 7 µs | 43 µs | 0.23 ms | ~1 ms | ~6 ms | ~32 ms | ~0.2 s | ~1.2 s | ~9 s |

Rounds 5+ are extrapolated from the round-4 throughput. Round 10 would need
~9 s per traversal on all 20 threads, and millions of traversals are needed.

## Consequences for rounds 2–10

1. **Do not fully expand the traverser's own actions in big rounds.**
   Sample them too, as in ESCHER (McAleer et al., ICLR 2023: samples traverser
   actions from a fixed policy and avoids importance-sampling variance
   with a learned value function), or expand fully only near the root.
2. **Generalise instead of tabulating**, using neural regret/policy
   networks. Round 1 (exact) and round 2 (sampled, large budget) serve as
   the yardstick: a network method is accepted only if its NashConv on
   round 1 stays close to the exact solution.
3. **Exact solving at decision time**: for a concrete spot late in a
   round the remaining tree is small, so full-width CFR over the remaining
   tricks (Phase 3) can solve it almost exactly.
