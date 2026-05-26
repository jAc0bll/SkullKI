"""End-to-end sanity test of the C++ bindings exposed as the `skullking` module.

Run from the repo root after `cmake --build build`:

    python bindings/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the directory containing the built .pyd to sys.path.
BUILD_PY = Path(__file__).resolve().parent.parent / "build" / "python"
sys.path.insert(0, str(BUILD_PY))

import numpy as np
import skullking as sk

print(f"Loaded skullking from {sk.__file__}")
print(f"N_PLAYERS={sk.N_PLAYERS}  ENC_DIM={sk.ENC_DIM}  N_CARDS={sk.N_CARDS}")

# ---- Walk through a single game with HeuristicAgent everywhere ----
rng = sk.Rng(seed=0xC0FFEE)
s = sk.initial_state(start_player=0)
sk.deal_round(s, rng)

agents = [sk.HeuristicAgent() for _ in range(sk.N_PLAYERS)]

steps = 0
while not sk.is_terminal(s):
    a = agents[s.current_player].select_action(s, rng)
    sk.step(s, a, rng)
    steps += 1

print(f"\nGame finished after {steps} actions")
print(f"Scores: {s.scores}")
print(f"Final round: {s.round_number}, phase: {s.phase}")

# ---- Encoder sanity ----
s2 = sk.initial_state(0)
sk.deal_round(s2, rng)
obs = sk.observe(s2, 0)
enc = sk.encode(obs)

print(f"\nEncoder output:")
print(f"  shape={enc.shape}  dtype={enc.dtype}  min={enc.min():.3f}  max={enc.max():.3f}")
assert enc.shape == (sk.ENC_DIM,), f"expected ({sk.ENC_DIM},), got {enc.shape}"
assert np.isfinite(enc).all()

# ---- Hidden-info privacy: opponent hands not leaked ----
print(f"\nPrivacy check:")
print(f"  perspective 0 sees own hand: {obs.own_hand}")
print(f"  perspective 0 sees hand_sizes: {obs.hand_sizes}")
print(f"  but opponents' actual hand contents are HIDDEN:")
for p in range(sk.N_PLAYERS):
    if p == obs.perspective:
        continue
    assert obs.state.hands[p] == [], f"opponent {p} hand leaked: {obs.state.hands[p]}"
print(f"  [OK] all opponents' hands empty in observation")

# ---- ISMCTS smoke test ----
cfg = sk.ISMCTSConfig()
cfg.num_simulations = 100
mcts = sk.ISMCTSAgent(cfg)
s3 = sk.initial_state(0)
sk.deal_round(s3, rng)
a = mcts.select_action(s3, rng)
print(f"\nISMCTS(100 sims) first move suggestion: {a}")
assert any(a == la for la in sk.legal_actions(s3)), "ISMCTS picked illegal action"

# ---- Batched encoding throughput estimate ----
import time
N = 5000
s4 = sk.initial_state(0)
sk.deal_round(s4, rng)
o4 = sk.observe(s4, 0)
t0 = time.perf_counter()
for _ in range(N):
    _ = sk.encode(o4)
t1 = time.perf_counter()
rate = N / (t1 - t0)
print(f"\nEncoding throughput: {rate:,.0f} obs/sec  ({(t1-t0)*1000/N:.3f}ms each)")

print("\nAll Python bindings checks passed.")
