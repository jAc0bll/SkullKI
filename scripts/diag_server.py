"""Server diagnostic: check C engine, traversal speed, bottleneck."""
import sys
import time

sys.path.insert(0, ".")

print("=== Skull King Server Diagnostic ===\n")

# 1. C extension
try:
    from skull_king._core.skull_king_engine import traverse
    print("[OK]  C engine extension built")
    c_ok = True
except ImportError as e:
    print(f"[!!]  C engine MISSING: {e}")
    print("      Fix: cd skull_king/_core && python setup_engine.py build_ext --inplace")
    c_ok = False

# 2. Worker init + C engine activation
from skull_king.training.cfr.networks import AdvantageNet
from skull_king.training.cfr.traversal import worker_init, worker_task
import skull_king.training.cfr.traversal as trav

net = AdvantageNet(hidden=(512, 512))
worker_init(net.state_dict(), n_players=4, heuristic_frac=0.4)

if trav._C_ENGINE is not None:
    print("[OK]  C engine active in worker")
else:
    print("[!!]  C engine NOT active in worker (using Python fallback)")

# 3. Single-worker traversal speed
N = 20
t0 = time.perf_counter()
for i in range(N):
    worker_task((i % 4, i * 99 + 1, 4))
ms = (time.perf_counter() - t0) * 1000 / N
backend = "C" if trav._C_ENGINE else "Python"
print(f"\n--- Traversal speed ({backend} backend) ---")
print(f"  {ms:.1f} ms/traversal")

# 4. Estimate collection time for current config
import yaml
try:
    with open("cfr_config_overnight_pc.yaml") as f:
        cfg = yaml.safe_load(f)
    tpp = cfg.get("traversals_per_player", 500)
    np_ = cfg.get("n_players", 4)
    nw  = cfg.get("num_workers", 8)
    total_trav = tpp * np_
    collection_s = total_trav * ms / nw / 1000
    print(f"\n--- Config: {tpp} traversals/player x {np_} players = {total_trav} total ---")
    print(f"  num_workers:       {nw}")
    print(f"  expected collection: {collection_s:.1f}s/iter")
    print(f"  + ~6s net training  => ~{collection_s + 6:.0f}s/iter total")
    print(f"  2500 iters => ~{(collection_s + 6) * 2500 / 3600:.1f}h")
except FileNotFoundError:
    print(f"\n  (config file not found, manual estimate:)")
    for workers in [8, 48]:
        s = 2000 * ms / workers / 1000
        print(f"  {workers} workers, 2000 traversals: {s:.1f}s collection + 6s train")

print("\n=== Done ===")
