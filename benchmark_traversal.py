import time
from skull_king.training.cfr.networks import BiddingAdvNet, PlayingAdvNet
from skull_king.training.cfr.traversal import traverse_split
from skull_king.env.skull_king_env import SkullKingEnv

bid = BiddingAdvNet()
play = PlayingAdvNet()
env = SkullKingEnv(n_players=4)

# warm-up
traverse_split(0, bid, play, env, seed=0, n_players=4)

N = 20
t0 = time.time()
for i in range(N):
    traverse_split(i % 4, bid, play, env, seed=i, n_players=4)
elapsed_ms = (time.time() - t0) * 1000

print(f"{elapsed_ms / N:.0f}ms per traversal (single-threaded, n={N})")
print(f"Projected 2000 traversals / 60 workers: {elapsed_ms / N * 2000 / 60 / 1000:.1f}s")
