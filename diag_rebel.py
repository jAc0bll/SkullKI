import numpy as np, torch, math, glob
from skull_king.training.rebel.networks import RebelPolicyNet
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.training.rebel.public_belief_state import pbs_encoding_size
from skull_king.training.rebel.buffers import PolicyBuffer

n_players = 4
pbs_size = pbs_encoding_size(n_players)
print("ACTION_SPACE_SIZE=%d  log(AS)=%.3f" % (ACTION_SPACE_SIZE, math.log(ACTION_SPACE_SIZE)))
print("pbs_size=%d" % pbs_size)

ckpts = sorted(glob.glob('models/rebel/*_policy.pt'))
net = RebelPolicyNet(n_players).cpu()
if ckpts:
    net.load_state_dict(torch.load(ckpts[-1], map_location='cpu'))
    print("Loaded: %s" % ckpts[-1])
else:
    print("No checkpoint - random weights")
net.eval()

# --- 1. Policy net output sanity check ---
mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
mask[:6] = True
enc = torch.zeros(1, pbs_size)
with torch.no_grad():
    lp = net(enc, mask.unsqueeze(0))[0]
legal = lp[:6]
print("\n[1] Bidding k=6:")
print("  legal  log_probs: " + str([round(x,3) for x in legal.tolist()]))
print("  illegal sample:   " + str([round(x,3) for x in lp[6:9].tolist()]))
h = -(legal.exp() * legal).sum().item()
print("  H(policy)=%.4f  log(6)=%.4f" % (h, math.log(6)))

mask2 = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
mask2[11:16] = True
with torch.no_grad():
    lp2 = net(enc, mask2.unsqueeze(0))[0]
legal2 = lp2[11:16]
print("\n[2] Playing k=5:")
print("  legal  log_probs: " + str([round(x,3) for x in legal2.tolist()]))
h2 = -(legal2.exp() * legal2).sum().item()
print("  H(policy)=%.4f  log(5)=%.4f" % (h2, math.log(5)))

# --- 2. Policy buffer sample analysis ---
import os
buf = PolicyBuffer(500_000, pbs_size, ACTION_SPACE_SIZE)
# Rebuild a tiny buffer by peeking at checkpoint arrays if available
# Instead just load latest checkpoint and check strategy entropy
print("\n[3] If we had the buffer, checking strategy entropy...")
print("  (No direct buffer access - checking via training reconstruction)")

# --- 3. Manually compute what pol_loss should be ---
# If net outputs uniform over k legal actions, cross_entropy = log(k)
# If net outputs uniform over 82 actions (wrong), CE = log(82) ~ 4.41
print("\n[4] Pol_loss decomposition:")
print("  log(ACTION_SPACE_SIZE=82) = %.4f" % math.log(82))
print("  E[log(k)] for typical game ~ %.4f" % np.mean([math.log(k) for k in [2,3,4,5,6,7,8,9,10,11]*4 + [5]*220]))
print("  Observed pol_loss = 4.75")
print("  => Policy is approx uniform over exp(4.75)=%.0f actions (not k legal)" % math.exp(4.75))

# --- 4. Check if strategies in buffer are actually uniform over legal ---
# Reconstruct what strategies look like from subgame solver
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase
from skull_king.training.rebel.public_belief_state import PublicBeliefState
from skull_king.training.rebel.subgame import SubgameSolver, _build_action_mask

print("\n[5] Sample subgame strategy check...")
engine = GameEngine(n_players=4, seed=42)
engine.start()
acting = engine._current_player_index()
pbs = PublicBeliefState.from_engine(engine, acting)
solver = SubgameSolver(value_net=None, device=torch.device('cpu'), n_cfr_iters=2, max_depth=4)
result = solver.solve(engine, pbs, acting)
strat = result['strategy']
mask_np = result['mask']
legal_idx = np.where(mask_np)[0]
print("  Phase: %s  Legal actions: %d" % (engine._phase, len(legal_idx)))
print("  Strategy on legal: " + str([round(strat[a], 4) for a in legal_idx]))
print("  Strategy sum:      %.6f" % strat.sum())
print("  Illegal actions nonzero: %d" % (strat[~mask_np] != 0).sum())
nonzero_illegal = strat[~mask_np]
if nonzero_illegal.any():
    print("  ILLEGAL nonzero values: " + str(nonzero_illegal[nonzero_illegal != 0][:5]))
