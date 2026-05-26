"""Generate self-play training data using ISMCTS as the teacher.

For each MCTS decision we record:
  features   : encoded observation               (ENC_DIM,)   float32
  policy     : normalized ISMCTS visit counts    (ACTION_DIM,) float32
  legal_mask : which actions are legal here      (ACTION_DIM,) bool
  value      : MCTS root value estimate / 200    scalar       float32

Usage:
    python train/data_gen.py --games 200 --sims 400 --out train/data/selfplay.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_PY  = REPO_ROOT / "build" / "python"
sys.path.insert(0, str(BUILD_PY))

import skullking as sk  # noqa: E402


def run_one_game(seed: int, num_sims: int) -> tuple[np.ndarray, ...]:
    """Plays one game with ISMCTS for all 4 seats and returns per-move training
    targets. The value target is the *actual* per-player round score delta —
    an unbiased Monte-Carlo estimate — not the MCTS's internal value estimate.
    Using MCTS estimates as value targets caused systematic value-head bias
    that poisoned PUCT search at higher sim counts (see project memory)."""
    rng = sk.Rng(seed=seed)
    s = sk.initial_state(start_player=0)
    sk.deal_round(s, rng)

    cfg = sk.ISMCTSConfig()
    cfg.num_simulations = num_sims
    mcts = sk.ISMCTSAgent(cfg)

    # Per-sample buffers; value targets are filled in retrospectively after
    # each round completes.
    feats_buf, policy_buf, legal_buf, value_buf = [], [], [], []
    hands_buf, perspective_buf = [], []   # ground-truth opponent hands for the belief net
    sample_round: list[int] = []

    # Track scores at the start of each round (round_number -> [scores]).
    round_start_scores: dict[int, list[int]] = {int(s.round_number): list(s.scores)}

    while not sk.is_terminal(s):
        round_now = int(s.round_number)
        if round_now not in round_start_scores:
            round_start_scores[round_now] = list(s.scores)

        obs    = sk.observe(s, s.current_player)
        feats  = sk.encode(obs)
        result = mcts.select_action_with_targets(s, rng)

        target = np.zeros(sk.ACTION_DIM, dtype=np.float32)
        legal  = np.zeros(sk.ACTION_DIM, dtype=bool)
        total  = sum(result.visits)
        if total > 0:
            for a, v in zip(result.root_actions, result.visits):
                idx = sk.action_to_index(a)
                target[idx] = v / total
                legal[idx]  = True
        else:
            for a in result.root_actions:
                idx = sk.action_to_index(a)
                target[idx] = 1.0 / len(result.root_actions)
                legal[idx]  = True

        # Ground-truth hands snapshot for the belief-net.
        cur_perspective = int(s.current_player)
        true_hands = np.zeros((sk.N_PLAYERS, sk.N_CARDS), dtype=np.float32)
        for p in range(sk.N_PLAYERS):
            for c in s.hands[p]:
                true_hands[p, c] = 1.0

        feats_buf.append(feats)
        policy_buf.append(target)
        legal_buf.append(legal)
        hands_buf.append(true_hands)
        perspective_buf.append(np.int8(cur_perspective))
        sample_round.append(round_now)
        # Placeholder; will overwrite after we know the round outcome.
        value_buf.append(np.zeros(sk.N_PLAYERS, dtype=np.float32))

        sk.step(s, result.action, rng)

    # End-of-game: compute round deltas. The scores at the start of round r+1
    # equal the scores at the END of round r. For the last completed round we
    # use s.scores (since the game is terminal).
    final_scores = list(s.scores)
    rounds_played = sorted(round_start_scores.keys())
    for r in rounds_played:
        end_scores = round_start_scores.get(r + 1, final_scores)
        delta = np.array([(end_scores[p] - round_start_scores[r][p]) for p in range(sk.N_PLAYERS)],
                         dtype=np.float32) / 200.0
        # Fill in all samples taken during round r.
        for i, sr in enumerate(sample_round):
            if sr == r:
                value_buf[i] = delta

    return (
        np.stack(feats_buf),
        np.stack(policy_buf),
        np.stack(legal_buf),
        np.stack(value_buf),
        np.stack(hands_buf),
        np.array(perspective_buf, dtype=np.int8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",   type=int, default=200)
    ap.add_argument("--sims",    type=int, default=400, help="ISMCTS simulations per move")
    ap.add_argument("--seed",    type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--out",     type=str, default="train/data/selfplay.npz")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.games))
    print(f"Generating {args.games} games with ISMCTS({args.sims} sims) "
          f"using {args.workers} worker(s) -> {out_path}")
    t0 = time.perf_counter()

    if args.workers == 1:
        results = [run_one_game(s, args.sims) for s in seeds]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.starmap(run_one_game, [(s, args.sims) for s in seeds])

    feats       = np.concatenate([r[0] for r in results], axis=0)
    policy      = np.concatenate([r[1] for r in results], axis=0)
    legal       = np.concatenate([r[2] for r in results], axis=0)
    value       = np.concatenate([r[3] for r in results], axis=0)
    hands       = np.concatenate([r[4] for r in results], axis=0)
    perspective = np.concatenate([r[5] for r in results], axis=0)

    elapsed = time.perf_counter() - t0
    n = feats.shape[0]
    print(f"\nGenerated {n:,} samples in {elapsed:.1f}s "
          f"({n/elapsed:,.0f} samples/sec, {args.games/elapsed:.1f} games/sec)")
    print(f"  features    {feats.shape}  {feats.dtype}")
    print(f"  policy      {policy.shape}  {policy.dtype}")
    print(f"  legal       {legal.shape}  {legal.dtype}")
    print(f"  value       {value.shape}  per-player mean={value.mean(axis=0)}  std={value.std(axis=0)}")
    print(f"  hands       {hands.shape}  mean cards-per-player={hands.sum(axis=(0,2)).mean()/n:.2f}")
    print(f"  perspective {perspective.shape}  distribution={[int((perspective==i).sum()) for i in range(sk.N_PLAYERS)]}")

    np.savez_compressed(
        out_path,
        features=feats, policy=policy, legal=legal, value=value,
        hands=hands, perspective=perspective,
    )
    print(f"\nSaved {out_path.stat().st_size/1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
