"""Generate self-play training data using NeuralMCTSAgent (hybrid mode by default).

Same output schema as data_gen.py (features / policy / legal / value / hands /
perspective). Designed to be the workhorse for AlphaZero iterations on vast.ai.

Usage:
    python train/selfplay.py --model train/checkpoints/bc_v3_mc.scripted.pt \
                             --games 200 --sims 100 --workers 4 \
                             --out train/data/selfplay_iter1.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import site
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def _add_torch_dll_dir() -> None:
    """On Windows, Python ≥3.8 needs add_dll_directory for the LibTorch DLLs.
    On Linux/macOS the dynamic linker uses RPATH set by pybind11/torch."""
    if platform.system() != "Windows":
        return
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        torch_lib = Path(sp) / "torch" / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            return


# ---- Worker globals (lazy-initialised per process) ----
_sk = None
_agent = None
_belief = None
_evaluator = None
_use_belief = False


def _init_worker(model_path: str, num_sims: int, belief_path: str | None,
                 use_nn_value: bool, device: str) -> None:
    _add_torch_dll_dir()
    sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
    import skullking as sk

    global _sk, _agent, _evaluator, _belief, _use_belief
    _sk = sk
    _evaluator = sk.TorchModelEvaluator(model_path, device)
    cfg = sk.NeuralMCTSConfig()
    cfg.mcts.num_simulations = num_sims
    cfg.use_nn_value         = use_nn_value
    if belief_path:
        _belief = sk.BeliefEvaluator(belief_path, device)
        cfg.set_belief(_belief)
        _use_belief = True
    _agent = sk.NeuralMCTSAgent(_evaluator, cfg)


def _run_one(seed: int):
    sk = _sk
    rng = sk.Rng(seed=seed)
    s = sk.initial_state(start_player=0)
    sk.deal_round(s, rng)

    feats_buf, policy_buf, legal_buf, value_buf = [], [], [], []
    hands_buf, perspective_buf = [], []
    sample_round: list[int] = []

    round_start_scores: dict[int, list[int]] = {int(s.round_number): list(s.scores)}

    while not sk.is_terminal(s):
        round_now = int(s.round_number)
        if round_now not in round_start_scores:
            round_start_scores[round_now] = list(s.scores)

        obs    = sk.observe(s, s.current_player)
        feats  = sk.encode(obs)
        result = _agent.select_action_with_targets(s, rng)

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
        value_buf.append(np.zeros(sk.N_PLAYERS, dtype=np.float32))

        sk.step(s, result.action, rng)

    final_scores = list(s.scores)
    rounds = sorted(round_start_scores.keys())
    for r in rounds:
        end_scores = round_start_scores.get(r + 1, final_scores)
        delta = np.array([end_scores[p] - round_start_scores[r][p]
                          for p in range(sk.N_PLAYERS)], dtype=np.float32) / 200.0
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
    ap.add_argument("--model",   required=True, help="TorchScript PolicyValueNet checkpoint")
    ap.add_argument("--belief",  default="",     help="Optional TorchScript BeliefNet checkpoint")
    ap.add_argument("--games",   type=int,   default=200)
    ap.add_argument("--sims",    type=int,   default=100)
    ap.add_argument("--seed",    type=int,   default=1)
    ap.add_argument("--workers", type=int,   default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--use-nn-value", action="store_true",
                    help="Use NN value head as leaf evaluation (otherwise MC rollout)")
    ap.add_argument("--device",  default="cpu", help="cpu or cuda (single inference device)")
    ap.add_argument("--out",     default="train/data/selfplay.npz")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.games))
    print(f"Selfplay: {args.games} games × {args.sims} sims, {args.workers} worker(s)")
    print(f"  model:  {args.model}")
    print(f"  belief: {args.belief or '<none>'}")
    print(f"  value:  {'NN' if args.use_nn_value else 'MC-rollout (hybrid)'}")
    print(f"  device: {args.device}")
    print(f"  out:    {out_path}")
    t0 = time.perf_counter()

    init_args = (str(REPO_ROOT / args.model), args.sims,
                 str(REPO_ROOT / args.belief) if args.belief else None,
                 args.use_nn_value, args.device)
    if args.workers == 1:
        _init_worker(*init_args)
        results = [_run_one(s) for s in seeds]
    else:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
            results = pool.map(_run_one, seeds)

    feats       = np.concatenate([r[0] for r in results], axis=0)
    policy      = np.concatenate([r[1] for r in results], axis=0)
    legal       = np.concatenate([r[2] for r in results], axis=0)
    value       = np.concatenate([r[3] for r in results], axis=0)
    hands       = np.concatenate([r[4] for r in results], axis=0)
    perspective = np.concatenate([r[5] for r in results], axis=0)

    elapsed = time.perf_counter() - t0
    n = feats.shape[0]
    print(f"\nGenerated {n:,} samples in {elapsed:.1f}s "
          f"({n/elapsed:,.0f} samples/sec, {args.games/elapsed:.2f} games/sec)")
    print(f"  features    {feats.shape}  {feats.dtype}")
    print(f"  policy      {policy.shape}  {policy.dtype}")
    print(f"  value       {value.shape}  mean={value.mean(axis=0)}  std={value.std(axis=0)}")

    np.savez_compressed(out_path, features=feats, policy=policy, legal=legal,
                        value=value, hands=hands, perspective=perspective)
    print(f"\nSaved {out_path.stat().st_size/1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
