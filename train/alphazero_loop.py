"""AlphaZero-style iteration orchestrator.

Each iteration:
    1. Selfplay with the current best model (Hybrid NeuralMCTS: NN-prior + MC-rollout).
    2. Train a candidate model on the new selfplay data, warm-starting from current best.
    3. Export candidate to TorchScript.
    4. Eval gate: candidate must beat current best with >= --gate winrate on N games.
       Pass → promote candidate to new best. Fail → keep current best, throw away
       candidate (or keep for inspection).

Output layout under --workdir:
    workdir/
      best.pt                    PyTorch checkpoint of current best
      best.scripted.pt           TorchScript export of current best
      iter_<n>/
        selfplay.npz
        candidate.pt
        candidate.scripted.pt
        eval.log
      log.jsonl                  one line per iteration with stats

Usage:
    python train/alphazero_loop.py \
        --workdir runs/az1 \
        --init train/checkpoints/bc_v3_mc.pt \
        --iterations 5 \
        --selfplay-games 1000 --selfplay-sims 100 \
        --train-epochs 10 --train-batch 2048 --train-lr 1e-3 \
        --gate-games 80 --gate-winrate 0.55
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

REPO_ROOT = Path(__file__).resolve().parent.parent


def add_torch_dll_dir() -> None:
    if platform.system() != "Windows":
        return
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        lib = Path(sp) / "torch" / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            return


def run(cmd: list[str], **kw) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


@dataclass
class IterStats:
    iteration: int
    selfplay_seconds: float
    train_seconds: float
    eval_seconds: float
    candidate_winrate: float
    candidate_avg_score: float
    best_avg_score: float
    promoted: bool


def selfplay_step(workdir: Path, iter_dir: Path, best_scripted: Path,
                  games: int, sims: int, workers: int, device: str,
                  *, impl: str = "python",
                  gpus: int = 1, threads_per_gpu: int = 32,
                  max_batch: int = 64, max_wait_us: int = 1000) -> Path:
    """Run one selfplay phase. `impl` picks between:
       - 'python' : multi-process Python workers (no GPU batching)
       - 'az'     : multi-threaded C++ selfplay_az with batched GPU inference
    """
    out = iter_dir / "selfplay.npz"
    seed = int(time.time()) & 0xFFFFFF

    if impl == "az":
        cmd = [sys.executable, str(REPO_ROOT / "train" / "selfplay_az_multi.py"),
               "--model",           str(best_scripted),
               "--games",           str(games),
               "--sims",            str(sims),
               "--gpus",            str(gpus),
               "--threads-per-gpu", str(threads_per_gpu),
               "--max-batch",       str(max_batch),
               "--max-wait-us",     str(max_wait_us),
               "--seed",            str(seed),
               "--out",             str(out.relative_to(REPO_ROOT))]
    else:  # 'python'
        cmd = [sys.executable, str(REPO_ROOT / "train" / "selfplay.py"),
               "--model",   str(best_scripted),
               "--games",   str(games),
               "--sims",    str(sims),
               "--workers", str(workers),
               "--device",  device,
               "--seed",    str(seed),
               "--out",     str(out.relative_to(REPO_ROOT))]
    run(cmd, cwd=REPO_ROOT)
    return out


def train_step(iter_dir: Path, data: Path, init_ckpt: Path, epochs: int,
               batch_size: int, lr: float, hidden: int, device: str) -> Path:
    out = iter_dir / "candidate.pt"
    # Warm-start: we copy the init_ckpt to a "warmstart.pt" and let train.py
    # randomly initialise on top — current train.py doesn't support
    # warm-start, so for now we just re-train from scratch each iteration.
    # TODO: add --init flag to train.py for true warm-start.
    cmd = [sys.executable, str(REPO_ROOT / "train" / "train.py"),
           "--data",       str(data.relative_to(REPO_ROOT)),
           "--out",        str(out.relative_to(REPO_ROOT)),
           "--epochs",     str(epochs),
           "--batch-size", str(batch_size),
           "--lr",         str(lr),
           "--hidden",     str(hidden),
           "--device",     device]
    run(cmd, cwd=REPO_ROOT)
    return out


def export_step(ckpt: Path, scripted_out: Path) -> Path:
    cmd = [sys.executable, str(REPO_ROOT / "train" / "export.py"),
           "--ckpt", str(ckpt.relative_to(REPO_ROOT)),
           "--out",  str(scripted_out.relative_to(REPO_ROOT))]
    run(cmd, cwd=REPO_ROOT)
    return scripted_out


def eval_gate(candidate_scripted: Path, best_scripted: Path, games: int,
              sims: int, device: str) -> tuple[float, float, float]:
    """Plays candidate vs best in seat-rotated tournament.
    Returns (candidate_winrate, candidate_avg_score, best_avg_score)."""
    import numpy as np
    add_torch_dll_dir()
    sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
    import skullking as sk

    cand_eval = sk.TorchModelEvaluator(str(candidate_scripted), device)
    best_eval = sk.TorchModelEvaluator(str(best_scripted),       device)

    def make(eval_obj):
        cfg = sk.NeuralMCTSConfig()
        cfg.mcts.num_simulations = sims
        cfg.use_nn_value = False
        return sk.NeuralMCTSAgent(eval_obj, cfg)

    cand = make(cand_eval)
    best = make(best_eval)

    # Seat-rotated: candidate plays each seat in 1/N_PLAYERS of games.
    rotations = sk.N_PLAYERS
    per_rot = max(1, games // rotations)
    total = per_rot * rotations

    rng = sk.Rng(seed=42)
    cand_score = best_score = 0
    cand_wins = 0
    cand_seats = best_seats = 0

    pbar = tqdm(total=total, desc='eval-gate', unit='game', dynamic_ncols=True) if HAS_TQDM else None
    for rot in range(rotations):
        cand_seat = rot
        lineup = [best] * sk.N_PLAYERS
        lineup[cand_seat] = cand
        for _ in range(per_rot):
            state = sk.initial_state(0)
            sk.deal_round(state, rng)
            while not sk.is_terminal(state):
                a = lineup[state.current_player].select_action(state, rng)
                sk.step(state, a, rng)
            scores = state.scores
            winner = int(np.argmax(scores))
            if winner == cand_seat:
                cand_wins += 1
            for p in range(sk.N_PLAYERS):
                if p == cand_seat:
                    cand_score += scores[p]; cand_seats += 1
                else:
                    best_score += scores[p]; best_seats += 1
            if pbar is not None:
                wr = cand_wins / max(1, cand_seats)
                pbar.set_postfix(cand_wr=f"{wr*100:5.1f}%")
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    winrate     = cand_wins / total
    cand_avg    = cand_score / cand_seats
    best_avg    = best_score / best_seats
    return winrate, cand_avg, best_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir",        required=True)
    ap.add_argument("--init",           required=True, help="Initial PyTorch checkpoint")
    ap.add_argument("--iterations",     type=int,   default=3)
    ap.add_argument("--selfplay-games", type=int,   default=500)
    ap.add_argument("--selfplay-sims",  type=int,   default=100)
    ap.add_argument("--workers",        type=int,   default=4)
    ap.add_argument("--train-epochs",   type=int,   default=10)
    ap.add_argument("--train-batch",    type=int,   default=2048)
    ap.add_argument("--train-lr",       type=float, default=1e-3)
    ap.add_argument("--train-hidden",   type=int,   default=1024)
    ap.add_argument("--gate-games",     type=int,   default=80)
    ap.add_argument("--gate-sims",      type=int,   default=100)
    ap.add_argument("--gate-winrate",   type=float, default=0.55)
    ap.add_argument("--device",         default="cuda" if Path("/proc/driver/nvidia").exists() else "cpu",
                    help="Device for training and eval-gate")
    ap.add_argument("--selfplay-device", default="cpu",
                    help="Device for selfplay. Default cpu — each worker creates its own CUDA "
                         "context (~500 MB VRAM each), so cuda + many workers OOMs the GPU. "
                         "At our model size single-sample inference is not faster on GPU anyway.")
    # Phase 5C — multi-threaded batched-GPU selfplay path.
    ap.add_argument("--selfplay-impl",  default="python", choices=["python", "az"],
                    help="'python' = old multi-process selfplay.py path (CPU workers). "
                         "'az' = new C++ selfplay_az with batched GPU inference (Phase 5C).")
    ap.add_argument("--az-gpus",            type=int, default=1,
                    help="Number of GPUs to use for selfplay-impl=az")
    ap.add_argument("--az-threads-per-gpu", type=int, default=32,
                    help="MCTS threads per GPU for selfplay-impl=az")
    ap.add_argument("--az-max-batch",       type=int, default=64)
    ap.add_argument("--az-max-wait-us",     type=int, default=1000)
    args = ap.parse_args()

    # Worker sanity check — too many workers thrash RAM and L3 cache without helping.
    # Each worker holds its own model copy (~30 MB) plus tree state (~MBs).
    import os as _os
    cpu_count = _os.cpu_count() or 1
    if args.workers > cpu_count:
        print(f"[warn] --workers {args.workers} > os.cpu_count() {cpu_count}; capping to {cpu_count}")
        args.workers = cpu_count
    if args.workers > 32 and args.selfplay_device == "cuda":
        print(f"[warn] --workers {args.workers} with --selfplay-device cuda will likely OOM the GPU "
              f"(each worker creates its own ~500 MB CUDA context). Consider --selfplay-device cpu.")

    workdir = (REPO_ROOT / args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "log.jsonl"

    # Initialise "best" from --init.
    best_ckpt      = workdir / "best.pt"
    best_scripted  = workdir / "best.scripted.pt"
    if not best_ckpt.exists():
        shutil.copy(args.init, best_ckpt)
        export_step(best_ckpt, best_scripted)

    overall_t0 = time.perf_counter()
    for it in range(1, args.iterations + 1):
        iter_dir = workdir / f"iter_{it:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        elapsed_h = (time.perf_counter() - overall_t0) / 3600
        print(f"\n========== Iteration {it}/{args.iterations}  "
              f"(wallclock {elapsed_h:.2f} h) ==========")

        t0 = time.perf_counter()
        sp_path = selfplay_step(workdir, iter_dir, best_scripted,
                                args.selfplay_games, args.selfplay_sims,
                                args.workers, args.selfplay_device,
                                impl=args.selfplay_impl,
                                gpus=args.az_gpus,
                                threads_per_gpu=args.az_threads_per_gpu,
                                max_batch=args.az_max_batch,
                                max_wait_us=args.az_max_wait_us)
        t_sp = time.perf_counter() - t0

        t0 = time.perf_counter()
        cand_ckpt = train_step(iter_dir, sp_path, best_ckpt,
                               args.train_epochs, args.train_batch, args.train_lr,
                               args.train_hidden, args.device)
        cand_scripted = export_step(cand_ckpt, iter_dir / "candidate.scripted.pt")
        t_tr = time.perf_counter() - t0

        t0 = time.perf_counter()
        winrate, cand_avg, best_avg = eval_gate(cand_scripted, best_scripted,
                                                args.gate_games, args.gate_sims, args.device)
        t_ev = time.perf_counter() - t0

        promoted = winrate >= args.gate_winrate
        if promoted:
            shutil.copy(cand_ckpt,     best_ckpt)
            shutil.copy(cand_scripted, best_scripted)
            print(f"\n[PROMOTE] new best (winrate {winrate*100:.1f}% ≥ {args.gate_winrate*100:.0f}%)")
        else:
            print(f"\n[REJECT]  winrate {winrate*100:.1f}% < {args.gate_winrate*100:.0f}% — keep previous best")

        stats = IterStats(
            iteration=it,
            selfplay_seconds=t_sp,
            train_seconds=t_tr,
            eval_seconds=t_ev,
            candidate_winrate=winrate,
            candidate_avg_score=cand_avg,
            best_avg_score=best_avg,
            promoted=promoted,
        )
        with log_path.open("a") as f:
            f.write(json.dumps(asdict(stats)) + "\n")
        print(json.dumps(asdict(stats), indent=2))

    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
