"""Multi-GPU wrapper around the C++ selfplay_az tool.

Spawns N selfplay_az subprocesses (one per GPU), each generating its share
of games on cuda:i. After all complete, loads the per-GPU SKAZ files and
writes one combined .npz that drops into train.py / train_belief.py.

Usage:
    python train/selfplay_az_multi.py \
        --binary build/tools/selfplay_az \
        --model train/checkpoints/bc_v3_mc.scripted.pt \
        --games 5000 --threads-per-gpu 32 --sims 100 \
        --max-batch 64 --max-wait-us 1000 \
        --gpus 4 \
        --out train/data/az_multi.npz

The .npz output schema is identical to data_gen.py / selfplay.py so the
existing training scripts work unchanged.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from load_skaz import load as load_skaz  # noqa: E402


def default_binary() -> Path:
    name = "selfplay_az.exe" if platform.system() == "Windows" else "selfplay_az"
    return REPO_ROOT / "build" / "tools" / name


def split_games(total: int, gpus: int) -> list[int]:
    base, rem = divmod(total, gpus)
    return [base + (1 if i < rem else 0) for i in range(gpus)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",  default=str(default_binary()), help="Path to selfplay_az exe")
    ap.add_argument("--model",   required=True)
    ap.add_argument("--games",   type=int, required=True, help="Total games across all GPUs")
    ap.add_argument("--threads-per-gpu", type=int, default=32)
    ap.add_argument("--sims",    type=int, default=100)
    ap.add_argument("--max-batch",   type=int, default=64)
    ap.add_argument("--max-wait-us", type=int, default=0,
                    help="Scheduler batch-fill wait window. Default 0 = flush "
                         "immediately. Empirically best for our small (1.9M param) "
                         "model where GPU forward is fast and waiting wastes time.")
    ap.add_argument("--gpus",    type=int, default=1, help="Number of GPUs to use (cuda:0 .. cuda:N-1)")
    ap.add_argument("--seed",    type=int, default=1)
    ap.add_argument("--out",     required=True, help="Path to combined .npz output")
    ap.add_argument("--keep-bin", action="store_true", help="Don't delete the per-GPU SKAZ files")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise SystemExit(f"selfplay_az binary not found at {binary}")

    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_npz.parent / (out_npz.stem + "_shards")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    per_gpu = split_games(args.games, args.gpus)
    print(f"Dispatching {args.games} games across {args.gpus} GPU(s): {per_gpu}")
    print(f"  threads/gpu={args.threads_per_gpu}  sims={args.sims}  "
          f"max-batch={args.max_batch}  max-wait-us={args.max_wait_us}")

    t0 = time.perf_counter()
    procs: list[tuple[subprocess.Popen, Path]] = []
    for i in range(args.gpus):
        shard = tmp_dir / f"shard_{i}.bin"
        cmd = [
            str(binary),
            "--model",       args.model,
            "--device",      f"cuda:{i}",
            "--games",       str(per_gpu[i]),
            "--threads",     str(args.threads_per_gpu),
            "--sims",        str(args.sims),
            "--max-batch",   str(args.max_batch),
            "--max-wait-us", str(args.max_wait_us),
            "--seed",        str(args.seed + i * 100003),
            "--out",         str(shard),
        ]
        print(f"  [gpu {i}] {' '.join(cmd)}")
        procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT)), shard))

    # Wait for all
    failed = False
    for p, _ in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  [warn] shard exited with code {rc}")
            failed = True
    if failed:
        raise SystemExit("at least one shard failed")

    # Merge.
    print("\nMerging shards…")
    merged: dict[str, list[np.ndarray]] = {}
    total_samples = 0
    for _, shard_path in procs:
        d = load_skaz(shard_path)
        for k, v in d.items():
            merged.setdefault(k, []).append(v)
        total_samples += d["features"].shape[0]
        print(f"  {shard_path.name}: {d['features'].shape[0]:,} samples")

    final = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    np.savez_compressed(out_npz, **final)
    sz_mb = out_npz.stat().st_size / 1e6

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s ({args.games/elapsed:.2f} games/sec total)")
    print(f"  total samples: {total_samples:,}")
    print(f"  wrote {sz_mb:.1f} MB -> {out_npz}")

    if not args.keep_bin:
        shutil.rmtree(tmp_dir)
        print(f"  removed shard dir {tmp_dir}")


if __name__ == "__main__":
    main()
