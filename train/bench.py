"""Where does an iteration spend its time? Runs a few training iterations of
the given rounds with the real (full-profile) settings and prints the mean
seconds per phase.

  python train/bench.py --rounds 4,10 --iters 3            # data scale auto
  python train/bench.py --rounds 10 --iters 3 --data-scale 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import round_config, to_flags  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default="4,10")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--data-scale", type=int, default=0)
    ap.add_argument("--workdir", type=Path, default=Path("runs/bench"))
    args = ap.parse_args()
    scale = args.data_scale or max(1, min(4, (os.cpu_count() or 1) // 48))

    for r in [int(x) for x in args.rounds.split(",")]:
        cfg, _ = round_config(r, "full")
        cfg["value_games"] *= scale
        cfg["regret_traj"] *= scale
        cfg["policy_keep"] = round(1.0 / scale, 4)
        cfg["iters"] = args.iters
        wd = args.workdir / f"round{r:02d}"
        shutil.rmtree(wd, ignore_errors=True)
        # eval at the last iteration so the average-net training is measured too
        cmd = [sys.executable, str(REPO / "train" / "deep_cfr.py"), "--round", str(r),
               "--workdir", str(wd), "--eval-every", str(args.iters), "--avg-weight-power", "2",
               "--checkpoint-every", "2"] + to_flags(cfg)
        print(f"\n=== round {r} (data scale x{scale}) ===", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        recs = [json.loads(l) for l in (wd / "log.jsonl").read_text().splitlines() if l.strip()]
        phases: dict[str, list[float]] = {}
        for rec in recs:
            for k, v in rec.get("t", {}).items():
                phases.setdefault(k, []).append(v)
        total = sum(rec["iter_seconds"] for rec in recs) / len(recs)
        print(f"{'phase':<18}{'mean s':>9}{'share':>8}   (per iteration, {len(recs)} iterations)")
        for k, vs in phases.items():
            m = sum(vs) / len(recs)
            print(f"{k:<18}{m:>9.1f}{100 * m / total:>7.0f}%")
        print(f"{'TOTAL':<18}{total:>9.1f}")
        print(f"samples: value {recs[-1].get('value_samples')}, regret {recs[-1].get('regret_samples')}, "
              f"policy {recs[-1].get('policy_samples')}")
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()
