"""Train every round of Skull King with neural CFR, then measure each round's
exploitability with trained exploiters. Safe to restart at any time: finished
steps are skipped (DONE markers), interrupted ones resume from checkpoints.

  python train/run_all.py --workdir runs/all                 # rounds 1-10, full settings
  python train/run_all.py --workdir runs/smoke --profile smoke --rounds 1-2

Layout:
  <workdir>/roundNN/train/               deep_cfr.py run (avg.bin = the strategy)
  <workdir>/roundNN/exploit_seatK/       exploiter for seat K against it
  <workdir>/summary.json                 one entry per finished round
  <workdir>/status.json                  what is running right now
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEEP = REPO / "train" / "deep_cfr.py"


def round_config(r: int, profile: str) -> tuple[dict, dict]:
    """(training args, exploiter args) for round r."""
    if profile == "smoke":
        train = dict(iters=2, value_games=500, regret_traj=300, buffer=200_000, hidden=64,
                     layers=2, regret_steps=50, value_steps=50, policy_steps=50, batch=256)
        exploit = dict(iters=1, value_games=500, regret_traj=300, buffer=200_000, hidden=64,
                       layers=2, regret_steps=50, value_steps=50, batch=256, match_deals=20_000)
        return train, exploit
    if r == 1:
        train = dict(iters=30, value_games=50_000, regret_traj=20_000, buffer=2_000_000,
                     hidden=256, layers=3, regret_steps=6000, value_steps=1500,
                     policy_steps=8000, batch=2048)
    elif r <= 3:
        train = dict(iters=40, value_games=50_000, regret_traj=20_000, buffer=3_000_000,
                     hidden=256, layers=3, regret_steps=6000, value_steps=2000,
                     policy_steps=10_000, batch=4096)
    else:
        train = dict(iters=60, value_games=30_000, regret_traj=20_000, buffer=6_000_000,
                     hidden=512, layers=4, regret_steps=8000, value_steps=3000,
                     policy_steps=16_000, batch=8192)
    exploit = {k: v for k, v in train.items() if k not in ("policy_steps",)}
    exploit.update(iters=15 if r <= 3 else 20, match_deals=1_000_000)
    return train, exploit


def to_flags(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg.items():
        out += ["--" + k.replace("_", "-"), str(v)]
    return out


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def run_step(cmd: list[str], logfile: Path) -> None:
    """Run a child, mirror its output to the console and to logfile."""
    with open(logfile, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        if proc.wait() != 0:
            raise RuntimeError(f"step failed (exit {proc.returncode}): {' '.join(cmd)}")


def parse_rounds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


MATCH_RE = re.compile(r"utility ([+-][0-9.]+) \+- ([0-9.]+)")


def exploit_gain(ed: Path) -> dict | None:
    """Gain of the exploiter over the frozen strategy, from its last log entry."""
    log = ed / "log.jsonl"
    if not log.exists():
        return None
    last = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    last = [r for r in last if "br_match" in r]
    if not last:
        return None
    br = MATCH_RE.search(last[-1]["br_match"])
    base = MATCH_RE.search(last[-1]["base_match"])
    gain = float(br.group(1)) - float(base.group(1))
    err = (float(br.group(2)) ** 2 + float(base.group(2)) ** 2) ** 0.5
    return {"gain": round(gain, 4), "stderr": round(err, 4),
            "exploiter_utility": float(br.group(1)), "baseline_utility": float(base.group(1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("runs/all"))
    ap.add_argument("--rounds", default="1-10")
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--exploit-seats", default="1,4", help="seats (1-4) to attack per round")
    args = ap.parse_args()

    wd = args.workdir.resolve()
    wd.mkdir(parents=True, exist_ok=True)
    logfile = wd / "run.log"
    summary_path = wd / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    seats = [int(s) - 1 for s in args.exploit_seats.split(",")]
    rounds = parse_rounds(args.rounds)
    py = sys.executable

    def status(msg: str, **extra) -> None:
        write_json(wd / "status.json", {"step": msg, "rounds": rounds, "updated": time.time(),
                                        "pid": os.getpid(), **extra})
        print(f"\n=== {msg} ===", flush=True)

    for r in rounds:
        train_cfg, exploit_cfg = round_config(r, args.profile)
        rd = wd / f"round{r:02d}"
        td = rd / "train"
        if not (td / "DONE").exists():
            status(f"round {r}: training", round=r, phase="train", dir=str(td))
            eval_every = 5 if r == 1 else train_cfg["iters"]   # exact eval exists only for round 1
            run_step([py, str(DEEP), "--round", str(r), "--workdir", str(td), "--resume",
                      "--avg-weight-power", "2", "--eval-every", str(eval_every),
                      "--label", f"round {r} training"] + to_flags(train_cfg), logfile)

        result = summary.get(str(r), {"round": r})
        for seat in seats:
            ed = rd / f"exploit_seat{seat + 1}"
            if not (ed / "DONE").exists():
                status(f"round {r}: exploiter seat {seat + 1}", round=r, phase="exploit",
                       dir=str(ed))
                run_step([py, str(DEEP), "--round", str(r), "--workdir", str(ed), "--resume",
                          "--br-vs", str(td / "avg.bin"), "--br-player", str(seat),
                          "--eval-every", str(exploit_cfg["iters"]),
                          "--label", f"round {r} exploiter seat {seat + 1}"]
                         + to_flags(exploit_cfg), logfile)
            result[f"exploit_seat{seat + 1}"] = exploit_gain(ed)
        if r == 1:
            evals = [json.loads(l) for l in (td / "log.jsonl").read_text().splitlines() if "eval_avg" in l]
            if evals:
                m = re.search(r"NashConv ([0-9.]+)", evals[-1]["eval_avg"])
                result["exact_nashconv"] = float(m.group(1)) if m else None
        result["strategy"] = str(td / "avg.bin")
        summary[str(r)] = result
        write_json(summary_path, summary)

    status("all done")


if __name__ == "__main__":
    main()
