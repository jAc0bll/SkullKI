"""Progress view for train/run_all.py.

  python train/status.py --workdir runs/all           # one snapshot
  python train/status.py --workdir runs/all --watch   # refresh every 10 s (Ctrl+C to leave)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from run_all import exploit_gain   # same directory (train/)


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def bar(frac: float, width: int = 24) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + f"] {frac * 100:5.1f}%"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int | None, status: dict, wd: Path) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows; use the age
        # of the newest progress file instead.
        newest = max((p.stat().st_mtime for p in wd.glob("round*/*/progress.json")), default=0.0)
        return time.time() - max(newest, status.get("updated", 0.0)) < 20 * 60
    if not pid:
        return False
    try:
        os.kill(pid, 0)   # signal 0: existence check only (POSIX)
        return True
    except OSError:
        return False


def snapshot(wd: Path) -> str:
    lines = []
    status = load(wd / "status.json") or {}
    summary = load(wd / "summary.json") or {}
    # All rounds that exist anywhere (a later partial run, e.g. --redo-exploiters
    # --rounds 1-4, must not hide the others).
    found = {int(p.name[5:]) for p in wd.glob("round[0-9][0-9]")}
    rounds = sorted(found | {int(k) for k in summary} | set(status.get("rounds") or []))         or list(range(1, 11))
    alive = pid_alive(status.get("pid"), status, wd)
    lines.append(f"Skull King GTO training  |  {wd}")
    if status.get("step") == "all done":
        state = "FINISHED"
    else:
        state = "RUNNING" if alive else "NOT RUNNING (resume with: bash scripts/start.sh)"
    lines.append(f"state: {state}   current step: {status.get('step', '-')}")
    lines.append("")
    lines.append(f"{'round':>5}  {'training':<40} {'exploit seat1':>14} {'exploit seat4':>14}")

    for r in rounds:
        rd = wd / f"round{r:02d}"
        tr = rd / "train"
        if (tr / "DONE").exists():
            tcol = "done"
        else:
            p = load(tr / "progress.json")
            if p:
                tcol = f"{bar(p['iter'] / p['iters'], 16)} {p['iter']}/{p['iters']} ETA {fmt_dur(p['eta_seconds'])}"
            else:
                tcol = "waiting"
        cols = []
        for seat in (1, 4):
            ed = rd / f"exploit_seat{seat}"
            res = (summary.get(str(r)) or {}).get(f"exploit_seat{seat}")
            if not res and (ed / "DONE").exists():
                res = exploit_gain(ed)   # finished, round summary not written yet
            if res:
                cols.append(f"{res['gain']:+.3f}±{res['stderr']:.3f}")
            elif (ed / "progress.json").exists():
                p = load(ed / "progress.json") or {}
                cols.append(f"{p.get('iter', 0)}/{p.get('iters', '?')} {fmt_dur(p.get('eta_seconds'))}")
            else:
                cols.append("-")
        lines.append(f"{r:>5}  {tcol:<40} {cols[0]:>14} {cols[1]:>14}")

    lines.append("")
    if "1" in summary and summary["1"].get("exact_nashconv") is not None:
        lines.append(f"round 1 exact NashConv: {summary['1']['exact_nashconv']:.4f} points/round")
    lines.append("exploit = points per round a trained exploiter wins over the strategy "
                 "(lower = closer to GTO)")
    cur = load(Path(status["dir"]) / "progress.json") if status.get("dir") else None
    eta = fmt_dur(cur["eta_seconds"]) if cur and state.startswith("RUNNING") else "-"
    lines.append(f"updated {time.strftime('%H:%M:%S')}  (current step ETA {eta})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("runs/all"))
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()
    if not args.watch:
        print(snapshot(args.workdir))
        return
    try:
        while True:
            text = snapshot(args.workdir)
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
