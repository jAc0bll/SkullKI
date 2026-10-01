"""Neural CFR (ESCHER-style) for one Skull King round.

The C++ tool `sk_deep` plays the games and evaluates the networks; this
script trains them (PyTorch) and drives the loop:

  for t = 1..T:
      1. sk_deep gen-values   (sigma_t)        -> train value net V_t
      2. sk_deep gen-regrets  (sigma_t, V_t)   -> add to regret / policy buffers
      3. train regret net from scratch on the regret buffer (weight t)
         -> sigma_{t+1} = regret matching on its output
      every --eval-every iterations: train the average-strategy net on the
      policy buffer and measure exact NashConv (round 1 only).

Usage (round 1 gate, from the repo root):
  .venv/Scripts/python train/deep_cfr.py --round 1 --iters 30 --workdir runs/deep_r1
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SK_DEEP = REPO / "build" / "tools" / ("sk_deep.exe" if sys.platform == "win32" else "sk_deep")

INFO_DIM = 593
HIST_DIM = INFO_DIM + 3 * 61
ACT_DIM = 74


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
def mlp(inp: int, hidden: int, layers: int, out: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


def export(net: nn.Sequential, path: Path, out_scale: float = 1.0) -> None:
    """Write SKMLP001 (see solver/include/sk/solver/mlp.hpp). The last layer is
    multiplied by out_scale so C++ sees outputs in real units."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with open(path, "wb") as f:
        f.write(b"SKMLP001")
        f.write(struct.pack("<I", len(linears)))
        for i, lin in enumerate(linears):
            w = lin.weight.detach().cpu().numpy().astype(np.float32)
            b = lin.bias.detach().cpu().numpy().astype(np.float32)
            if i == len(linears) - 1:
                w, b = w * out_scale, b * out_scale
            f.write(struct.pack("<II", w.shape[1], w.shape[0]))
            f.write(np.ascontiguousarray(w).tobytes())
            f.write(np.ascontiguousarray(b).tobytes())


# --------------------------------------------------------------------------
# Reservoir buffer (Deep CFR keeps a uniform sample of all iterations)
# --------------------------------------------------------------------------
class Reservoir:
    def __init__(self, capacity: int, x_dim: int, rng: np.random.Generator):
        self.cap = capacity
        self.x = np.zeros((capacity, x_dim), np.uint8)
        self.mask = np.zeros((capacity, ACT_DIM), np.uint8)
        self.target = np.zeros((capacity, ACT_DIM), np.float32)
        self.iter = np.zeros(capacity, np.float32)
        self.size = 0
        self.seen = 0
        self.rng = rng

    def add(self, x, mask, target, it: int) -> None:
        n = len(x)
        # Fill free slots first, then reservoir-replace.
        free = min(n, self.cap - self.size)
        if free:
            sl = slice(self.size, self.size + free)
            self.x[sl], self.mask[sl], self.target[sl], self.iter[sl] = x[:free], mask[:free], target[:free], it
            self.size += free
        rest = n - free
        if rest:
            idx = self.seen + free + np.arange(rest)
            slots = (self.rng.random(rest) * (idx + 1)).astype(np.int64)
            keep = slots < self.cap
            src = free + np.nonzero(keep)[0]
            dst = slots[keep]
            self.x[dst], self.mask[dst], self.target[dst], self.iter[dst] = x[src], mask[src], target[src], it
        self.seen += n


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_masked(net, buf: Reservoir, steps: int, batch: int, lr: float, kind: str,
                 scale: float, gen: torch.Generator, weight_power: float = 1.0) -> float:
    """kind='regret': weighted MSE on legal actions (targets / scale).
    kind='policy': weighted cross-entropy of masked softmax vs. target sigma.
    Sample weight = iteration ** weight_power (1 = Linear CFR; 2 = DCFR's
    gamma=2 for the average strategy, forgetting early iterations faster)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = buf.size
    last = 0.0
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen).numpy()
        x = torch.from_numpy(buf.x[idx]).float()
        m = torch.from_numpy(buf.mask[idx]).bool()
        y = torch.from_numpy(buf.target[idx])
        w = torch.from_numpy(buf.iter[idx]) ** weight_power
        w = w / w.mean()
        out = net(x)
        if kind == "regret":
            err = ((out - y / scale) ** 2) * m
            loss = ((err.sum(1) / m.sum(1).clamp(min=1)) * w).mean()
        else:
            logits = out.masked_fill(~m, -1e9)
            logp = torch.log_softmax(logits, dim=1)
            loss = (-(y * logp).masked_fill(~m, 0.0).sum(1) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == steps - 1:
            last = loss.item()
    return last


def train_value(net, x: np.ndarray, y: np.ndarray, steps: int, batch: int, lr: float,
                scale: float, gen: torch.Generator) -> float:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(y)
    last = 0.0
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen).numpy()
        xb = torch.from_numpy(x[idx]).float()
        yb = torch.from_numpy(y[idx]) / scale
        loss = ((net(xb).squeeze(1) - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == steps - 1:
            last = loss.item()
    return last


def run(cmd: list[str]) -> str:
    res = subprocess.run([str(SK_DEEP)] + cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"sk_deep {' '.join(cmd)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--workdir", type=Path, default=Path("runs/deep"))
    ap.add_argument("--value-games", type=int, default=50000)
    ap.add_argument("--regret-traj", type=int, default=20000, help="trajectories per traverser")
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--regret-steps", type=int, default=2000)
    ap.add_argument("--value-steps", type=int, default=1500)
    ap.add_argument("--policy-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=50.0, help="points per network unit")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--regret-weight-power", type=float, default=1.0)
    ap.add_argument("--avg-weight-power", type=float, default=1.0)
    ap.add_argument("--exact-values", action="store_true",
                    help="diagnostic: exact action values instead of a value net (tiny rounds)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wd = args.workdir
    wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "log.jsonl", "a")

    regret_buf = Reservoir(args.buffer, INFO_DIM, rng)
    policy_buf = Reservoir(args.buffer, INFO_DIM, rng)
    value_net = mlp(HIST_DIM, args.hidden, args.layers, 1)
    policy_path: Path | None = None
    t_start = time.time()

    for t in range(1, args.iters + 1):
        it0 = time.time()
        pol = ["--policy", str(policy_path)] if policy_path else []
        common = ["--round", str(args.round), "--seed", str(args.seed * 1000 + t)] + pol
        rec = {"iter": t}

        # 1. value net on on-policy returns of sigma_t
        if not args.exact_values:
            out = run(["gen-values", "--count", str(args.value_games), "--out", str(wd / "v")] + common)
            vx = np.load(wd / "v_x.npy")
            vy = np.load(wd / "v_y.npy")
            rec["value_samples"] = len(vy)
            rec["value_loss"] = train_value(value_net, vx, vy, args.value_steps, args.batch,
                                            args.lr, args.scale, gen)
            export(value_net, wd / "value.bin", args.scale)
            vals = ["--value", str(wd / "value.bin")]
        else:
            vals = ["--exact-values"]

        # 2. regret + policy samples
        out = run(["gen-regrets", "--count", str(args.regret_traj), "--out", str(wd / "s")] + common + vals)
        rx, rm, rt = (np.load(wd / f"s_{k}.npy") for k in ("rx", "rmask", "rtarget"))
        px, pm, pt = (np.load(wd / f"s_{k}.npy") for k in ("px", "pmask", "ptarget"))
        regret_buf.add(rx, rm, rt, t)
        policy_buf.add(px, pm, pt, t)
        rec["regret_samples"], rec["policy_samples"] = len(rx), len(px)

        # 3. regret net from scratch -> sigma_{t+1}
        rnet = mlp(INFO_DIM, args.hidden, args.layers, ACT_DIM)
        rec["regret_loss"] = train_masked(rnet, regret_buf, args.regret_steps, args.batch,
                                          args.lr, "regret", args.scale, gen,
                                          args.regret_weight_power)
        policy_path = wd / "regret.bin"
        export(rnet, policy_path, args.scale)

        # Average strategy + exact evaluation
        if t % args.eval_every == 0 or t == args.iters:
            pnet = mlp(INFO_DIM, args.hidden, args.layers, ACT_DIM)
            rec["policy_loss"] = train_masked(pnet, policy_buf, args.policy_steps, args.batch,
                                              args.lr, "policy", 1.0, gen,
                                              args.avg_weight_power)
            export(pnet, wd / "avg.bin")
            if args.round == 1:
                rec["eval_avg"] = run(["eval", "--round", "1", "--policy", str(wd / "avg.bin"),
                                       "--mode", "softmax"])
                rec["eval_current"] = run(["eval", "--round", "1", "--policy", str(policy_path),
                                           "--mode", "rm"])

        rec["iter_seconds"] = round(time.time() - it0, 1)
        rec["total_seconds"] = round(time.time() - t_start, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
        print(json.dumps(short), flush=True)


if __name__ == "__main__":
    main()
