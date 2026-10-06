"""Neural CFR (ESCHER-style) for one Skull King round.

The C++ tool `sk_deep` plays the games and evaluates the networks; this
script trains them (PyTorch) and drives the loop:

  for t = 1..T:
      1. sk_deep gen-values   (sigma_t)        -> train value net V_t
      2. sk_deep gen-regrets  (sigma_t, V_t)   -> add to regret / policy buffers
      3. train regret net from scratch on the regret buffer (weight t)
         -> sigma_{t+1} = regret matching on its output
      every --eval-every iterations (and at the end): train the
      average-strategy net on the policy buffer; round 1 also gets its exact
      NashConv.

Long runs are resumable: every --checkpoint-every iterations the buffers,
the value net and the iteration counter are saved to the workdir; --resume
continues from there. progress.json in the workdir always holds the current
iteration and an ETA (read by scripts/status.sh).

Usage (round 1 gate, from the repo root):
  .venv/Scripts/python train/deep_cfr.py --round 1 --iters 30 --workdir runs/deep_r1

Best-response mode (exploitability estimate where exact evaluation is
impossible): --br-vs avg.bin --br-player i trains only seat i against the
fixed average strategy of the others and finally reports how much seat i
gains over playing the average strategy itself (a lower bound on that
seat's exploitability).
"""
from __future__ import annotations

import argparse
import json
import os
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
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
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Reservoir buffer (Deep CFR keeps a uniform sample of all iterations)
# --------------------------------------------------------------------------
class Reservoir:
    FIELDS = ("x", "mask", "target", "iter")

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

    def save(self, prefix: Path) -> None:
        for f in self.FIELDS:
            np.save(f"{prefix}_{f}.npy", getattr(self, f)[: self.size])
        (Path(f"{prefix}_meta.json")).write_text(json.dumps({"size": self.size, "seen": self.seen}))

    def load(self, prefix: Path) -> None:
        meta = json.loads(Path(f"{prefix}_meta.json").read_text())
        self.size, self.seen = meta["size"], meta["seen"]
        for f in self.FIELDS:
            getattr(self, f)[: self.size] = np.load(f"{prefix}_{f}.npy")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
class Batches:
    """Random training batches drawn from numpy arrays.

    On CUDA the arrays are copied to GPU memory once per training phase and
    batches are gathered there: much faster than gathering on the CPU and
    copying every step (that was the bottleneck: one busy CPU core, idle
    GPU). Falls back to CPU gathering if the GPU runs out of memory."""

    def __init__(self, arrays: list[np.ndarray], n: int, dev: torch.device,
                 gen: torch.Generator):
        self.n, self.dev, self.gen = n, dev, gen
        self.cpu = [a[:n] for a in arrays]
        self.gpu = None
        if dev.type == "cuda":
            try:
                self.gpu = [torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in self.cpu]
            except torch.cuda.OutOfMemoryError:
                self.gpu = None
                torch.cuda.empty_cache()
                print("note: training data does not fit on the GPU, gathering on the CPU",
                      flush=True)

    def sample(self, batch: int) -> list[torch.Tensor]:
        b = min(batch, self.n)
        idx = torch.randint(0, self.n, (b,), generator=self.gen)
        if self.gpu is not None:
            idx = idx.to(self.dev)
            return [t[idx] for t in self.gpu]
        np_idx = idx.numpy()
        return [torch.from_numpy(a[np_idx]).to(self.dev) for a in self.cpu]

    def close(self) -> None:
        self.gpu = None
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()


def train_masked(net, buf: Reservoir, steps: int, batch: int, lr: float, kind: str,
                 scale: float, gen: torch.Generator, dev: torch.device,
                 weight_power: float = 1.0) -> float:
    """kind='regret': weighted MSE on legal actions (targets / scale).
    kind='policy': weighted cross-entropy of masked softmax vs. target sigma.
    Sample weight = iteration ** weight_power (1 = Linear CFR; 2 = DCFR's
    gamma=2 for the average strategy, forgetting early iterations faster)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data = Batches([buf.x, buf.mask, buf.target, buf.iter], buf.size, dev, gen)
    last = 0.0
    for step in range(steps):
        xb, mb, y, it = data.sample(batch)
        x = xb.float()
        m = mb.bool()
        w = it ** weight_power
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
    data.close()
    return last


def train_value(net, x: np.ndarray, y: np.ndarray, steps: int, batch: int, lr: float,
                scale: float, gen: torch.Generator, dev: torch.device) -> float:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data = Batches([x, y], len(y), dev, gen)
    last = 0.0
    for step in range(steps):
        xb, yb = data.sample(batch)
        loss = ((net(xb.float()).squeeze(1) - yb / scale) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == steps - 1:
            last = loss.item()
    data.close()
    return last


def run(cmd: list[str]) -> str:
    res = subprocess.run([str(SK_DEEP)] + cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"sk_deep {' '.join(cmd)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout.strip()


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(wd: Path, t: int, regret_buf: Reservoir, policy_buf: Reservoir,
                    value_net: nn.Module, elapsed: float) -> None:
    ck = wd / "ckpt"
    ck.mkdir(exist_ok=True)
    regret_buf.save(ck / "regret")
    policy_buf.save(ck / "policy")
    torch.save(value_net.state_dict(), ck / "value.pt")
    # Written last: a checkpoint only counts once this file says so.
    write_json(ck / "state.json", {"iter": t, "elapsed": elapsed})


def load_checkpoint(wd: Path, regret_buf: Reservoir, policy_buf: Reservoir,
                    value_net: nn.Module) -> tuple[int, float]:
    ck = wd / "ckpt"
    state_file = ck / "state.json"
    if not state_file.exists():
        return 0, 0.0
    state = json.loads(state_file.read_text())
    regret_buf.load(ck / "regret")
    policy_buf.load(ck / "policy")
    value_net.load_state_dict(torch.load(ck / "value.pt", map_location="cpu"))
    return int(state["iter"]), float(state.get("elapsed", 0.0))


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
    ap.add_argument("--br-vs", type=Path, default=None,
                    help="train a best response for --br-player against this average-strategy net")
    ap.add_argument("--br-player", type=int, default=0)
    ap.add_argument("--match-deals", type=int, default=2_000_000)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="continue from workdir/ckpt if present")
    ap.add_argument("--label", default="", help="shown in progress.json (e.g. 'round 5')")
    ap.add_argument("--policy-keep", type=float, default=1.0,
                    help="fraction of average-strategy samples kept (thin when generating lots of data)")
    args = ap.parse_args()

    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wd = args.workdir
    wd.mkdir(parents=True, exist_ok=True)
    log = open(wd / "log.jsonl", "a")

    regret_buf = Reservoir(args.buffer, INFO_DIM, rng)
    policy_buf = Reservoir(args.buffer, INFO_DIM, rng)
    value_net = mlp(HIST_DIM, args.hidden, args.layers, 1)
    start_iter, elapsed_before = (load_checkpoint(wd, regret_buf, policy_buf, value_net)
                                  if args.resume else (0, 0.0))
    value_net.to(dev)
    policy_path: Path | None = (wd / "regret.bin") if start_iter > 0 else None
    if start_iter:
        print(f"resumed after iteration {start_iter} (buffers {regret_buf.size}/{policy_buf.size})",
              flush=True)
    print(f"device {dev}", flush=True)
    t_start = time.time() - elapsed_before
    iter_times: list[float] = []

    for t in range(start_iter + 1, args.iters + 1):
        it0 = time.time()
        pol = ["--policy", str(policy_path)] if policy_path else []
        if args.br_vs:
            pol += ["--learner", str(args.br_player), "--opp-policy", str(args.br_vs),
                    "--opp-mode", "softmax"]
        common = ["--round", str(args.round), "--seed", str(args.seed * 1000 + t)] + pol
        rec = {"iter": t}

        # 1. value net on on-policy returns of sigma_t
        if not args.exact_values:
            run(["gen-values", "--count", str(args.value_games), "--out", str(wd / "v")] + common)
            vx = np.load(wd / "v_x.npy")
            vy = np.load(wd / "v_y.npy")
            rec["value_samples"] = len(vy)
            rec["value_loss"] = train_value(value_net, vx, vy, args.value_steps, args.batch,
                                            args.lr, args.scale, gen, dev)
            export(value_net, wd / "value.bin", args.scale)
            vals = ["--value", str(wd / "value.bin")]
        else:
            vals = ["--exact-values"]

        # 2. regret + policy samples
        keep = 0.0 if args.br_vs else args.policy_keep   # exploiters need no policy samples
        run(["gen-regrets", "--count", str(args.regret_traj), "--out", str(wd / "s"),
             "--policy-keep", str(keep)] + common + vals)
        rx, rm, rt = (np.load(wd / f"s_{k}.npy") for k in ("rx", "rmask", "rtarget"))
        regret_buf.add(rx, rm, rt, t)
        if not args.br_vs:
            px, pm, pt = (np.load(wd / f"s_{k}.npy") for k in ("px", "pmask", "ptarget"))
            policy_buf.add(px, pm, pt, t)
            rec["policy_samples"] = len(px)
        rec["regret_samples"] = len(rx)

        # 3. regret net from scratch -> sigma_{t+1}
        rnet = mlp(INFO_DIM, args.hidden, args.layers, ACT_DIM).to(dev)
        rec["regret_loss"] = train_masked(rnet, regret_buf, args.regret_steps, args.batch,
                                          args.lr, "regret", args.scale, gen, dev,
                                          args.regret_weight_power)
        policy_path = wd / "regret.bin"
        export(rnet, policy_path, args.scale)

        is_eval = t % args.eval_every == 0 or t == args.iters

        # Best-response mode: how much does the trained seat gain?
        if args.br_vs and is_eval:
            seat = ["--round", str(args.round), "--learner", str(args.br_player),
                    "--opp-policy", str(args.br_vs), "--count", str(args.match_deals),
                    "--seed", str(424242)]
            rec["br_match"] = run(["match", "--policy", str(policy_path), "--mode", "rm"] + seat)
            # A best response against fixed opponents should be deterministic:
            # also measure always playing the highest-regret action.
            rec["br_match_argmax"] = run(["match", "--policy", str(policy_path), "--mode", "argmax"]
                                         + seat)
            rec["base_match"] = run(["match", "--policy", str(args.br_vs), "--mode", "softmax"] + seat)

        # Average strategy (+ exact evaluation in round 1)
        if not args.br_vs and is_eval:
            pnet = mlp(INFO_DIM, args.hidden, args.layers, ACT_DIM).to(dev)
            rec["policy_loss"] = train_masked(pnet, policy_buf, args.policy_steps, args.batch,
                                              args.lr, "policy", 1.0, gen, dev,
                                              args.avg_weight_power)
            export(pnet, wd / "avg.bin")
            if args.round == 1:
                rec["eval_avg"] = run(["eval", "--round", "1", "--policy", str(wd / "avg.bin"),
                                       "--mode", "softmax"])
                rec["eval_current"] = run(["eval", "--round", "1", "--policy", str(policy_path),
                                           "--mode", "rm"])

        if t % args.checkpoint_every == 0 and t < args.iters:
            save_checkpoint(wd, t, regret_buf, policy_buf, value_net, time.time() - t_start)

        iter_times.append(time.time() - it0)
        rec["iter_seconds"] = round(iter_times[-1], 1)
        rec["total_seconds"] = round(time.time() - t_start, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        recent = iter_times[-5:]
        write_json(wd / "progress.json", {
            "label": args.label, "iter": t, "iters": args.iters,
            "sec_per_iter": round(sum(recent) / len(recent), 1),
            "eta_seconds": round((args.iters - t) * sum(recent) / len(recent)),
            "elapsed_seconds": round(time.time() - t_start), "updated": time.time(),
            "last": {k: v for k, v in rec.items() if isinstance(v, (int, float, str))},
        })
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
        print(json.dumps(short), flush=True)

    (wd / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
