"""Export a trained PolicyValueNet to TorchScript so LibTorch (C++) can load it.

Usage:
    python train/export.py --ckpt train/checkpoints/bc_400sims.pt \
                           --out  train/checkpoints/bc_400sims.scripted.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))

from model import PolicyValueNet, ENC_DIM, ACTION_DIM, N_PLAYERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint produced by train.py")
    ap.add_argument("--out",  required=True, help="output .pt path for TorchScript module")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    hidden = ckpt.get("hidden", 512)
    model = PolicyValueNet(hidden=hidden)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Scripting works (no data-dependent control flow in PolicyValueNet) and is
    # robust under shape variation, so we prefer it over tracing.
    scripted = torch.jit.script(model)

    # Quick sanity test on a batch of 2.
    with torch.no_grad():
        x = torch.zeros(2, ENC_DIM)
        policy, value = scripted(x)
    assert policy.shape == (2, ACTION_DIM),  f"expected (2,{ACTION_DIM}), got {tuple(policy.shape)}"
    assert value.shape  == (2, N_PLAYERS),   f"expected (2,{N_PLAYERS}), got {tuple(value.shape)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))

    sz_mb = out.stat().st_size / 1e6
    print(f"Exported TorchScript module")
    print(f"  src   : {args.ckpt} (hidden={hidden})")
    print(f"  dst   : {out}  ({sz_mb:.2f} MB)")
    print(f"  shape : input (B, {ENC_DIM})  ->  policy (B, {ACTION_DIM}), value (B, {N_PLAYERS})")


if __name__ == "__main__":
    main()
