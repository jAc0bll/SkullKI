"""Export a trained BeliefNet to TorchScript for LibTorch loading.

Usage:
    python train/export_belief.py --ckpt train/checkpoints/belief_v1.pt \
                                  --out  train/checkpoints/belief_v1.scripted.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))

from belief_model import BeliefNet, ENC_DIM, N_PLAYERS, N_CARDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    hidden = ckpt.get("hidden", 512)
    model = BeliefNet(hidden=hidden)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scripted = torch.jit.script(model)

    with torch.no_grad():
        x = torch.zeros(2, ENC_DIM)
        logits = scripted(x)
    assert logits.shape == (2, N_PLAYERS, N_CARDS), f"unexpected shape {tuple(logits.shape)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))

    sz_mb = out.stat().st_size / 1e6
    print(f"Exported BeliefNet TorchScript module")
    print(f"  src   : {args.ckpt} (hidden={hidden})")
    print(f"  dst   : {out}  ({sz_mb:.2f} MB)")
    print(f"  shape : input (B, {ENC_DIM})  ->  logits (B, {N_PLAYERS}, {N_CARDS})")


if __name__ == "__main__":
    main()
