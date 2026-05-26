"""Loader for the SKAZ binary self-play format written by tools/selfplay_az.

The file layout is one header block followed by sequential column blocks:

    magic     : 4 bytes ("SKAZ")
    version   : uint32   (currently 1)
    N         : uint64   (number of samples)
    ENC_DIM   : uint32
    ACTION_DIM: uint32
    N_PLAYERS : uint32
    N_CARDS   : uint32

    features    : float32 [N, ENC_DIM]
    policy      : float32 [N, ACTION_DIM]
    legal       : uint8   [N, ACTION_DIM]   (0/1 mask)
    value       : float32 [N, N_PLAYERS]
    hands       : uint8   [N, N_PLAYERS, N_CARDS]   (0/1 mask)
    perspective : uint8   [N]
    round       : uint8   [N]

Returns the same dict shape that train/train.py and train/train_belief.py
expect (mirrors data_gen.py's npz schema).

Usage:
    python -c "from train.load_skaz import load; d = load('out.bin'); print(d['features'].shape)"
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(4)
        if magic != b"SKAZ":
            raise ValueError(f"{p}: not a SKAZ file (magic={magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"{p}: unsupported SKAZ version {version}")
        N, ENC_DIM, ACTION_DIM, N_PLAYERS, N_CARDS = struct.unpack("<QIIII", f.read(24))

        # Each column block is contiguous; np.frombuffer just reads bytes.
        features = np.frombuffer(f.read(N * ENC_DIM * 4),  dtype=np.float32).reshape(N, ENC_DIM)
        policy   = np.frombuffer(f.read(N * ACTION_DIM * 4), dtype=np.float32).reshape(N, ACTION_DIM)
        legal_u8 = np.frombuffer(f.read(N * ACTION_DIM),     dtype=np.uint8  ).reshape(N, ACTION_DIM)
        value    = np.frombuffer(f.read(N * N_PLAYERS * 4),  dtype=np.float32).reshape(N, N_PLAYERS)
        hands_u8 = np.frombuffer(f.read(N * N_PLAYERS * N_CARDS), dtype=np.uint8).reshape(N, N_PLAYERS, N_CARDS)
        perspective = np.frombuffer(f.read(N), dtype=np.uint8).copy()
        round_     = np.frombuffer(f.read(N), dtype=np.uint8).copy()

    return {
        "features":    features.copy(),     # detach from mmap-style buffer
        "policy":      policy.copy(),
        "legal":       legal_u8.astype(bool),
        "value":       value.copy(),
        "hands":       hands_u8.astype(np.float32),  # match data_gen.py dtype
        "perspective": perspective.astype(np.int8),
        "round":       round_,
    }


def convert_to_npz(skaz_path: str | Path, npz_path: str | Path) -> None:
    """Convenience: load a SKAZ file and re-save as numpy npz (so it drops
    straight into the existing train.py / train_belief.py pipelines)."""
    d = load(skaz_path)
    np.savez_compressed(npz_path, **d)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="input .bin (SKAZ)")
    ap.add_argument("--to-npz", default=None, help="optional: also write to this .npz")
    args = ap.parse_args()
    d = load(args.path)
    print(f"Loaded {args.path}: {d['features'].shape[0]} samples")
    for k, v in d.items():
        print(f"  {k:<12}  shape={v.shape}  dtype={v.dtype}")
    if args.to_npz:
        np.savez_compressed(args.to_npz, **d)
        print(f"Wrote {args.to_npz}")
