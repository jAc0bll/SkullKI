"""Copy the trained average strategies into the app as float16 nets.

  python scripts/export_models.py runs/full_2026-10-08     # -> app/assets/models/r1..r10.bin

Each round's roundNN/train/avg.bin (SKMLP001, float32) becomes SKMLP016
(float16): half the download; probabilities change by < 0.01.
"""
import struct
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1])
dst = Path(__file__).resolve().parent.parent / "app" / "assets" / "models"
dst.mkdir(parents=True, exist_ok=True)
for r in range(1, 11):
    data = (src / f"round{r:02d}" / "train" / "avg.bin").read_bytes()
    assert data[:8] == b"SKMLP001"
    (n,) = struct.unpack_from("<I", data, 8)
    out, pos = [b"SKMLP016", struct.pack("<I", n)], 12
    for _ in range(n):
        i, o = struct.unpack_from("<II", data, pos)
        pos += 8
        vals = np.frombuffer(data, np.float32, i * o + o, pos)
        pos += (i * o + o) * 4
        out += [struct.pack("<II", i, o), vals.astype(np.float16).tobytes()]
    assert pos == len(data)
    (dst / f"r{r}.bin").write_bytes(b"".join(out))
    print(f"round {r}: {len(data) / 1e6:.1f} MB -> {(dst / f'r{r}.bin').stat().st_size / 1e6:.1f} MB")
