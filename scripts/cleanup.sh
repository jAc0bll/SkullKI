#!/usr/bin/env bash
# Free disk space: delete checkpoints and sample files of FINISHED steps only
# (those with a DONE marker). The running step and all results (*.bin
# networks, logs, summary.json) are kept. Safe to run while training.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKDIR=${WORKDIR:-runs/all}
before=$(du -sh "$WORKDIR" | cut -f1)
for d in "$WORKDIR"/round*/train "$WORKDIR"/round*/exploit_seat*; do
    [ -f "$d/DONE" ] || continue
    rm -rf "$d/ckpt" "$d"/s_*.npy "$d"/v_*.npy
done
echo "$WORKDIR: $before -> $(du -sh "$WORKDIR" | cut -f1)"
df -h / | tail -1
