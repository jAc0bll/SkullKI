#!/usr/bin/env bash
# Run on the NEW server: copy the training progress directly from the OLD
# server (server to server, nothing passes through your own PC).
#
#   bash scripts/migrate_pull.sh root@<old-ip> <old-ssh-port>
#
# Needs SSH access from the new to the old server (see docs/SERVER.md,
# "Umzug"). Copies runs/all including the running step's checkpoint, so the
# new server continues mid-step; per-iteration sample files are skipped
# (they are regenerated). Safe to re-run: rsync only copies what changed.
set -euo pipefail
SRC=${1:?usage: migrate_pull.sh root@<old-ip> [port]}
PORT=${2:-22}
cd "$(dirname "$0")/.."
mkdir -p runs/all
rsync -avP --exclude 's_*.npy' --exclude 'v_*.npy' \
      -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
      "$SRC:SkullKI/runs/all/" runs/all/
echo
echo "Copied. Check with: bash scripts/status.sh    Continue with: bash scripts/start.sh"
