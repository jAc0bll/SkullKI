#!/usr/bin/env bash
# Progress of the training started with scripts/start.sh.
#   bash scripts/status.sh            snapshot
#   bash scripts/status.sh --watch    refresh every 10 s; Ctrl+C closes only the view
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python train/status.py --workdir "${WORKDIR:-runs/all}" "$@"
