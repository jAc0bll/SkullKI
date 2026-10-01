#!/usr/bin/env bash
# Stop the background training. Progress is kept: checkpoints are written every
# few iterations and finished rounds are never redone. Resume with start.sh.
set -euo pipefail
SESSION=skullki
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Stopped. Resume with: bash scripts/start.sh"
else
    echo "No training running."
fi
