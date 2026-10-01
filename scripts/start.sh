#!/usr/bin/env bash
# Start (or resume) the full training in the background.
#
#   bash scripts/start.sh                      # rounds 1-10, full settings
#   bash scripts/start.sh --rounds 4-10        # extra args go to train/run_all.py
#
# The run lives in a tmux session called "skullki", so it keeps going when
# your SSH connection or terminal closes. Everything is resumable: running
# this again after a crash or reboot continues where it stopped.
#
#   bash scripts/status.sh --watch     progress view (Ctrl+C only closes the view)
#   tmux attach -t skullki             live log (leave with Ctrl+B, then D)
#   bash scripts/stop.sh               stop (resume later with start.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

WORKDIR=${WORKDIR:-runs/all}
SESSION=skullki
mkdir -p "$WORKDIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Training is already running (tmux session '$SESSION')."
    echo "Watch it:  bash scripts/status.sh --watch    or    tmux attach -t $SESSION"
    exit 0
fi

CMD=".venv/bin/python -u train/run_all.py --workdir $WORKDIR $*"
# `exec bash` at the end keeps the tmux window open if the run fails, so the
# error stays readable after `tmux attach`.
tmux new-session -d -s "$SESSION" "$CMD 2>&1 | tee -a $WORKDIR/console.log; echo; echo '[run ended]'; exec bash"

echo "Training started in the background (tmux session '$SESSION')."
echo "  progress:  bash scripts/status.sh --watch"
echo "  live log:  tmux attach -t $SESSION   (leave with Ctrl+B, then D)"
echo "  stop:      bash scripts/stop.sh"
