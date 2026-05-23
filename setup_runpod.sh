#!/usr/bin/env bash
# RunPod setup + training launcher for Skull King CFR v7 (split nets, 5090)
#
# Usage (paste into RunPod web terminal or SSH):
#   bash setup_runpod.sh
#
# What it does:
#   1. Clones the repo (or pulls latest if already present)
#   2. Installs Python deps
#   3. Verifies CUDA / GPU
#   4. Detects available CPU cores and patches worker count
#   5. Starts training inside a tmux session (survives SSH disconnect)
#   6. Tails the live log

set -euo pipefail

REPO_URL="https://github.com/jAc0bll/SkullKI.git"
REPO_DIR="/workspace/SkullKI"
CONFIG="cfr_config_v7_5090.yaml"
LOG="training_v7_5090.log"
SESSION="cfr_v7"

echo "=== Skull King CFR v7 — RunPod Setup ==="

# ── 1. Clone or update repo ────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Updating repo..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/5] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 2. Install Python dependencies ────────────────────────────────────────
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 3. Verify GPU ─────────────────────────────────────────────────────────
echo "[3/5] GPU check..."
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("  WARNING: CUDA not available — training will use CPU only")
else:
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {name}  ({mem:.0f} GB VRAM)")
    print(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
EOF

# ── 4. Auto-tune worker count ─────────────────────────────────────────────
echo "[4/5] Tuning worker count..."
NCORES=$(nproc --all)
# Leave 4 cores for OS, IPC, and main training process
WORKERS=$(( NCORES > 8 ? NCORES - 4 : NCORES ))
echo "  vCPUs detected: $NCORES  ->  num_workers: $WORKERS"
# Patch the config in-place (sed on the YAML)
sed -i "s/^num_workers:.*/num_workers: $WORKERS/" "$CONFIG"

# ── 5. Launch in tmux ─────────────────────────────────────────────────────
echo "[5/5] Starting training in tmux session '$SESSION'..."
echo "      Log: $REPO_DIR/$LOG"
echo ""

# Kill existing session if leftover from a previous run
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
    "cd '$REPO_DIR' && python -m skull_king.training.cfr.train --config '$CONFIG' \
     2>&1 | tee '$LOG'; echo 'Training finished.'"

echo "=== Setup complete ==="
echo ""
echo "  Attach to live output:   tmux attach -t $SESSION"
echo "  Detach (keep running):   Ctrl+B then D"
echo "  Follow log file:         tail -f $REPO_DIR/$LOG"
echo "  Kill training:           tmux kill-session -t $SESSION"
echo ""
echo "  Checkpoints saved to:    $REPO_DIR/models/skull_king/"
echo "  Pattern: cfr_v7_5090_iter{250,500,...}_{bid,play}_{adv,strat}.pt"
echo ""

# Tail the log so the user sees it starting up
sleep 2
tail -f "$LOG"
