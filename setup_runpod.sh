#!/usr/bin/env bash
# RunPod setup + training launcher for Skull King CFR v7 (split nets, 5090)
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/features/setup_runpod.sh | bash

set -euo pipefail

REPO_URL="https://github.com/jAc0bll/SkullKI.git"
REPO_DIR="/workspace/SkullKI"
CONFIG="$REPO_DIR/cfr_config_v7_5090.yaml"
LOG="$REPO_DIR/training_v7_5090.log"
SESSION="cfr_v7"

echo "=== Skull King CFR v7 — RunPod Setup ==="

# ── 1. Clone or update repo ────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Repo exists — switching to features branch..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout -B features origin/features
else
    echo "[1/5] Cloning repo (features branch)..."
    git clone --branch features "$REPO_URL" "$REPO_DIR"
fi

echo "      Files in repo:"
ls "$REPO_DIR"/*.yaml

# ── 2. Install Python dependencies ────────────────────────────────────────
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_DIR/requirements.txt"

# ── 3. Verify GPU ─────────────────────────────────────────────────────────
echo "[3/5] GPU check..."
python3 - <<'EOF'
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
WORKERS=$(( NCORES > 8 ? NCORES - 4 : NCORES ))
echo "  vCPUs detected: $NCORES  ->  num_workers: $WORKERS"
sed -i "s/^num_workers:.*/num_workers: $WORKERS/" "$CONFIG"
echo "  Config: $CONFIG"

# ── 5. Launch in tmux ─────────────────────────────────────────────────────
echo "[5/5] Starting training in tmux session '$SESSION'..."

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
    "cd '$REPO_DIR' && python -m skull_king.training.cfr.train --config '$CONFIG' \
     2>&1 | tee '$LOG'; echo 'Training finished.'"

echo ""
echo "=== Setup complete ==="
echo "  Attach:      tmux attach -t $SESSION"
echo "  Detach:      Ctrl+B then D"
echo "  Follow log:  tail -f $LOG"
echo "  Checkpoints: $REPO_DIR/models/skull_king/"
echo ""

sleep 2
tail -f "$LOG"
