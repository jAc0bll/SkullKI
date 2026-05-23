#!/usr/bin/env bash
# Vast.ai setup + training launcher for Skull King CFR v7 (split nets, 4090)
# AMD EPYC 7B13 (64c/128t) + RTX 4090 24GB
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/features/setup_vast_4090.sh | bash

set -euo pipefail

REPO_URL="https://github.com/jAc0bll/SkullKI.git"
# Vast.ai uses /root, RunPod uses /workspace — pick whichever exists
if [ -d "/workspace" ]; then
    REPO_DIR="/workspace/SkullKI"
else
    REPO_DIR="/root/SkullKI"
fi
CONFIG="$REPO_DIR/cfr_config_v7_4090.yaml"
LOG="$REPO_DIR/training_v7_4090.log"
SESSION="cfr_v7"

echo "=== Skull King CFR v7 — Vast.ai RTX 4090 Setup ==="
echo "    Repo dir: $REPO_DIR"

# ── 1. Clone or update repo ────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/6] Repo exists — pulling features branch..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout -B features origin/features
else
    echo "[1/6] Cloning repo (features branch)..."
    git clone --branch features "$REPO_URL" "$REPO_DIR"
fi
echo "      Config files found:"
ls "$REPO_DIR"/*.yaml

# ── 2. Install Python dependencies ────────────────────────────────────────
echo "[2/6] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REPO_DIR/requirements.txt"

# ── 3. Verify GPU ─────────────────────────────────────────────────────────
echo "[3/6] GPU check..."
python3 - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("  ERROR: CUDA not available — aborting")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU:   {name}  ({mem:.0f} GB VRAM)")
print(f"  CUDA:  {torch.version.cuda}  |  PyTorch: {torch.__version__}")
EOF

# ── 4. Build C engine ─────────────────────────────────────────────────────
echo "[4/6] Building C extension (traverse_split ~5ms/game)..."
(
    cd "$REPO_DIR/skull_king/_core"
    python setup_engine.py build_ext --inplace 2>&1 | tail -5
)
echo "      Build complete."

# Verify C engine loaded
python3 - "$REPO_DIR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from skull_king.cfr_engine import SplitCEngine
if SplitCEngine.available:
    print("  C engine: ACTIVE (split traversal)")
else:
    print("  WARNING: C engine NOT available — using slow Python traversal")
PYEOF

# ── 5. Tune worker count to this machine ──────────────────────────────────
echo "[5/6] Tuning worker count..."
NCORES=$(nproc --all)
# Leave ~28 threads for OS / manager / GPU trainer
WORKERS=$(( NCORES > 32 ? NCORES - 28 : NCORES - 4 ))
echo "  Threads detected: $NCORES  ->  num_workers: $WORKERS"
sed -i "s/^num_workers:.*/num_workers: $WORKERS/" "$CONFIG"
cat "$CONFIG"

# ── 6. Launch in tmux (fall back to screen if missing) ────────────────────
echo "[6/6] Starting training..."
TRAIN_CMD="cd '$REPO_DIR' && python -m skull_king.training.cfr.train --config '$CONFIG' 2>&1 | tee '$LOG'; echo 'Done.'"

if command -v tmux &>/dev/null; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" bash -c "$TRAIN_CMD"
    echo ""
    echo "=== Setup complete ==="
    echo "  Attach:      tmux attach -t $SESSION"
    echo "  Detach:      Ctrl+B then D"
elif command -v screen &>/dev/null; then
    screen -S "$SESSION" -d -m bash -c "$TRAIN_CMD"
    echo ""
    echo "=== Setup complete ==="
    echo "  Attach:      screen -r $SESSION"
    echo "  Detach:      Ctrl+A then D"
else
    echo "  No tmux/screen — running in foreground (use Ctrl+C to stop)..."
    eval "$TRAIN_CMD"
    exit 0
fi

echo "  Follow log:  tail -f $LOG"
echo "  Checkpoints: $REPO_DIR/models/skull_king/"
echo ""

sleep 2
tail -f "$LOG"
