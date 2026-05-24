#!/usr/bin/env bash
# Vast.ai setup + training launcher for Skull King CFR v8 (split nets, 2x RTX PRO 6000 S)
# AMD EPYC 9754 (128c/256t) + 2x RTX PRO 6000 S + 386 GB RAM
# v8: win-bonus utility, heuristic_frac 0.6, 150 workers, multi-GPU parallel training
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/features/scripts/setup_vast_v8_6000.sh | bash

set -euo pipefail

REPO_URL="https://github.com/jAc0bll/SkullKI.git"
if [ -d "/workspace" ]; then
    REPO_DIR="/workspace/SkullKI"
else
    REPO_DIR="/root/SkullKI"
fi
VENV="$REPO_DIR/venv"
PY=""
CONFIG="$REPO_DIR/configs/cfr/v8_6000.yaml"
LOG="$REPO_DIR/training_v8_6000.log"
SESSION="cfr_v8"

echo "=== Skull King CFR v8 — Vast.ai 2x RTX PRO 6000 S Setup ==="
echo "    Repo dir: $REPO_DIR"

# ── 1. Clone or update repo ────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/6] Repo exists — pulling features branch..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout -B features origin/features
    git -C "$REPO_DIR" reset --hard origin/features
else
    echo "[1/6] Cloning repo (features branch)..."
    git clone --branch features "$REPO_URL" "$REPO_DIR"
fi
echo "      Config files found:"
ls "$REPO_DIR"/configs/cfr/*.yaml "$REPO_DIR"/configs/rebel/*.yaml 2>/dev/null || true

# ── 2. Venv + dependencies ────────────────────────────────────────────────
echo "[2/6] Setting up venv..."
if [ -f "/venv/main/bin/activate" ]; then
    VENV="/venv/main"
    PY="$VENV/bin/python"
    echo "      Using Vast.ai system venv: $VENV"
elif [ -f "$REPO_DIR/venv/bin/activate" ]; then
    PY="$VENV/bin/python"
    echo "      Using existing repo venv: $VENV"
else
    python3 -m venv "$VENV"
    PY="$VENV/bin/python"
    echo "      Created new venv: $VENV"
fi

echo "      Installing/checking requirements (may take a minute)..."
"$VENV/bin/pip" install --root-user-action=ignore -r "$REPO_DIR/requirements.txt"
echo "      Dependencies OK."

# ── 3. Verify GPUs ────────────────────────────────────────────────────────
echo "[3/6] GPU check..."
"$PY" - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("  ERROR: CUDA not available — aborting")
    sys.exit(1)
n = torch.cuda.device_count()
print(f"  GPUs detected: {n}")
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  ({mem:.0f} GB VRAM)")
print(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
if n < 2:
    print("  WARNING: Only 1 GPU found — training will use single-GPU mode")
EOF

# ── 4. Build C engine ─────────────────────────────────────────────────────
echo "[4/6] Building C extension (traverse_split ~5ms/game)..."
(
    cd "$REPO_DIR/skull_king/_core"
    "$PY" setup_engine.py build_ext --inplace 2>&1 | tail -5
)
echo "      Build complete."

"$PY" - "$REPO_DIR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from skull_king.cfr_engine import SplitCEngine
if SplitCEngine.available:
    print("  C engine: ACTIVE (split traversal)")
else:
    print("  WARNING: C engine NOT available — using slow Python traversal")
PYEOF

# ── 5. Show config ────────────────────────────────────────────────────────
echo "[5/6] Config (150 workers for 128c/256t EPYC 9754)..."
cat "$CONFIG"

# ── 6. Launch in tmux (fall back to screen if missing) ────────────────────
echo "[6/6] Starting training..."
TRAIN_CMD="source '$VENV/bin/activate' && cd '$REPO_DIR' && python -m skull_king.training.cfr.train --config '$CONFIG' 2>&1 | tee '$LOG'; echo 'Done.'"

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
