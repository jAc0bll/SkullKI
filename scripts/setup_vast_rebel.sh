#!/usr/bin/env bash
# Vast.ai setup + training launcher for Skull King ReBeL
# 2x RTX PRO 6000 S Blackwell + AMD EPYC 9754 (128c/256t) + 386 GB RAM
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/rebel/scripts/setup_vast_rebel.sh | bash

set -euo pipefail

REPO_URL="https://github.com/jAc0bll/SkullKI.git"
if [ -d "/workspace" ]; then
    REPO_DIR="/workspace/SkullKI"
else
    REPO_DIR="/root/SkullKI"
fi
VENV="$REPO_DIR/venv"
PY=""
CONFIG="$REPO_DIR/configs/rebel/rebel_v1.yaml"
LOG="$REPO_DIR/training_rebel_v1.log"
SESSION="rebel"

echo "=== Skull King ReBeL — Vast.ai Setup ==="
echo "    Repo dir: $REPO_DIR"

# ── 1. Clone or update repo (rebel branch) ────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/6] Repo exists — pulling rebel branch..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout -B rebel origin/rebel
    git -C "$REPO_DIR" reset --hard origin/rebel
else
    echo "[1/6] Cloning repo (rebel branch)..."
    git clone --branch rebel "$REPO_URL" "$REPO_DIR"
fi
echo "      Config: $CONFIG"

# ── 2. Venv + dependencies ────────────────────────────────────────────────
echo "[2/6] Setting up venv..."
if [ -f "/venv/main/bin/activate" ]; then
    VENV="/venv/main"
    PY="$VENV/bin/python"
    echo "      Using Vast.ai system venv: $VENV"
elif [ -f "$REPO_DIR/venv/bin/activate" ]; then
    PY="$VENV/bin/python"
    echo "      Using existing repo venv"
else
    python3 -m venv "$VENV"
    PY="$VENV/bin/python"
    echo "      Created new venv"
fi

"$VENV/bin/pip" install --root-user-action=ignore -r "$REPO_DIR/requirements.txt"
echo "      Dependencies OK."

# ── 3. Verify GPUs ────────────────────────────────────────────────────────
echo "[3/6] GPU check..."
"$PY" - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("  ERROR: CUDA not available — aborting"); sys.exit(1)
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name}  ({mem:.0f} GB)")
EOF

# ── 4. Build C engine ─────────────────────────────────────────────────────
echo "[4/6] Building C extension..."
(
    cd "$REPO_DIR/skull_king/_core"
    "$PY" setup_engine.py build_ext --inplace 2>&1 | tail -3
)
echo "      Build complete."

# ── 5. Show config ────────────────────────────────────────────────────────
echo "[5/6] ReBeL config:"
cat "$CONFIG"

# ── 6. Launch in tmux ─────────────────────────────────────────────────────
echo "[6/6] Starting ReBeL training..."
TRAIN_CMD="source '$VENV/bin/activate' && cd '$REPO_DIR' && python -m skull_king.training.rebel.train --config '$CONFIG' 2>&1 | tee '$LOG'; echo 'Done.'"

if command -v tmux &>/dev/null; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" bash -c "$TRAIN_CMD"
    echo ""
    echo "=== Setup complete ==="
    echo "  Attach:      tmux attach -t $SESSION"
    echo "  Detach:      Ctrl+B then D"
    echo "  Follow log:  tail -f $LOG"
else
    eval "$TRAIN_CMD"
fi

sleep 2
tail -f "$LOG"
