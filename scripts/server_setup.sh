#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU server (Ubuntu 22.04/24.04, e.g. RunPod or vast.ai).
#
#   curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/gto/scripts/server_setup.sh | bash
#   # or, inside a clone:  bash scripts/server_setup.sh
#
# Installs build tools + tmux, clones/updates the repo, builds the C++ tools,
# runs the fast tests, creates .venv with PyTorch (CUDA wheels on Linux) and
# checks that the GPU is visible.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/jAc0bll/SkullKI}
BRANCH=${BRANCH:-gto}
DIR=${DIR:-$HOME/SkullKI}

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "==> system packages"
$SUDO apt-get update -qq
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    git build-essential cmake ninja-build python3 python3-venv python3-pip tmux htop >/dev/null

echo "==> repository ($BRANCH) in $DIR"
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" fetch -q origin
    git -C "$DIR" checkout -q "$BRANCH"
    git -C "$DIR" pull -q --ff-only origin "$BRANCH"
else
    git clone -q --branch "$BRANCH" "$REPO_URL" "$DIR"
fi
cd "$DIR"

echo "==> build ($(nproc) cores)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSK_BUILD_TORCH=OFF -DSK_BUILD_PYTHON=OFF >/dev/null
cmake --build build -j"$(nproc)"

echo "==> tests"
./build/engine/tests/sk_tests --reporter compact
./build/search/tests/sk_search_tests --reporter compact
# Solver tests without the slow exact-evaluation cases.
./build/solver/tests/sk_solver_tests --reporter compact "~[br]~[eval]~[mccfr]"

echo "==> python environment"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy torch
.venv/bin/python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA available:", ok,
      "|", torch.cuda.get_device_name(0) if ok else "training will run on the CPU")
EOF

echo
echo "Setup done. Start training with:   bash scripts/start.sh"
echo "Watch progress with:               bash scripts/status.sh --watch"
