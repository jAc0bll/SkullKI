#!/usr/bin/env bash
# Bootstrap a fresh vast.ai instance for Skull King AlphaZero training.
#
# Expected base image (recommended on vast.ai):
#     pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
#
# Usage on the instance after SSH'ing in:
#     git clone <your-repo-url> /workspace
#     cd /workspace
#     bash scripts/vastai_setup.sh
#
# After this finishes you can launch the AlphaZero loop, e.g.:
#     python train/alphazero_loop.py --workdir runs/az1 \
#         --init train/checkpoints/bc_v3_mc.pt --iterations 5 \
#         --selfplay-games 2000 --workers $(nproc) --device cuda

set -euo pipefail

echo "=== installing system build tools ==="
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates htop tmux

echo
echo "=== python + torch sanity ==="
python -c "
import torch
print('torch:   ', torch.__version__)
print('cuda:    ', torch.cuda.is_available())
print('devices: ', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  [{i}] {torch.cuda.get_device_name(i)}')
"

echo
echo "=== building project ==="
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DSK_BUILD_TESTS=ON \
    -DSK_BUILD_PYTHON=ON \
    -DSK_BUILD_TORCH=ON
cmake --build build -j"$(nproc)"

echo
echo "=== running smoke tests ==="
./build/engine/tests/sk_tests       --reporter compact
./build/agents/tests/sk_agent_tests --reporter compact
./build/search/tests/sk_search_tests --reporter compact

echo
echo "=== quick python bindings check ==="
python - <<'PY'
import sys
sys.path.insert(0, 'build/python')
import skullking as sk
print('HAS_TORCH:', sk.HAS_TORCH)
print('N_PLAYERS=%d  N_CARDS=%d  ENC_DIM=%d  ACTION_DIM=%d' %
      (sk.N_PLAYERS, sk.N_CARDS, sk.ENC_DIM, sk.ACTION_DIM))
PY

echo
echo "=== setup complete ==="
echo "Next: launch the training loop, e.g."
echo "  python train/alphazero_loop.py --workdir runs/az1 \\"
echo "    --init train/checkpoints/bc_v3_mc.pt --iterations 5 \\"
echo "    --selfplay-games 2000 --workers \$(nproc) --device cuda"
