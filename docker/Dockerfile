# Build environment for Skull King engine + AlphaZero training on vast.ai
# (or any CUDA-equipped Linux host).
#
# Base: PyTorch's official CUDA-enabled devel image. It ships with python,
# pip, torch (CUDA build), nvcc and a complete C++ toolchain. We layer
# build tools (cmake, ninja) on top.
#
# Build:
#     cd <repo>
#     docker build -f docker/Dockerfile -t skullking:latest .
#
# Run interactively with GPUs:
#     docker run --gpus all -it -v $(pwd):/workspace skullking:latest
#
# Run a one-shot AlphaZero iteration:
#     docker run --gpus all -v $(pwd):/workspace skullking:latest \
#         python train/alphazero_loop.py --workdir runs/az1 \
#             --init train/checkpoints/bc_v3_mc.pt --iterations 3 \
#             --selfplay-games 1000 --workers 16

ARG TORCH_TAG=2.7.0-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:${TORCH_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates htop \
 && rm -rf /var/lib/apt/lists/*

# Sanity: torch picks up CUDA, numpy is present.
RUN python -c "import torch, numpy; \
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
          'numpy', numpy.__version__)"

WORKDIR /workspace

# The image builds the project at COPY time. Mount-as-volume users will rebuild
# on demand; that's also fine because the build cache lives under /workspace/build.
COPY . /workspace

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DSK_BUILD_TESTS=ON \
        -DSK_BUILD_PYTHON=ON \
        -DSK_BUILD_TORCH=ON \
 && cmake --build build -j$(nproc)

# Smoke tests at image-build time (fail fast on broken images).
RUN ./build/engine/tests/sk_tests       --reporter compact \
 && ./build/agents/tests/sk_agent_tests --reporter compact \
 && ./build/search/tests/sk_search_tests --reporter compact

# Default landing shell.
CMD ["/bin/bash"]
