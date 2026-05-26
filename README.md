# SkullKI — Skull King engine + AlphaZero-style training

A from-scratch C++ engine for the Skull King card game (base rules, 4 players)
paired with a PyTorch / LibTorch training stack: behaviour-cloned baselines,
PUCT-MCTS with a neural policy prior, a belief network for hidden-information
reasoning, and an AlphaZero-style iteration loop ready to run on GPU.

## Status

| Layer                                     | State                                               |
| ----------------------------------------- | --------------------------------------------------- |
| Game engine (rules, scoring, Tigress)     | 50 test cases, 3.9M assertions                      |
| Random + Heuristic baselines              | Heuristic wins 50% vs Random's 0.2%                 |
| ISMCTS (uniform-determinizer)             | Beats Heuristic clearly at 800 sims                 |
| Python bindings (pybind11)                | Includes Torch classes                              |
| Behaviour-cloned policy/value net         | Beats Heuristic; 10× faster than its ISMCTS teacher |
| TorchScript export + LibTorch C++ infer   | 96k batched obs/s on CPU                            |
| Neural PUCT-MCTS (NN-prior + MC rollout)  | Competitive with ISMCTS at equal sim count          |
| Belief network                            | 45% recall on opponent hands (5× uniform baseline)  |
| Self-play data pipeline                   | Multi-worker, MC-return value targets               |
| AlphaZero iteration orchestrator          | Smoke-tested end-to-end                             |
| vast.ai / Docker deployment               | Dockerfile + bootstrap script                       |

## Repository layout

```
engine/      C++ game engine — rules, state, scoring, observation, encoder.
agents/      Random + Heuristic agents.
search/      ISMCTS (vanilla, uniform determinizer).
nn_torch/    LibTorch-backed inference + PUCT-MCTS + belief sampler.
bindings/    pybind11 module `skullking`.
tools/       CLI utilities (tournament, NN benchmark).
train/       Python training stack (data_gen, selfplay, model, train, export, alphazero_loop).
docker/      Container + vast.ai launch guide.
scripts/     Bootstrap scripts.
```

## Build

Requires CMake ≥ 3.20 and a C++20 compiler. On Windows you must use `clang-cl`
because LibTorch's CMake exports MSVC-style flags; clang in GNU-driver mode
rejects them.

### Windows (clang-cl + LibTorch via installed `pip install torch`)

```powershell
$env:Path = 'C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;' + $env:Path
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
```

### Linux (g++)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j$(nproc)
```

### Run tests

```
./build/engine/tests/sk_tests       --reporter compact
./build/agents/tests/sk_agent_tests --reporter compact
./build/search/tests/sk_search_tests --reporter compact
```

### Run a tournament

```
./build/tools/sk_tournament --games 200 --agents random,random,heuristic,heuristic
```

### Benchmark the neural agent

```
./build/tools/benchmark_nn --model train/checkpoints/bc_v3_mc.scripted.pt --games 80
```

## Training (Python)

After building the bindings (`SK_BUILD_PYTHON=ON`, default), the `skullking`
module lives at `build/python/skullking.<...>.{pyd,so}`.

On Windows you must call `os.add_dll_directory(...)` to the torch lib dir
before importing the module; the training scripts handle this automatically.

```bash
# Quick demo: full game with C++ agents from Python
python bindings/demo.py

# Generate self-play data with ISMCTS teacher
python train/data_gen.py --games 800 --sims 400 --workers 4 --out train/data/v1.npz

# Train policy/value net
python train/train.py --data train/data/v1.npz --epochs 25 --out train/checkpoints/v1.pt

# Export to TorchScript
python train/export.py --ckpt train/checkpoints/v1.pt --out train/checkpoints/v1.scripted.pt

# Evaluate the trained agent
python train/eval.py --games 80 --agents nn,heuristic,heuristic,heuristic \
                     --ckpt train/checkpoints/v1.pt
```

## AlphaZero loop on a GPU host

See [`docker/README.md`](docker/README.md) for the full vast.ai walkthrough.
Short version:

```bash
bash scripts/vastai_setup.sh        # install build deps + build + run smoke tests
python train/alphazero_loop.py \
    --workdir runs/az1 \
    --init train/checkpoints/bc_v3_mc.pt \
    --iterations 10 \
    --selfplay-games 5000 --selfplay-sims 100 \
    --workers $(nproc) \
    --train-epochs 8 --train-batch 4096 \
    --gate-games 200 --gate-winrate 0.55 \
    --device cuda
```

Each iteration writes a `runs/az1/iter_NNN/` folder with the new self-play data,
candidate checkpoint, and eval log; promotion is gated on the candidate
winning ≥ 55% of an N-game seat-rotated tournament against the current best.

## Game rules

Base game only — 4 players, Skull King scoring, no Rascal-from-Rotan scoring,
no advanced expansions (Kraken / Whale / Loot / extended pirate abilities).
Rules PDF (German, official Ravensburger edition) is included for reference.

## Licence

MIT.
