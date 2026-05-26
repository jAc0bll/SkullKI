# Vast.ai deployment guide — Skull King AlphaZero

This directory contains the deployment artefacts for running the training loop
on a Linux GPU instance (typically rented from [vast.ai](https://vast.ai)).

The recommended workflow does **not** rely on Docker — vast.ai instances already
ship the right PyTorch+CUDA stack via a base image. We just bootstrap build deps
and compile the C++ engine on the instance. The `Dockerfile` in this directory
is provided for users who prefer a fully self-contained image; it is otherwise
optional.

---

## Recommended path — direct on a vast.ai instance

1. **Pick an instance**. On vast.ai filter for
   - **Image** `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel` (or a newer 2.x).
   - **GPU** 4× RTX 4090 (or whatever your budget allows; the code scales down).
   - **Disk** ≥ 50 GB for selfplay data and checkpoints.

2. **SSH in** and pull this repository:
   ```bash
   git clone https://github.com/<you>/skullking.git /workspace
   cd /workspace
   ```

3. **Bootstrap the build**:
   ```bash
   bash scripts/vastai_setup.sh
   ```
   This installs cmake/ninja, builds engine + agents + search + nn_torch + the
   Python bindings, and runs the three test suites. Expected runtime: ~5 min.

4. **Verify GPU inference** with a 4-game smoke selfplay run:
   ```bash
   python train/selfplay.py \
       --model train/checkpoints/bc_v3_mc.scripted.pt \
       --games 4 --sims 50 --workers 4 --device cuda \
       --out train/data/gpu_smoke.npz
   ```
   Throughput should be **5-10× higher** than the CPU baseline (≈3-5 games/sec
   per worker vs 0.13 on the dev laptop).

5. **Launch the AlphaZero loop** inside a `tmux` session so it survives
   SSH disconnects:
   ```bash
   tmux new -s az
   python train/alphazero_loop.py \
       --workdir runs/az1 \
       --init train/checkpoints/bc_v3_mc.pt \
       --iterations 10 \
       --selfplay-games 5000 --selfplay-sims 100 \
       --workers $(nproc) \
       --train-epochs 8 --train-batch 4096 --train-lr 1e-3 \
       --gate-games 200 --gate-winrate 0.55 \
       --device cuda
   # Ctrl+B then D to detach. `tmux attach -t az` to reattach.
   ```

   Each iteration writes a folder `runs/az1/iter_NNN/` with the selfplay data,
   candidate checkpoint, and eval result. `runs/az1/best.scripted.pt` is the
   current strongest model.

6. **Pull the trained model back** before terminating the instance:
   ```bash
   # On your local machine:
   scp -P <ssh-port> root@<host>:/workspace/runs/az1/best.scripted.pt ./
   ```

---

## Cost expectations

- vast.ai 4× RTX 4090 typically rents at $1.20–$2.00/hour depending on time of day.
- A single AlphaZero iteration with 5000 selfplay games and 100 sims is roughly
  20–60 minutes on 4× 4090, depending on workers and GPU sharing pattern.
- A 10-iteration run is therefore ≈ 4–10 hours = **$5–$20**.

To stay cheap during early experimentation:
- Start with `--iterations 2 --selfplay-games 1000` to validate the pipeline.
- Once you see a positive `winrate` in the log, scale up.

---

## Docker path (alternative)

Build the self-contained image once and reuse it:

```bash
docker build -f docker/Dockerfile -t skullking:latest .

# Push to a registry for re-use across vast.ai instances:
docker tag skullking:latest <your-dockerhub-user>/skullking:latest
docker push <your-dockerhub-user>/skullking:latest
```

Then on a vast.ai instance you select your custom image instead of the base
PyTorch one, skipping the setup script entirely. Useful if you spin up many
short-lived instances; less convenient for iterative dev because every code
change requires a rebuild + repush.

---

## Files in this directory

- `Dockerfile` — full reproducible image (PyTorch CUDA base + build).
- `README.md` — this document.

Companion files elsewhere in the repo:

- `scripts/vastai_setup.sh` — bootstrap script for a fresh instance.
- `train/selfplay.py` — selfplay data generator (uses C++ NeuralMCTSAgent).
- `train/alphazero_loop.py` — iteration orchestrator with eval gate.
- `train/train.py` — supervised training on selfplay targets.
- `train/export.py` / `train/export_belief.py` — PyTorch → TorchScript export.
