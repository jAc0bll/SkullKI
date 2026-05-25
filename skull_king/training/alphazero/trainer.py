"""AlphaZero trainer — single GPU or DDP multi-GPU.

Single GPU:
    python -m skull_king.training.alphazero.train --config configs/alphazero/alphazero_v1.yaml

Multi-GPU (4× RTX 4090):
    torchrun --nproc_per_node=4 --master_port=29500 \\
        -m skull_king.training.alphazero.train --config configs/alphazero/alphazero_v1.yaml

Each rank runs its own self-play loop + replay buffer. Gradients are
synced across ranks during the training phase via DistributedDataParallel.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.alphazero.buffers import AZReplayBuffer
from skull_king.training.alphazero.networks import AlphaZeroNet
from skull_king.training.alphazero.runner import AlphaZeroRunner
from skull_king.training.rebel.public_belief_state import pbs_encoding_size

if TYPE_CHECKING:
    from skull_king.training.alphazero.train import AlphaZeroConfig


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main() -> bool:
    return _local_rank() == 0


def _setup_ddp() -> torch.device:
    if _is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_local_rank())
        return torch.device(f"cuda:{_local_rank()}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AlphaZeroTrainer:
    def __init__(self, cfg: "AlphaZeroConfig") -> None:
        self.cfg = cfg
        self.device = _setup_ddp()
        self.rank = _local_rank()
        self.world_size = _world_size()

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        n = cfg.n_players
        pbs_size = pbs_encoding_size(n)

        self.network = AlphaZeroNet(n, hidden=tuple(cfg.hidden), value_hidden=cfg.value_hidden).to(self.device)
        self._net_eager = self.network  # eager handle for self-play + eval

        if _is_distributed():
            self.network = DDP(
                self.network,
                device_ids=[self.rank] if self.device.type == "cuda" else None,
                output_device=self.rank if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )

        self.opt = torch.optim.Adam(self.network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.buf = AZReplayBuffer(
            cfg.buffer_capacity,
            pbs_size,
            ACTION_SPACE_SIZE,
            seed=cfg.seed + self.rank,
        )

        self.runner = AlphaZeroRunner(
            n_envs=cfg.n_envs,
            n_players=cfg.n_players,
            device=self.device,
            seed=cfg.seed + self.rank * 7919,
        )

        if _is_main():
            os.makedirs(cfg.model_dir, exist_ok=True)

        if cfg.resume_from:
            sd = torch.load(f"{cfg.resume_from}.pt", map_location=self.device)
            self._unwrapped().load_state_dict(sd)
            if _is_main():
                print(f"Resumed from {cfg.resume_from}", flush=True)

    # ------------------------------------------------------------------

    def _unwrapped(self) -> AlphaZeroNet:
        net = self.network
        if isinstance(net, DDP):
            net = net.module
        return net

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        if _is_main():
            print(f"\n{'='*64}")
            print(f"  AlphaZero — {cfg.run_name}")
            print(f"  {cfg.n_iterations} iters × {cfg.collect_decisions:,} dec/iter (per rank)")
            print(f"  world_size={self.world_size}  n_envs/rank={cfg.n_envs}")
            print(f"  MCTS sims/move: {cfg.n_simulations}  c_puct={cfg.c_puct}")
            print(f"  device: {self.device}  GPU: {torch.cuda.get_device_name(self.rank) if self.device.type=='cuda' else 'n/a'}")
            print(f"{'='*64}\n")

        try:
            for t in range(cfg.start_iter, cfg.n_iterations + 1):
                t0 = time.time()

                temperature = (
                    cfg.temperature_initial
                    if t <= cfg.temperature_drop_iter
                    else cfg.temperature_final
                )

                self.runner.collect(
                    network=self._net_eager,
                    buf=self.buf,
                    n_simulations=cfg.n_simulations,
                    c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_eps=cfg.dirichlet_eps,
                    temperature=temperature,
                    max_decisions=cfg.collect_decisions,
                )
                t_collect = time.time() - t0

                t1 = time.time()
                losses = self._train_step()
                t_train = time.time() - t1

                if _is_main():
                    elapsed = time.time() - t0
                    print(
                        f"iter {t:5d}/{cfg.n_iterations}"
                        f"  p_loss={losses['policy']:.4f}  v_loss={losses['value']:.4f}"
                        f"  buf={len(self.buf):,}  τ={temperature:.2f}"
                        f"  collect={t_collect:.1f}s train={t_train:.1f}s total={elapsed:.1f}s",
                        flush=True,
                    )

                if t % cfg.eval_every == 0 and _is_main():
                    self._evaluate(t)

                if t % cfg.checkpoint_every == 0 and _is_main():
                    path = os.path.join(cfg.model_dir, f"{cfg.run_name}_iter{t}")
                    self._save(path)
                    print(f"  [Checkpoint] → {path}.pt", flush=True)

                if _is_distributed():
                    dist.barrier()
        finally:
            if _is_main():
                final = os.path.join(cfg.model_dir, f"{cfg.run_name}_final")
                self._save(final)
                print(f"\nDone. → {final}.pt", flush=True)
            _cleanup_ddp()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float]:
        cfg = self.cfg
        if len(self.buf) < cfg.batch_size:
            return {"policy": 0.0, "value": 0.0}

        steps = min(cfg.train_steps, max(1, len(self.buf) // (cfg.batch_size // 4)))
        self.network.train()
        p_total = v_total = 0.0
        amp_ctx = torch.amp.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda"),
        )

        enc, mask, pi, val = self.buf.sample(cfg.batch_size)
        enc_t  = self._to_device(enc, torch.float32)
        mask_t = self._to_device(mask, torch.bool)
        pi_t   = self._to_device(pi, torch.float32)
        val_t  = self._to_device(val, torch.float32)

        for k in range(steps):
            if k + 1 < steps:
                ne, nm, npi, nval = self.buf.sample(cfg.batch_size)

            with amp_ctx:
                log_probs, value_pred = self.network(enc_t, mask_t)
                # Policy loss: cross-entropy with MCTS target (masked positions
                # already have log_probs ≈ -inf so contribute 0 because π_target ≈ 0 there).
                policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                value_loss  = F.mse_loss(value_pred, val_t)
                loss = policy_loss + cfg.value_loss_weight * value_loss

            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            p_total += policy_loss.item()
            v_total += value_loss.item()

            if k + 1 < steps:
                enc_t  = self._to_device(ne, torch.float32)
                mask_t = self._to_device(nm, torch.bool)
                pi_t   = self._to_device(npi, torch.float32)
                val_t  = self._to_device(nval, torch.float32)

        self.network.eval()
        return {"policy": p_total / steps, "value": v_total / steps}

    def _to_device(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        t = torch.from_numpy(arr).to(dtype=dtype)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Evaluation + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int) -> None:
        """Two-tier eval:
           - Every ``eval_every`` iters: fast policy-only eval (no MCTS).
             Cheap — runs ~1k decisions through batch-1 forward passes.
           - Every ``mcts_eval_every`` iters: full MCTS-augmented eval.
             Slower, gives the realistic tournament strength.
        """
        from skull_king.training.alphazero.agent import AlphaZeroAgent
        cfg = self.cfg
        n = cfg.n_players
        tournament = TournamentRunner(seed=999)

        def _eval_with(use_mcts: bool, n_games: int, tag: str) -> None:
            agent = AlphaZeroAgent(
                self._net_eager,
                n_players=n,
                name="AlphaZero",
                device=self.device,
                n_simulations=cfg.eval_simulations if use_mcts else 0,
                c_puct=cfg.c_puct,
                use_mcts=use_mcts,
            )
            r_r = tournament.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=n_games)
            r_h = tournament.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=n_games)
            wr_r = r_r.win_rates().get("AlphaZero", 0.0)
            wr_h = r_h.win_rates().get("AlphaZero", 0.0)
            av_r = r_r.avg_scores().get("AlphaZero", 0.0)
            av_h = r_h.avg_scores().get("AlphaZero", 0.0)
            print(
                f"  [EVAL iter={t} {tag}]"
                f"  vs_random={wr_r:.1%} ({av_r:+.0f})"
                f"  vs_heuristic={wr_h:.1%} ({av_h:+.0f})",
                flush=True,
            )

        # Always run the fast policy eval
        _eval_with(use_mcts=False, n_games=cfg.eval_games_fast, tag="policy")

        # MCTS eval only on a slower cadence
        if t % cfg.mcts_eval_every == 0:
            _eval_with(use_mcts=True, n_games=cfg.eval_games_mcts, tag=f"mcts{cfg.eval_simulations}")

    def _save(self, base_path: str) -> None:
        torch.save(self._unwrapped().state_dict(), f"{base_path}.pt")
