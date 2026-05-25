"""DouZero trainer — single- or multi-GPU via DDP.

Each rank runs its own ``CurriculumRunner`` with its own replay buffer.
Gradients are synced across ranks by ``DistributedDataParallel`` on every
backward pass. Only rank 0 evaluates, prints, and saves checkpoints.

Single-GPU mode:  python -m skull_king.training.douzero.train --config <yaml>
Multi-GPU mode:   torchrun --nproc_per_node=N -m skull_king.training.douzero.train --config <yaml>
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
from skull_king.training.douzero.buffers import MCReplayBuffer
from skull_king.training.douzero.networks import DouZeroQNet
from skull_king.training.douzero.opponents import (
    CurriculumSchedule,
    LeaguePool,
    assign_opponents,
)
from skull_king.training.douzero.runner import CurriculumRunner
from skull_king.training.rebel.public_belief_state import pbs_encoding_size

if TYPE_CHECKING:
    from skull_king.training.douzero.train import DouZeroConfig


# ---------------------------------------------------------------------------
# Distributed helpers
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
    """Init NCCL process group + return this rank's device."""
    if _is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_local_rank())
        return torch.device(f"cuda:{_local_rank()}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DouZeroTrainer:
    def __init__(self, cfg: "DouZeroConfig") -> None:
        self.cfg = cfg
        self.device = _setup_ddp()
        self.rank = _local_rank()
        self.world_size = _world_size()

        # TF32 matmul + cuDNN autotuner for big speedups on modern GPUs
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        n = cfg.n_players
        pbs_size = pbs_encoding_size(n)

        # Build Q-net on this rank's device, keep an eager handle for eval
        self.q_net = DouZeroQNet(n, hidden=tuple(cfg.hidden)).to(self.device)
        self._q_eager = self.q_net  # used for eval (batch=1) and league snapshots

        # Wrap in DDP if multi-GPU
        if _is_distributed():
            self.q_net = DDP(
                self.q_net,
                device_ids=[self.rank] if self.device.type == "cuda" else None,
                output_device=self.rank if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )

        # Optimizer + AMP scaler
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Replay buffer (per rank)
        self.buf = MCReplayBuffer(
            cfg.buffer_capacity,
            pbs_size,
            ACTION_SPACE_SIZE,
            seed=cfg.seed + self.rank,
        )

        # Curriculum components
        self.schedule = CurriculumSchedule(
            self_start=cfg.self_start, self_end=cfg.self_end,
            league_start=cfg.league_start, league_end=cfg.league_end,
            heuristic_start=cfg.heuristic_start, heuristic_end=cfg.heuristic_end,
            random_start=cfg.random_start, random_end=cfg.random_end,
        )
        self.league = LeaguePool(capacity=cfg.league_capacity)

        # Per-rank runner; seed offset by rank to avoid duplicate trajectories
        self.runner = CurriculumRunner(
            n_envs=cfg.n_envs,
            n_players=cfg.n_players,
            device=self.device,
            seed=cfg.seed + self.rank * 7919,  # prime offset
        )

        if _is_main():
            os.makedirs(cfg.model_dir, exist_ok=True)

        # Resume support
        if cfg.resume_from:
            ckpt = torch.load(f"{cfg.resume_from}.pt", map_location=self.device)
            self._unwrapped().load_state_dict(ckpt)
            if _is_main():
                print(f"Resumed from {cfg.resume_from}", flush=True)

        # Optional compile (single-GPU only). torch.compile + DDP under
        # reduce-overhead mode is brittle; the DDP gradient sync overhead
        # dominates anyway on multi-GPU so the compile gain is marginal.
        # _q_eager always stays uncompiled for batch-1 eval and league use.
        if cfg.compile_nets and self.device.type == "cuda" and not _is_distributed():
            self.q_net = torch.compile(self.q_net, mode="reduce-overhead", dynamic=True)

    # ------------------------------------------------------------------

    def _unwrapped(self) -> DouZeroQNet:
        """Return the underlying DouZeroQNet (unwrap DDP / compile if needed)."""
        net = self.q_net
        if isinstance(net, DDP):
            net = net.module
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        return net

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + self.rank)

        if _is_main():
            print(f"\n{'='*62}")
            print(f"  DouZero — {cfg.run_name}")
            print(f"  {cfg.n_iterations} iters × {cfg.collect_steps:,} dec/iter")
            print(f"  world_size={self.world_size}  n_envs/rank={cfg.n_envs}")
            print(f"  ε: {cfg.epsilon_start} → {cfg.epsilon_end} over {cfg.epsilon_decay} iters")
            print(f"  device: {self.device}  GPU: {torch.cuda.get_device_name(self.rank) if self.device.type=='cuda' else 'n/a'}")
            print(f"{'='*62}\n")

        try:
            for t in range(cfg.start_iter, cfg.n_iterations + 1):
                t0 = time.time()

                # Anneal exploration
                epsilon = max(
                    cfg.epsilon_end,
                    cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * t / max(1, cfg.epsilon_decay),
                )

                # Refresh per-env opponent assignments
                opp_table = assign_opponents(
                    n_envs=cfg.n_envs,
                    n_players=cfg.n_players,
                    schedule=self.schedule,
                    league_size=len(self.league),
                    t=t,
                    total_t=cfg.n_iterations,
                    rng=rng,
                )
                self.runner.set_opponent_assignments(opp_table)

                # Collect
                self.runner.collect(
                    live_q=self._unwrapped(),  # eager net for in-loop inference
                    league=self.league,
                    buf=self.buf,
                    epsilon=epsilon,
                    n_decisions=cfg.collect_steps,
                )
                t_collect = time.time() - t0

                # Train
                t1 = time.time()
                loss = self._train_step()
                t_train = time.time() - t1

                # League snapshot
                if t % cfg.league_snapshot_every == 0:
                    if _is_main():
                        self.league.snapshot(self._unwrapped())
                    # Broadcast snapshot to other ranks
                    if _is_distributed():
                        dist.barrier()

                if _is_main():
                    elapsed = time.time() - t0
                    print(
                        f"iter {t:5d}/{cfg.n_iterations}"
                        f"  loss={loss:.4f}"
                        f"  buf={len(self.buf):,}"
                        f"  ε={epsilon:.4f}"
                        f"  league={len(self.league)}"
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
    # Q-learning step
    # ------------------------------------------------------------------

    def _train_step(self) -> float:
        cfg = self.cfg
        if len(self.buf) < cfg.batch_size:
            return 0.0

        steps = min(cfg.train_steps, max(1, len(self.buf) // (cfg.batch_size // 4)))
        self.q_net.train()
        total = 0.0
        amp_ctx = torch.amp.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda"),
        )

        # Prefetch first batch
        enc, mask, action, returns = self.buf.sample(cfg.batch_size)
        enc_t  = self._to_device(enc, torch.float32)
        mask_t = self._to_device(mask, torch.bool)
        act_t  = self._to_device(action.astype(np.int64), torch.int64)
        ret_t  = self._to_device(returns, torch.float32)

        for k in range(steps):
            if k + 1 < steps:
                next_enc, next_mask, next_action, next_returns = self.buf.sample(cfg.batch_size)

            with amp_ctx:
                q_all = self.q_net(enc_t, mask_t)
                q_taken = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_taken, ret_t)

            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            total += loss.item()

            if k + 1 < steps:
                enc_t  = self._to_device(next_enc, torch.float32)
                mask_t = self._to_device(next_mask, torch.bool)
                act_t  = self._to_device(next_action.astype(np.int64), torch.int64)
                ret_t  = self._to_device(next_returns, torch.float32)

        self.q_net.eval()
        return total / steps

    def _to_device(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """Pin → async non-blocking transfer to GPU."""
        t = torch.from_numpy(arr).to(dtype=dtype)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Eval + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int) -> None:
        from skull_king.training.douzero.agent import DouZeroAgent
        n = self.cfg.n_players
        # Use eager (uncompiled) Q-net — tournament has batch=1 per decision,
        # which would cause torch.compile recompile storms.
        agent = DouZeroAgent(self._q_eager, n_players=n, name="DouZero", device=self.device)
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=200)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=200)
        wr_r = r_r.win_rates().get("DouZero", 0.0)
        wr_h = r_h.win_rates().get("DouZero", 0.0)
        av_r = r_r.avg_scores().get("DouZero", 0.0)
        av_h = r_h.avg_scores().get("DouZero", 0.0)
        print(
            f"  [EVAL iter={t}]"
            f"  vs_random={wr_r:.1%} ({av_r:+.0f})"
            f"  vs_heuristic={wr_h:.1%} ({av_h:+.0f})",
            flush=True,
        )

    def _save(self, base_path: str) -> None:
        torch.save(self._unwrapped().state_dict(), f"{base_path}.pt")
