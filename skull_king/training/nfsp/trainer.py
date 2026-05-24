"""NFSP trainer — vectorized self-play + GPU network training."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.rebel.public_belief_state import pbs_encoding_size
from skull_king.training.nfsp.buffers import RLBuffer, SLBuffer
from skull_king.training.nfsp.networks import NfspAvgNet, NfspQNet
from skull_king.training.nfsp.runner import VectorizedRunner

if TYPE_CHECKING:
    from skull_king.training.nfsp.train import NfspConfig


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except (ImportError, Exception):
        pass
    return torch.device("cpu")


class NfspTrainer:
    def __init__(self, cfg: NfspConfig) -> None:
        self.cfg = cfg
        self.device = _pick_device()

        n = cfg.n_players
        pbs_size = pbs_encoding_size(n)

        self.q_net = NfspQNet(n, hidden=tuple(cfg.hidden)).to(self.device)
        self.avg_net = NfspAvgNet(n, hidden=tuple(cfg.hidden)).to(self.device)
        self.q_net.eval()
        self.avg_net.eval()

        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=cfg.rl_lr)
        self.avg_opt = torch.optim.Adam(self.avg_net.parameters(), lr=cfg.sl_lr)

        self.rl_buf = RLBuffer(cfg.rl_buffer_capacity, pbs_size, ACTION_SPACE_SIZE, seed=cfg.seed)
        self.sl_buf = SLBuffer(cfg.sl_buffer_capacity, pbs_size, ACTION_SPACE_SIZE, seed=cfg.seed + 1)

        self._scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.runner = VectorizedRunner(cfg.n_envs, cfg.n_players, self.device, seed=cfg.seed + 2)

        os.makedirs(cfg.model_dir, exist_ok=True)

        if cfg.resume_from:
            self.q_net.load_state_dict(
                torch.load(f"{cfg.resume_from}_q.pt", map_location=self.device)
            )
            self.avg_net.load_state_dict(
                torch.load(f"{cfg.resume_from}_avg.pt", map_location=self.device)
            )
            print(f"Resumed from {cfg.resume_from}", flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg

        print(f"\n{'='*62}")
        print(f"  NFSP  —  {cfg.run_name}")
        print(f"  {cfg.n_iterations} iters  ×  {cfg.collect_steps:,} decisions/iter")
        print(f"  {cfg.n_envs} parallel envs  η={cfg.eta}  ε={cfg.epsilon_start}→{cfg.epsilon_end}")
        print(f"  device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*62}\n")

        for t in range(cfg.start_iter, cfg.n_iterations + 1):
            cfg.epsilon = max(
                cfg.epsilon_end,
                cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * t / cfg.epsilon_decay,
            )

            t0 = time.time()
            self.runner.collect(
                self.q_net, self.avg_net,
                self.rl_buf, self.sl_buf,
                cfg, cfg.collect_steps,
            )
            t_collect = time.time() - t0

            t1 = time.time()
            rl_loss = self._train_q()
            sl_loss = self._train_avg()
            t_train = time.time() - t1

            elapsed = time.time() - t0
            print(
                f"iter {t:4d}/{cfg.n_iterations}"
                f"  rl={rl_loss:.4f}  sl={sl_loss:.4f}"
                f"  rl_buf={len(self.rl_buf):,}  sl_buf={len(self.sl_buf):,}"
                f"  ε={cfg.epsilon:.4f}"
                f"  collect={t_collect:.1f}s train={t_train:.1f}s total={elapsed:.1f}s",
                flush=True,
            )

            if t % cfg.eval_every == 0:
                self._evaluate(t)

            if t % cfg.checkpoint_every == 0:
                path = os.path.join(cfg.model_dir, f"{cfg.run_name}_iter{t}")
                self._save(path)
                print(f"  [Checkpoint] → {path}_{{q,avg}}.pt", flush=True)

        final = os.path.join(cfg.model_dir, f"{cfg.run_name}_final")
        self._save(final)
        print(f"\nDone. → {final}_{{q,avg}}.pt", flush=True)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _train_q(self) -> float:
        cfg = self.cfg
        if len(self.rl_buf) < cfg.rl_batch_size:
            return 0.0

        steps = min(cfg.rl_train_steps, max(1, len(self.rl_buf) // (cfg.rl_batch_size // 4)))
        self.q_net.train()
        total = 0.0
        amp_ctx = torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda"))

        for _ in range(steps):
            enc, mask, action, returns = self.rl_buf.sample(cfg.rl_batch_size)
            enc_t = torch.from_numpy(enc).float().to(self.device)
            mask_t = torch.from_numpy(mask).bool().to(self.device)
            act_t = torch.from_numpy(action.astype(np.int64)).to(self.device)
            ret_t = torch.from_numpy(returns).float().to(self.device)

            with amp_ctx:
                q_all = self.q_net(enc_t, mask_t)
                q_taken = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_taken, ret_t)

            self.q_opt.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.q_opt)
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self._scaler.step(self.q_opt)
            self._scaler.update()
            total += loss.item()

        self.q_net.eval()
        return total / steps

    def _train_avg(self) -> float:
        cfg = self.cfg
        if len(self.sl_buf) < cfg.sl_batch_size:
            return 0.0

        steps = min(cfg.sl_train_steps, max(1, len(self.sl_buf) // (cfg.sl_batch_size // 4)))
        self.avg_net.train()
        total = 0.0
        amp_ctx = torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda"))

        for _ in range(steps):
            enc, mask, action = self.sl_buf.sample(cfg.sl_batch_size)
            enc_t = torch.from_numpy(enc).float().to(self.device)
            mask_t = torch.from_numpy(mask).bool().to(self.device)
            act_t = torch.from_numpy(action).to(self.device)

            with amp_ctx:
                log_probs = self.avg_net(enc_t, mask_t)
                loss = F.nll_loss(log_probs, act_t)

            self.avg_opt.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.avg_opt)
            nn.utils.clip_grad_norm_(self.avg_net.parameters(), 1.0)
            self._scaler.step(self.avg_opt)
            self._scaler.update()
            total += loss.item()

        self.avg_net.eval()
        return total / steps

    # ------------------------------------------------------------------
    # Evaluation + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int) -> None:
        from skull_king.training.nfsp.agent import NfspAgent
        n = self.cfg.n_players
        agent = NfspAgent(self.avg_net, n_players=n, name="NFSP", device=self.device)
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=200)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=200)
        wr_r = r_r.win_rates().get("NFSP", 0.0)
        wr_h = r_h.win_rates().get("NFSP", 0.0)
        avg_r = r_r.avg_scores().get("NFSP", 0.0)
        avg_h = r_h.avg_scores().get("NFSP", 0.0)
        print(
            f"  [EVAL iter={t}]"
            f"  vs_random={wr_r:.1%} ({avg_r:+.0f})"
            f"  vs_heuristic={wr_h:.1%} ({avg_h:+.0f})",
            flush=True,
        )

    def _save(self, base_path: str) -> None:
        torch.save(self.q_net.state_dict(), f"{base_path}_q.pt")
        torch.save(self.avg_net.state_dict(), f"{base_path}_avg.pt")
