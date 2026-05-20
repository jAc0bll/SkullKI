"""Deep CFR training loop."""
from __future__ import annotations

import multiprocessing
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.cfr.buffers import AdvantageBuffer, StrategyBuffer
from skull_king.training.cfr.networks import AdvantageNet, StrategyNet
from skull_king.training.cfr.traversal import worker_init, worker_task

if TYPE_CHECKING:
    from skull_king.training.cfr.train import CFRConfig

_console = Console(highlight=False)


class DeepCFRTrainer:
    """Orchestrates iterative Deep CFR training.

    Each iteration:
      1. Run ``traversals_per_player × n_players`` game traversals in parallel.
         Workers use cached network weights; results are collected as training
         samples for the advantage and strategy buffers.
      2. Train AdvantageNet  (MSE on taken-action regret targets).
      3. Train StrategyNet   (cross-entropy on regret-matched strategy targets).
      4. Evaluate and checkpoint periodically.
    """

    def __init__(self, cfg: "CFRConfig") -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = tuple(cfg.net_hidden)
        self.adv_net = AdvantageNet(hidden=hidden).to(self.device)
        self.strat_net = StrategyNet(hidden=hidden).to(self.device)
        self.adv_net.eval()
        self.strat_net.eval()

        self.adv_opt = torch.optim.Adam(self.adv_net.parameters(), lr=cfg.adv_lr)
        self.strat_opt = torch.optim.Adam(self.strat_net.parameters(), lr=cfg.strat_lr)

        self.adv_buf = AdvantageBuffer(capacity=cfg.adv_buffer_capacity)
        self.strat_buf = StrategyBuffer(capacity=cfg.strat_buffer_capacity)

        self._rng = np.random.default_rng(cfg.env_seed)
        os.makedirs(cfg.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        total_traversals = cfg.traversals_per_player * cfg.n_players
        _console.print(f"\n{'='*66}")
        _console.print(f"  Deep CFR  -  {cfg.run_name}")
        _console.print(
            f"  {cfg.n_cfr_iterations} iters  x  {total_traversals} traversals/iter"
            f"  x  workers={cfg.num_workers}"
        )
        _console.print(f"  device: {self.device}")
        _console.print(f"{'='*66}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[metrics]}"),
            TimeRemainingColumn(),
            console=_console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                "CFR Training",
                total=cfg.n_cfr_iterations,
                metrics="",
            )

            for t in range(1, cfg.n_cfr_iterations + 1):
                t0 = time.time()

                # Deep CFR paper §4: fresh advantage net each iteration so stale
                # regret estimates from old strategies don't corrupt training.
                self._reset_adv()
                self._collect(t)
                adv_loss = self._train_adv()
                strat_loss = self._train_strat()

                elapsed = time.time() - t0
                progress.update(
                    task,
                    advance=1,
                    metrics=(
                        f"adv={adv_loss:.4f}  strat={strat_loss:.4f}"
                        f"  buf={len(self.adv_buf):,}  {elapsed:.1f}s/it"
                    ),
                )

                if t % cfg.eval_every_n_iters == 0:
                    progress.print("")
                    self._evaluate(t, progress)

                if t % cfg.checkpoint_every_n_iters == 0:
                    path = os.path.join(cfg.model_dir, f"{cfg.run_name}_iter{t}")
                    self._save(path)
                    progress.print(f"  [Checkpoint] -> {path}_{{adv,strat}}.pt")

        final = os.path.join(cfg.model_dir, "cfr_final")
        self._save(final)
        _console.print(f"\nTraining complete. Saved -> {final}_{{adv,strat}}.pt")

    # ------------------------------------------------------------------
    # Advantage buffer cycle (per-iteration refresh)
    # ------------------------------------------------------------------

    def _reset_adv(self) -> None:
        """Clear the advantage buffer and reset the optimizer each iteration.

        Buffer clear: removes stale regret targets computed under old strategies.
        Optimizer reset: drops accumulated momentum so the warm-started weights
        adapt cleanly to the new samples without gradient artifacts.
        Network weights are kept (warm start) — re-initialising to random and
        training for only a few dozen steps leaves the net near-random, which
        makes all strategies uniform and collapses the strategy net.
        """
        self.adv_opt = torch.optim.Adam(self.adv_net.parameters(), lr=self.cfg.adv_lr)
        self.adv_buf.clear()

    # ------------------------------------------------------------------
    # Traversal collection
    # ------------------------------------------------------------------

    def _collect(self, iteration: int) -> None:
        cfg = self.cfg
        adv_weights = {k: v.cpu() for k, v in self.adv_net.state_dict().items()}
        strat_weights = {k: v.cpu() for k, v in self.strat_net.state_dict().items()}

        # Each task: (traverser_player, seed, n_players)
        tasks = [
            (
                i % cfg.n_players,
                int(self._rng.integers(0, 2**31)),
                cfg.n_players,
            )
            for i in range(cfg.traversals_per_player * cfg.n_players)
        ]

        if cfg.num_workers <= 1:
            # Single-process fallback (Windows dev / smoke test)
            worker_init(adv_weights, strat_weights)
            for task in tasks:
                adv_s, strat_s = worker_task(task)
                self.adv_buf.add_batch(adv_s)
                self.strat_buf.add_batch(strat_s)
        else:
            # Parallel: weights are sent ONCE per worker via initializer.
            # fork is instant on Linux (no re-import); spawn is required on Windows.
            import sys
            ctx = multiprocessing.get_context("spawn" if sys.platform == "win32" else "fork")
            with ctx.Pool(
                processes=cfg.num_workers,
                initializer=worker_init,
                initargs=(adv_weights, strat_weights),
            ) as pool:
                for adv_s, strat_s in pool.imap_unordered(
                    worker_task, tasks, chunksize=4
                ):
                    self.adv_buf.add_batch(adv_s)
                    self.strat_buf.add_batch(strat_s)

    # ------------------------------------------------------------------
    # Network training
    # ------------------------------------------------------------------

    def _train_adv(self) -> float:
        cfg = self.cfg
        if len(self.adv_buf) < cfg.adv_batch_size:
            return 0.0
        self.adv_net.train()
        total = 0.0
        for _ in range(cfg.adv_train_epochs):
            obs, _, targets, actions = self.adv_buf.sample(cfg.adv_batch_size)
            obs_t = torch.FloatTensor(obs).to(self.device)
            targets_t = torch.FloatTensor(targets).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)

            pred = self.adv_net(obs_t)                              # [B, 82]
            taken_pred = pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            taken_tgt = targets_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(taken_pred, taken_tgt)

            self.adv_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), 1.0)
            self.adv_opt.step()
            total += loss.item()
        self.adv_net.eval()
        return total / cfg.adv_train_epochs

    def _train_strat(self) -> float:
        cfg = self.cfg
        if len(self.strat_buf) < cfg.strat_batch_size:
            return 0.0
        self.strat_net.train()
        total = 0.0
        for _ in range(cfg.strat_train_epochs):
            obs, masks, strategies = self.strat_buf.sample(cfg.strat_batch_size)
            obs_t = torch.FloatTensor(obs).to(self.device)
            mask_t = torch.BoolTensor(masks).to(self.device)
            strat_t = torch.FloatTensor(strategies).to(self.device)

            logits = self.strat_net(obs_t)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            log_probs = torch.log_softmax(logits, dim=-1)
            # Cross-entropy: −Σ target * log_prob (only over legal actions).
            # nan_to_num handles 0 * -inf for illegal-action slots (IEEE 754).
            loss = -(strat_t * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

            self.strat_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strat_net.parameters(), 1.0)
            self.strat_opt.step()
            total += loss.item()
        self.strat_net.eval()
        return total / cfg.strat_train_epochs

    # ------------------------------------------------------------------
    # Evaluation + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int, progress: Progress | None = None) -> None:
        from skull_king.training.cfr.agent import CFRAgent
        n = self.cfg.n_players
        agent = CFRAgent(self.strat_net, n_players=n, name="CFR")
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=50)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=50)
        wr_r = r_r.win_rates().get("CFR", 0.0)
        wr_h = r_h.win_rates().get("CFR", 0.0)
        avg_h = r_h.avg_scores().get("CFR", 0.0)
        msg = (
            f"  [Eval iter={t}]  vs_random={wr_r:.1%}  "
            f"vs_heuristic={wr_h:.1%}  avg_score_H={avg_h:+.0f}"
        )
        if progress is not None:
            progress.print(msg)
        else:
            _console.print(msg)

    def _save(self, base_path: str) -> None:
        torch.save(self.adv_net.state_dict(), f"{base_path}_adv.pt")
        torch.save(self.strat_net.state_dict(), f"{base_path}_strat.pt")
