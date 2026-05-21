"""Deep CFR training loop."""
from __future__ import annotations

import multiprocessing
import os
import sys
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
from skull_king.training.cfr.traversal import worker_init, worker_task, worker_update_nets

if TYPE_CHECKING:
    from skull_king.training.cfr.train import CFRConfig

_console = Console(highlight=False)


def _pick_device() -> torch.device:
    """Prefer CUDA > DirectML (AMD/Intel on Windows) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore[import]
        dev = torch_directml.device()
        _console.print(f"  [GPU] Using DirectML device: {torch_directml.device_name(0)}")
        return dev
    except (ImportError, Exception):
        pass
    return torch.device("cpu")


class DeepCFRTrainer:
    """Orchestrates iterative Deep CFR training.

    Each iteration:
      1. Broadcast updated network weights to the persistent worker pool.
      2. Run ``traversals_per_player × n_players`` game traversals in parallel.
         Advantage samples accumulate across iterations (never cleared) so the
         advantage net always has a rich training set.  Strategy samples also
         accumulate (standard Deep CFR average-strategy approximation).
      3. Warm-start the AdvantageNet optimizer (reset momentum only) and train
         for ``adv_train_steps`` gradient steps.
      4. Train StrategyNet for ``strat_train_steps`` gradient steps.
      5. Evaluate and checkpoint periodically.

    Why accumulated advantage buffer?
        Deep CFR requires the advantage net to fit the regret function for the
        current strategy.  With only ~50-200k new samples per iteration the net
        can barely generalise.  Accumulating samples gives millions of training
        points by iteration ~30, enabling stable advantage estimates.  Samples
        from older iterations are slightly stale but still useful: strategies
        change slowly between consecutive iterations, so the advantage landscape
        does not shift drastically.  The circular buffer automatically evicts the
        oldest data once capacity is reached.

    Why ``adv_train_steps`` instead of epochs?
        Epoch-based training ties cost to buffer size.  Fixed steps give
        predictable per-iteration wall-clock time and let you tune the
        training budget independently of the buffer size.
    """

    def __init__(self, cfg: "CFRConfig") -> None:
        self.cfg = cfg
        self.device = _pick_device()

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

        # On Linux (fork): pool is recreated each iteration with current weights
        # passed via the initializer.  fork is copy-on-write so this is nearly
        # free — no IPC overhead for weight transfer.
        # On Windows (spawn): pool is created once and weights are broadcast via
        # worker_update_nets() to avoid expensive per-iteration process creation.
        self._is_windows = sys.platform == "win32"
        self._ctx = multiprocessing.get_context("spawn" if self._is_windows else "fork")

        persistent_pool = None
        if self._is_windows and cfg.num_workers > 1:
            init_adv = {k: v.cpu() for k, v in self.adv_net.state_dict().items()}
            init_strat = {k: v.cpu() for k, v in self.strat_net.state_dict().items()}
            persistent_pool = self._ctx.Pool(
                processes=cfg.num_workers,
                initializer=worker_init,
                initargs=(init_adv, init_strat),
            )

        try:
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

                    self._reset_adv()
                    self._collect(t, persistent_pool)
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
        finally:
            if persistent_pool is not None:
                persistent_pool.close()
                persistent_pool.join()

        final = os.path.join(cfg.model_dir, "cfr_final")
        self._save(final)
        _console.print(f"\nTraining complete. Saved -> {final}_{{adv,strat}}.pt")

    # ------------------------------------------------------------------
    # Advantage net reset (warm-start only — buffer is kept)
    # ------------------------------------------------------------------

    def _reset_adv(self) -> None:
        """Reset optimizer momentum; keep network weights and advantage buffer.

        Optimizer reset: drops accumulated Adam momentum so the warm-started
        weights adapt cleanly to the new iteration's samples.

        Buffer intentionally NOT cleared: with ~50-200k new samples per
        iteration, clearing would starve the advantage net.  Accumulated
        samples across iterations give millions of training points by
        iteration ~30, enabling stable advantage estimates.  Circular-buffer
        eviction naturally discards the oldest (most-stale) data.
        """
        self.adv_opt = torch.optim.Adam(self.adv_net.parameters(), lr=self.cfg.adv_lr)

    # ------------------------------------------------------------------
    # Traversal collection
    # ------------------------------------------------------------------

    def _collect(self, iteration: int, persistent_pool) -> None:
        cfg = self.cfg
        adv_weights = {k: v.cpu() for k, v in self.adv_net.state_dict().items()}
        strat_weights = {k: v.cpu() for k, v in self.strat_net.state_dict().items()}

        tasks = [
            (
                i % cfg.n_players,
                int(self._rng.integers(0, 2**31)),
                cfg.n_players,
            )
            for i in range(cfg.traversals_per_player * cfg.n_players)
        ]

        if cfg.num_workers <= 1:
            worker_init(adv_weights, strat_weights)
            for task in tasks:
                adv_s, strat_s = worker_task(task)
                self.adv_buf.add_batch(adv_s)
                self.strat_buf.add_batch(strat_s)
        elif persistent_pool is not None:
            # Windows: broadcast weights to persistent workers via IPC.
            update_args = (adv_weights, strat_weights)
            persistent_pool.map(
                worker_update_nets,
                [update_args] * cfg.num_workers,
                chunksize=1,
            )
            for adv_s, strat_s in persistent_pool.imap_unordered(
                worker_task, tasks, chunksize=4
            ):
                self.adv_buf.add_batch(adv_s)
                self.strat_buf.add_batch(strat_s)
        else:
            # Linux: create a fresh pool each iteration.  fork is copy-on-write
            # so the weights are inherited instantly — no IPC serialisation cost.
            with self._ctx.Pool(
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
    # Network training (fixed gradient steps per iteration)
    # ------------------------------------------------------------------

    def _train_adv(self) -> float:
        cfg = self.cfg
        if len(self.adv_buf) < cfg.adv_batch_size:
            return 0.0
        self.adv_net.train()
        total = 0.0
        for _ in range(cfg.adv_train_steps):
            obs, _, targets, actions = self.adv_buf.sample(cfg.adv_batch_size)
            obs_t = torch.FloatTensor(obs).to(self.device)
            targets_t = torch.FloatTensor(targets).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)

            pred = self.adv_net(obs_t)                              # [B, n_actions]
            taken_pred = pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            taken_tgt = targets_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(taken_pred, taken_tgt)

            self.adv_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), 1.0)
            self.adv_opt.step()
            total += loss.item()
        self.adv_net.eval()
        return total / cfg.adv_train_steps

    def _train_strat(self) -> float:
        cfg = self.cfg
        if len(self.strat_buf) < cfg.strat_batch_size:
            return 0.0
        self.strat_net.train()
        total = 0.0
        for _ in range(cfg.strat_train_steps):
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
        return total / cfg.strat_train_steps

    # ------------------------------------------------------------------
    # Evaluation + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int, progress: Progress | None = None) -> None:
        from skull_king.training.cfr.agent import CFRAgent
        n = self.cfg.n_players
        agent = CFRAgent(self.strat_net, n_players=n, name="CFR")
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=100)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=100)
        wr_r = r_r.win_rates().get("CFR", 0.0)
        wr_h = r_h.win_rates().get("CFR", 0.0)
        avg_r = r_r.avg_scores().get("CFR", 0.0)
        avg_h = r_h.avg_scores().get("CFR", 0.0)
        msg = (
            f"  [Eval iter={t}]"
            f"  vs_random={wr_r:.1%} ({avg_r:+.0f})"
            f"  vs_heuristic={wr_h:.1%} ({avg_h:+.0f})"
        )
        if progress is not None:
            progress.print(msg)
        else:
            _console.print(msg)

    def _save(self, base_path: str) -> None:
        torch.save(self.adv_net.state_dict(), f"{base_path}_adv.pt")
        torch.save(self.strat_net.state_dict(), f"{base_path}_strat.pt")
