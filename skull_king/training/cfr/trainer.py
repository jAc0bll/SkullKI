"""Deep CFR training loop."""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from typing import TYPE_CHECKING

if sys.platform != "win32":
    import resource

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

        self.adv_buf = AdvantageBuffer(
            capacity=cfg.adv_buffer_capacity, seed=cfg.env_seed + 1
        )
        self.strat_buf = StrategyBuffer(
            capacity=cfg.strat_buffer_capacity, seed=cfg.env_seed + 2
        )

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

        # Pool is always created once before training starts (persistent).
        # Reason: on Linux, fork copies the parent's page-table for ALL allocated
        # memory.  Once the advantage/strategy buffers are filled (~8 GB), forking
        # 44 workers per iteration costs ~50 s just in COW page-table setup.
        # Creating the pool once (when buffers are still empty) pays that cost
        # only at startup.  Weights are broadcast cheaply via worker_update_nets
        # each iteration (~0.3 s for [512,512] nets × 44 workers vs ~50 s fork).
        #
        # On Windows (spawn): persistent pool avoids per-iteration process creation
        # overhead (~2-4 s × n_iterations); same broadcast mechanism is used.
        self._is_windows = sys.platform == "win32"
        self._ctx = multiprocessing.get_context("spawn" if self._is_windows else "fork")

        # Limit torch threads in the main process.  Workers each call
        # torch.set_num_threads(1) in worker_init, so they are unaffected.
        # 8 threads is the sweet spot for our batch size (1024) on both GPU
        # boxes and CPU-only NUMA servers — more threads cause cross-socket
        # memory traffic that slows rather than speeds the small matmuls.
        if cfg.num_workers > 1:
            torch.set_num_threads(min(8, torch.get_num_threads()))

        # Raise the open-file limit before spawning the pool.
        # Each worker needs ~5 FDs (pipes + Manager socket); 220 workers easily
        # exceeds the default limit of 1024.  65536 is safe for any pod size.
        if sys.platform != "win32":
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                target = max(soft, 65536)
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, max(hard, target)))
            except ValueError:
                pass  # container hard limit below 65536; ulimit -n 65536 manually

        persistent_pool = None
        self._manager = None
        self._shared_w = None
        self._shared_v = None
        if cfg.num_workers > 1:
            init_adv = {k: v.cpu() for k, v in self.adv_net.state_dict().items()}
            # Manager-backed dict + shared atomic int for lockless weight
            # broadcast.  Workers read the version each traversal (single
            # atomic int — essentially free) and only do a full reload when
            # the version moves forward — guarantees every worker sees every
            # update, unlike the previous pool.map-based broadcast.
            self._manager = self._ctx.Manager()
            self._shared_w = self._manager.dict()
            self._shared_w["weights"] = init_adv
            self._shared_v = self._ctx.Value("Q", 0)  # iter-0 already loaded
            persistent_pool = self._ctx.Pool(
                processes=cfg.num_workers,
                initializer=worker_init,
                initargs=(init_adv, cfg.n_players, self._shared_w, self._shared_v,
                          cfg.heuristic_frac),
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
            if self._manager is not None:
                self._manager.shutdown()

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

        tasks = [
            (
                i % cfg.n_players,
                int(self._rng.integers(0, 2**31)),
                cfg.n_players,
            )
            for i in range(cfg.traversals_per_player * cfg.n_players)
        ]

        # Publish new weights via shared memory + bump the version counter.
        # Every worker re-checks the version at the start of every traversal
        # and reloads at most once per iteration — race-free, no broadcast
        # task scheduling involved.
        if self._shared_w is not None and self._shared_v is not None:
            self._shared_w["weights"] = adv_weights
            with self._shared_v.get_lock():
                self._shared_v.value += 1

        if persistent_pool is None:
            # Single-process fallback (num_workers <= 1).
            worker_init(adv_weights, cfg.n_players, heuristic_frac=cfg.heuristic_frac)
            results = (worker_task(t) for t in tasks)
        else:
            results = persistent_pool.imap_unordered(worker_task, tasks, chunksize=4)

        all_adv_obs, all_adv_masks, all_adv_targets, all_adv_actions = [], [], [], []
        all_strat_obs, all_strat_masks, all_strat_strategies = [], [], []
        for result in results:
            a_obs, a_masks, a_targets, a_actions, s_obs, s_masks, s_strats = result
            if a_obs.shape[0] > 0:
                all_adv_obs.append(a_obs)
                all_adv_masks.append(a_masks)
                all_adv_targets.append(a_targets)
                all_adv_actions.append(a_actions)
            if s_obs.shape[0] > 0:
                all_strat_obs.append(s_obs)
                all_strat_masks.append(s_masks)
                all_strat_strategies.append(s_strats)

        if all_adv_obs:
            self.adv_buf.add_batch_vec(
                np.concatenate(all_adv_obs),
                np.concatenate(all_adv_masks),
                np.concatenate(all_adv_targets),
                np.concatenate(all_adv_actions),
            )
        if all_strat_obs:
            self.strat_buf.add_batch_vec(
                np.concatenate(all_strat_obs),
                np.concatenate(all_strat_masks),
                np.concatenate(all_strat_strategies),
            )

    # ------------------------------------------------------------------
    # Network training (fixed gradient steps per iteration)
    # ------------------------------------------------------------------

    def _train_adv(self) -> float:
        cfg = self.cfg
        if len(self.adv_buf) < cfg.adv_batch_size:
            return 0.0
        # Scale steps with buffer size so early iters (~50k samples) don't
        # over-train and overfit to noisy early data.  Each sample should be
        # seen ~4× on average per iteration: steps = buffer_size / (batch//4).
        effective_steps = min(
            cfg.adv_train_steps,
            max(1, len(self.adv_buf) // (cfg.adv_batch_size // 4)),
        )
        self.adv_net.train()
        total = 0.0
        for _ in range(effective_steps):
            obs, masks, targets, _ = self.adv_buf.sample(cfg.adv_batch_size)
            # from_numpy is zero-copy (read-only view); contiguous() ensures
            # the tensor is writable before the forward pass modifies gradients.
            obs_t     = torch.from_numpy(obs).float().to(self.device)
            targets_t = torch.from_numpy(targets).float().to(self.device)
            mask_f    = torch.from_numpy(masks).float().to(self.device)

            pred = self.adv_net(obs_t)                              # [B, n_actions]
            # Element-wise masked MSE: avoids the boolean-gather (pred[mask_t])
            # which allocates a variable-length 1D tensor on CPU and is slow.
            # Mathematically identical: mean over legal (mask=1) action slots.
            diff = (pred - targets_t) * mask_f
            loss = (diff * diff).sum() / mask_f.sum().clamp(min=1)

            self.adv_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), 1.0)
            self.adv_opt.step()
            total += loss.item()
        self.adv_net.eval()
        return total / effective_steps

    def _train_strat(self) -> float:
        cfg = self.cfg
        if len(self.strat_buf) < cfg.strat_batch_size:
            return 0.0
        effective_steps = min(
            cfg.strat_train_steps,
            max(1, len(self.strat_buf) // (cfg.strat_batch_size // 4)),
        )
        self.strat_net.train()
        total = 0.0
        for _ in range(effective_steps):
            obs, masks, strategies = self.strat_buf.sample(cfg.strat_batch_size)
            obs_t   = torch.from_numpy(obs).float().to(self.device)
            mask_t  = torch.from_numpy(masks).to(self.device)
            strat_t = torch.from_numpy(strategies).float().to(self.device)

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
        return total / effective_steps

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
