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
from skull_king.env.skull_king_env import OBS_SIZE
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.cfr.buffers import AdvantageBuffer, StrategyBuffer
from skull_king.training.cfr.networks import (
    AdvantageNet,
    BID_ACTION_SIZE,
    BiddingAdvNet,
    BiddingStratNet,
    PLAY_ACTION_SIZE,
    PlayingAdvNet,
    PlayingStratNet,
    StrategyNet,
)
from skull_king.training.cfr.traversal import (
    worker_init,
    worker_init_split,
    worker_task,
    worker_task_split,
)

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
        # forkserver on Linux: the server process is started before CUDA is
        # initialized in worker processes, avoiding the CUDA+fork deadlock that
        # occurs when forking a process that already has an active CUDA context.
        if self._is_windows:
            ctx_method = "spawn"
        elif torch.cuda.is_available():
            ctx_method = "forkserver"
        else:
            ctx_method = "fork"
        self._ctx = multiprocessing.get_context(ctx_method)

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
            # chunksize=64: 2000 tasks / 64 = 31 scheduling round-trips instead of 500.
            # Each round-trip carries 64x55KB of result pickles — same total IPC volume
            # but 16x fewer main-process iterations and 16x fewer pipe-send syscalls.
            chunksize = max(4, len(tasks) // (cfg.num_workers * 2))
            results = persistent_pool.imap_unordered(worker_task, tasks, chunksize=chunksize)

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


# ---------------------------------------------------------------------------
# Split-network trainer
# ---------------------------------------------------------------------------


class SplitDeepCFRTrainer:
    """Deep CFR trainer with separate bidding and playing networks.

    Uses four networks:
      bid_adv_net / bid_strat_net  — for bidding decisions (output size 11)
      play_adv_net / play_strat_net — for playing decisions (output size 71)

    Each network has its own advantage/strategy buffer so gradients from
    bidding (hand-strength estimation) and playing (card selection) never
    interfere with each other.
    """

    def __init__(self, cfg: "CFRConfig") -> None:
        self.cfg = cfg
        self.device = _pick_device()

        bid_hidden = tuple(getattr(cfg, "bid_hidden", [256, 256]))
        play_hidden = tuple(getattr(cfg, "play_hidden", [512, 512]))

        self.bid_adv_net = BiddingAdvNet(hidden=bid_hidden).to(self.device)
        self.bid_strat_net = BiddingStratNet(hidden=bid_hidden).to(self.device)
        self.play_adv_net = PlayingAdvNet(hidden=play_hidden).to(self.device)
        self.play_strat_net = PlayingStratNet(hidden=play_hidden).to(self.device)
        for net in (self.bid_adv_net, self.bid_strat_net,
                    self.play_adv_net, self.play_strat_net):
            net.eval()

        self.bid_adv_opt = torch.optim.Adam(self.bid_adv_net.parameters(), lr=cfg.adv_lr)
        self.bid_strat_opt = torch.optim.Adam(self.bid_strat_net.parameters(), lr=cfg.strat_lr)
        self.play_adv_opt = torch.optim.Adam(self.play_adv_net.parameters(), lr=cfg.adv_lr)
        self.play_strat_opt = torch.optim.Adam(self.play_strat_net.parameters(), lr=cfg.strat_lr)

        cap_adv = cfg.adv_buffer_capacity
        cap_str = cfg.strat_buffer_capacity
        seed = cfg.env_seed
        self.bid_adv_buf = AdvantageBuffer(cap_adv, obs_size=OBS_SIZE,
                                           action_size=BID_ACTION_SIZE, seed=seed + 1)
        self.bid_strat_buf = StrategyBuffer(cap_str, obs_size=OBS_SIZE,
                                            action_size=BID_ACTION_SIZE, seed=seed + 2)
        self.play_adv_buf = AdvantageBuffer(cap_adv, obs_size=OBS_SIZE,
                                            action_size=PLAY_ACTION_SIZE, seed=seed + 3)
        self.play_strat_buf = StrategyBuffer(cap_str, obs_size=OBS_SIZE,
                                             action_size=PLAY_ACTION_SIZE, seed=seed + 4)

        self._rng = np.random.default_rng(seed)
        os.makedirs(cfg.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        total_traversals = cfg.traversals_per_player * cfg.n_players

        def _p(msg: str = "") -> None:
            print(msg, flush=True)

        _p(f"\n{'='*66}")
        _p(f"  Deep CFR (split nets)  -  {cfg.run_name}")
        _p(f"  {cfg.n_cfr_iterations} iters x {total_traversals} traversals/iter"
           f"  x  workers={cfg.num_workers}")
        _p(f"  device: {self.device}")
        if self.device.type == "cuda":
            import torch.cuda as _cu
            gpu_name = _cu.get_device_name(0)
            gpu_gb   = _cu.get_device_properties(0).total_memory / 1e9
            _test = torch.ones(128, 128, device=self.device) @ torch.ones(128, 128, device=self.device)
            _p(f"  GPU verified: {gpu_name}  ({gpu_gb:.0f} GB)  [matmul OK]")
            del _test
        from skull_king.cfr_engine import SplitCEngine
        _p(f"  C engine: {'active (split traversal ~5ms/game)' if SplitCEngine.available else 'NOT BUILT — using slow Python traversal (35ms/game)'}")
        _p(f"{'='*66}\n")

        self._is_windows = sys.platform == "win32"
        if self._is_windows:
            ctx_method = "spawn"
        elif torch.cuda.is_available():
            ctx_method = "forkserver"
        else:
            ctx_method = "fork"
        self._ctx = multiprocessing.get_context(ctx_method)

        if cfg.num_workers > 1:
            torch.set_num_threads(min(8, torch.get_num_threads()))

        if sys.platform != "win32":
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                target = max(soft, 65536)
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, max(hard, target)))
            except ValueError:
                pass

        persistent_pool = None
        self._manager = None
        self._shared_w = None
        self._shared_v = None
        if cfg.num_workers > 1:
            bid_w = {k: v.cpu() for k, v in self.bid_adv_net.state_dict().items()}
            play_w = {k: v.cpu() for k, v in self.play_adv_net.state_dict().items()}
            self._manager = self._ctx.Manager()
            self._shared_w = self._manager.dict()
            self._shared_w["weights"] = {"bid_weights": bid_w, "play_weights": play_w}
            self._shared_v = self._ctx.Value("Q", 0)
            persistent_pool = self._ctx.Pool(
                processes=cfg.num_workers,
                initializer=worker_init_split,
                initargs=(bid_w, play_w, cfg.n_players,
                          self._shared_w, self._shared_v, cfg.heuristic_frac),
            )

        try:
            for t in range(1, cfg.n_cfr_iterations + 1):
                t0 = time.time()

                self._reset_adv()
                self._collect(t, persistent_pool)
                bid_adv_loss = self._train_net(
                    self.bid_adv_net, self.bid_adv_opt, self.bid_adv_buf,
                    cfg.adv_batch_size, cfg.adv_train_steps, mode="adv",
                )
                bid_strat_loss = self._train_net(
                    self.bid_strat_net, self.bid_strat_opt, self.bid_strat_buf,
                    cfg.strat_batch_size, cfg.strat_train_steps, mode="strat",
                )
                play_adv_loss = self._train_net(
                    self.play_adv_net, self.play_adv_opt, self.play_adv_buf,
                    cfg.adv_batch_size, cfg.adv_train_steps, mode="adv",
                )
                play_strat_loss = self._train_net(
                    self.play_strat_net, self.play_strat_opt, self.play_strat_buf,
                    cfg.strat_batch_size, cfg.strat_train_steps, mode="strat",
                )

                elapsed = time.time() - t0
                print(
                    f"iter {t:4d}/{cfg.n_cfr_iterations}"
                    f"  b_adv={bid_adv_loss:.3f} b_str={bid_strat_loss:.3f}"
                    f"  p_adv={play_adv_loss:.3f} p_str={play_strat_loss:.3f}"
                    f"  bid_buf={len(self.bid_adv_buf):,}"
                    f"  play_buf={len(self.play_adv_buf):,}"
                    f"  {elapsed:.1f}s/it",
                    flush=True,
                )

                if t % cfg.eval_every_n_iters == 0:
                    self._evaluate(t)

                if t % cfg.checkpoint_every_n_iters == 0:
                    path = os.path.join(cfg.model_dir, f"{cfg.run_name}_iter{t}")
                    self._save(path)
                    print(f"  [Checkpoint] -> {path}_{{bid,play}}_{{adv,strat}}.pt",
                          flush=True)
        finally:
            if persistent_pool is not None:
                persistent_pool.close()
                persistent_pool.join()
            if self._manager is not None:
                self._manager.shutdown()

        final = os.path.join(cfg.model_dir, "cfr_split_final")
        self._save(final)
        print(f"\nTraining complete. Saved -> {final}_{{bid,play}}_{{adv,strat}}.pt",
              flush=True)

    def _reset_adv(self) -> None:
        self.bid_adv_opt = torch.optim.Adam(self.bid_adv_net.parameters(), lr=self.cfg.adv_lr)
        self.play_adv_opt = torch.optim.Adam(self.play_adv_net.parameters(), lr=self.cfg.adv_lr)

    def _collect(self, iteration: int, persistent_pool) -> None:
        cfg = self.cfg
        bid_w = {k: v.cpu() for k, v in self.bid_adv_net.state_dict().items()}
        play_w = {k: v.cpu() for k, v in self.play_adv_net.state_dict().items()}

        tasks = [
            (i % cfg.n_players, int(self._rng.integers(0, 2**31)), cfg.n_players)
            for i in range(cfg.traversals_per_player * cfg.n_players)
        ]

        if self._shared_w is not None and self._shared_v is not None:
            self._shared_w["weights"] = {"bid_weights": bid_w, "play_weights": play_w}
            with self._shared_v.get_lock():
                self._shared_v.value += 1

        if persistent_pool is None:
            from skull_king.training.cfr.traversal import (
                BiddingAdvNet as _BAN, PlayingAdvNet as _PAN,
                worker_init_split, traverse_split,
            )
            worker_init_split(bid_w, play_w, cfg.n_players,
                              heuristic_frac=cfg.heuristic_frac)
            results = (worker_task_split(t) for t in tasks)
        else:
            chunksize = max(4, len(tasks) // (cfg.num_workers * 2))
            results = persistent_pool.imap_unordered(worker_task_split, tasks,
                                                     chunksize=chunksize)

        ba_obs, ba_masks, ba_targets, ba_acts = [], [], [], []
        bs_obs, bs_masks, bs_strats = [], [], []
        pa_obs, pa_masks, pa_targets, pa_acts = [], [], [], []
        ps_obs, ps_masks, ps_strats = [], [], []

        for res in results:
            (b_ao, b_am, b_at, b_aa,
             b_so, b_sm, b_ss,
             p_ao, p_am, p_at, p_aa,
             p_so, p_sm, p_ss) = res
            if b_ao.shape[0]:
                ba_obs.append(b_ao); ba_masks.append(b_am)
                ba_targets.append(b_at); ba_acts.append(b_aa)
            if b_so.shape[0]:
                bs_obs.append(b_so); bs_masks.append(b_sm); bs_strats.append(b_ss)
            if p_ao.shape[0]:
                pa_obs.append(p_ao); pa_masks.append(p_am)
                pa_targets.append(p_at); pa_acts.append(p_aa)
            if p_so.shape[0]:
                ps_obs.append(p_so); ps_masks.append(p_sm); ps_strats.append(p_ss)

        if ba_obs:
            self.bid_adv_buf.add_batch_vec(
                np.concatenate(ba_obs), np.concatenate(ba_masks),
                np.concatenate(ba_targets), np.concatenate(ba_acts),
            )
        if bs_obs:
            self.bid_strat_buf.add_batch_vec(
                np.concatenate(bs_obs), np.concatenate(bs_masks),
                np.concatenate(bs_strats),
            )
        if pa_obs:
            self.play_adv_buf.add_batch_vec(
                np.concatenate(pa_obs), np.concatenate(pa_masks),
                np.concatenate(pa_targets), np.concatenate(pa_acts),
            )
        if ps_obs:
            self.play_strat_buf.add_batch_vec(
                np.concatenate(ps_obs), np.concatenate(ps_masks),
                np.concatenate(ps_strats),
            )

    def _train_net(
        self,
        net: nn.Module,
        opt: torch.optim.Optimizer,
        buf,
        batch_size: int,
        max_steps: int,
        mode: str,
    ) -> float:
        if len(buf) < batch_size:
            return 0.0
        effective_steps = min(max_steps, max(1, len(buf) // (batch_size // 4)))
        net.train()
        total = 0.0
        for _ in range(effective_steps):
            if mode == "adv":
                obs, masks, targets, _ = buf.sample(batch_size)
                obs_t = torch.from_numpy(obs).float().to(self.device)
                tgt_t = torch.from_numpy(targets).float().to(self.device)
                mask_f = torch.from_numpy(masks).float().to(self.device)
                pred = net(obs_t)
                diff = (pred - tgt_t) * mask_f
                loss = (diff * diff).sum() / mask_f.sum().clamp(min=1)
            else:
                obs, masks, strats = buf.sample(batch_size)
                obs_t = torch.from_numpy(obs).float().to(self.device)
                mask_t = torch.from_numpy(masks).to(self.device)
                strat_t = torch.from_numpy(strats).float().to(self.device)
                logits = net(obs_t).masked_fill(~mask_t, float("-inf"))
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -(strat_t * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item()
        net.eval()
        return total / effective_steps

    def _evaluate(self, t: int) -> None:
        from skull_king.training.cfr.agent import SplitCFRAgent
        n = self.cfg.n_players
        agent = SplitCFRAgent(
            self.bid_strat_net, self.play_strat_net, n_players=n, name="CFR-split"
        )
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=200)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=200)
        wr_r = r_r.win_rates().get("CFR-split", 0.0)
        wr_h = r_h.win_rates().get("CFR-split", 0.0)
        avg_r = r_r.avg_scores().get("CFR-split", 0.0)
        avg_h = r_h.avg_scores().get("CFR-split", 0.0)
        print(
            f"  [EVAL iter={t}]"
            f"  vs_random={wr_r:.1%} ({avg_r:+.0f})"
            f"  vs_heuristic={wr_h:.1%} ({avg_h:+.0f})",
            flush=True,
        )

    def _save(self, base_path: str) -> None:
        torch.save(self.bid_adv_net.state_dict(), f"{base_path}_bid_adv.pt")
        torch.save(self.bid_strat_net.state_dict(), f"{base_path}_bid_strat.pt")
        torch.save(self.play_adv_net.state_dict(), f"{base_path}_play_adv.pt")
        torch.save(self.play_strat_net.state_dict(), f"{base_path}_play_strat.pt")
