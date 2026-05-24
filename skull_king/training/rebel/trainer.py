"""ReBeL trainer — self-play + subgame solving + network training."""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GamePhase
from skull_king.tournament.runner import TournamentRunner
from skull_king.training.rebel.buffers import PolicyBuffer, ValueBuffer
from skull_king.training.rebel.networks import RebelPolicyNet, RebelValueNet
from skull_king.training.rebel.public_belief_state import (
    PublicBeliefState,
    pbs_encoding_size,
)
from skull_king.training.rebel.subgame import SubgameSolver, _action_to_card

if TYPE_CHECKING:
    from skull_king.training.rebel.train import RebelConfig


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except (ImportError, Exception):
        pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Module-level worker — must be at top level to be picklable with spawn.
# ---------------------------------------------------------------------------

def _play_one_game_worker(args: tuple) -> tuple:
    """Play one complete Skull King game. Runs in a subprocess on CPU."""
    seed, n_players, n_cfr_iters, max_depth, value_hidden, state_dict = args

    device = torch.device("cpu")
    value_net = None
    if state_dict is not None:
        value_net = RebelValueNet(n_players, hidden=tuple(value_hidden))
        # numpy arrays were passed to avoid multiprocessing resource_sharer fd issues
        value_net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
        value_net.eval()

    solver = SubgameSolver(
        value_net=value_net, device=device,
        n_cfr_iters=n_cfr_iters, max_depth=max_depth,
    )

    engine = GameEngine(n_players=n_players, seed=seed)
    engine.start()

    pbs_encs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    strategies: list[np.ndarray] = []

    while engine._phase != GamePhase.GAME_OVER:
        acting = engine._current_player_index()
        pbs = PublicBeliefState.from_engine(engine, acting)
        result = solver.solve(engine, pbs, acting)

        pbs_encs.append(result["pbs_enc"])
        masks.append(result["mask"])
        strategies.append(result["strategy"])

        legal = np.where(result["mask"])[0]
        probs = np.maximum(result["strategy"][legal], 0.0)
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones(len(legal)) / len(legal)
        action = int(legal[np.random.choice(len(legal), p=probs)])

        if engine._phase == GamePhase.BIDDING:
            engine.place_bid_no_state(acting, action)
        else:
            card, tm = _action_to_card(action, engine)
            engine.play_card_no_state(acting, card, tm)

    from skull_king.training.cfr.traversal import _utility_from_scores
    scores = [p.total_score for p in engine._players]
    term_vals = np.array(
        [_utility_from_scores(scores, i) for i in range(n_players)],
        dtype=np.float32,
    )
    return pbs_encs, masks, strategies, term_vals


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RebelTrainer:
    """Orchestrates ReBeL self-play training with parallel CPU game workers."""

    def __init__(self, cfg: "RebelConfig") -> None:
        self.cfg = cfg
        self.device = _pick_device()

        n = cfg.n_players
        pbs_size = pbs_encoding_size(n)

        self.value_net = RebelValueNet(n, hidden=tuple(cfg.value_hidden)).to(self.device)
        self.policy_net = RebelPolicyNet(n, hidden=tuple(cfg.policy_hidden)).to(self.device)
        self.value_net.eval()
        self.policy_net.eval()

        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)

        self.value_buf = ValueBuffer(cfg.buffer_capacity, pbs_size, n, seed=cfg.seed)
        self.policy_buf = PolicyBuffer(cfg.buffer_capacity, pbs_size, ACTION_SPACE_SIZE,
                                       seed=cfg.seed + 1)

        self._scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self._rng = np.random.default_rng(cfg.seed)

        os.makedirs(cfg.model_dir, exist_ok=True)

        if cfg.resume_from:
            self.value_net.load_state_dict(
                torch.load(f"{cfg.resume_from}_value.pt", map_location=self.device)
            )
            self.policy_net.load_state_dict(
                torch.load(f"{cfg.resume_from}_policy.pt", map_location=self.device)
            )
            print(f"Resumed from {cfg.resume_from}", flush=True)

        # Raise fd limit — 80 spawn workers need ~10 fds each, default 1024 is too low
        try:
            _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))
        except Exception:
            pass

        # Persistent worker pool — spawn avoids CUDA fork issues
        n_workers = min(cfg.games_per_iter, max(1, mp.cpu_count() - 2))
        self._n_workers = n_workers
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg

        def _p(msg: str = "") -> None:
            print(msg, flush=True)

        _p(f"\n{'='*66}")
        _p(f"  ReBeL  -  {cfg.run_name}")
        _p(f"  {cfg.n_iterations} iters  x  {cfg.games_per_iter} games/iter  ({self._n_workers} workers)")
        _p(f"  subgame: {cfg.n_cfr_iters_per_subgame} CFR iters  max_depth={cfg.max_depth}")
        _p(f"  device: {self.device}")
        if self.device.type == "cuda":
            _p(f"  GPU: {torch.cuda.get_device_name(0)}")
        _p(f"{'='*66}\n")

        try:
            for t in range(cfg.start_iter, cfg.n_iterations + 1):
                t0 = time.time()

                n_val, n_pol = self._self_play()
                t_play = time.time() - t0

                t1 = time.time()
                val_loss = self._train_value()
                pol_loss = self._train_policy()
                t_train = time.time() - t1

                elapsed = time.time() - t0
                print(
                    f"iter {t:4d}/{cfg.n_iterations}"
                    f"  val_loss={val_loss:.4f}  pol_loss={pol_loss:.4f}"
                    f"  val_buf={len(self.value_buf):,}  pol_buf={len(self.policy_buf):,}"
                    f"  play={t_play:.1f}s train={t_train:.1f}s total={elapsed:.1f}s",
                    flush=True,
                )

                if t % cfg.eval_every == 0:
                    self._evaluate(t)

                if t % cfg.checkpoint_every == 0:
                    path = os.path.join(cfg.model_dir, f"{cfg.run_name}_iter{t}")
                    self._save(path)
                    print(f"  [Checkpoint] -> {path}_{{value,policy}}.pt", flush=True)

        finally:
            self._executor.shutdown(wait=False)

        final = os.path.join(cfg.model_dir, f"{cfg.run_name}_final")
        self._save(final)
        print(f"\nTraining complete. Saved -> {final}_{{value,policy}}.pt", flush=True)

    # ------------------------------------------------------------------
    # Self-play — parallel across CPU workers
    # ------------------------------------------------------------------

    def _self_play(self) -> tuple[int, int]:
        cfg = self.cfg

        # Serialize as numpy — avoids multiprocessing resource_sharer fd leak
        state_dict = {k: v.cpu().numpy() for k, v in self.value_net.state_dict().items()}

        args = [
            (int(self._rng.integers(0, 2**31)), cfg.n_players,
             cfg.n_cfr_iters_per_subgame, cfg.max_depth,
             list(cfg.value_hidden), state_dict)
            for _ in range(cfg.games_per_iter)
        ]

        results = list(self._executor.map(_play_one_game_worker, args))

        n_val = n_pol = 0
        for pbs_encs, masks, strategies, term_vals in results:
            for enc in pbs_encs:
                self.value_buf.add(enc, term_vals)
                n_val += 1
            for enc, mask, strat in zip(pbs_encs, masks, strategies):
                self.policy_buf.add(enc, mask, strat)
                n_pol += 1

        return n_val, n_pol

    # ------------------------------------------------------------------
    # Network training
    # ------------------------------------------------------------------

    def _train_value(self) -> float:
        cfg = self.cfg
        if len(self.value_buf) < cfg.batch_size:
            return 0.0

        steps = min(cfg.train_steps, max(1, len(self.value_buf) // (cfg.batch_size // 4)))
        self.value_net.train()
        total = 0.0
        amp_ctx = torch.amp.autocast(device_type=self.device.type,
                                     enabled=(self.device.type == "cuda"))

        for _ in range(steps):
            enc, vals = self.value_buf.sample(cfg.batch_size)
            enc_t = torch.from_numpy(enc).float().to(self.device)
            val_t = torch.from_numpy(vals).float().to(self.device)

            with amp_ctx:
                pred = self.value_net(enc_t)
                loss = nn.functional.mse_loss(pred, val_t)

            self.value_opt.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.value_opt)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self._scaler.step(self.value_opt)
            self._scaler.update()
            total += loss.item()

        self.value_net.eval()
        return total / steps

    def _train_policy(self) -> float:
        cfg = self.cfg
        if len(self.policy_buf) < cfg.batch_size:
            return 0.0

        steps = min(cfg.train_steps, max(1, len(self.policy_buf) // (cfg.batch_size // 4)))
        self.policy_net.train()
        total = 0.0
        amp_ctx = torch.amp.autocast(device_type=self.device.type,
                                     enabled=(self.device.type == "cuda"))

        for _ in range(steps):
            enc, masks, strats = self.policy_buf.sample(cfg.batch_size)
            enc_t = torch.from_numpy(enc).float().to(self.device)
            mask_t = torch.from_numpy(masks).to(self.device)
            strat_t = torch.from_numpy(strats).float().to(self.device)

            with amp_ctx:
                log_probs = self.policy_net(enc_t, mask_t)
            # Cast to fp32 + clamp: fp16 can produce -inf for low-prob legal actions,
            # making -(strat * -inf) = +inf and blowing up the loss.
            loss = -(strat_t * log_probs.float().clamp(min=-100.0)).nan_to_num(0.0).sum(dim=-1).mean()

            self.policy_opt.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.policy_opt)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self._scaler.step(self.policy_opt)
            self._scaler.update()
            total += loss.item()

        self.policy_net.eval()
        return total / steps

    # ------------------------------------------------------------------
    # Evaluation + persistence
    # ------------------------------------------------------------------

    def _evaluate(self, t: int) -> None:
        from skull_king.training.rebel.agent import RebelAgent
        n = self.cfg.n_players
        agent = RebelAgent(self.policy_net, n_players=n, name="ReBeL",
                           value_net=self.value_net)
        runner = TournamentRunner(seed=999)
        r_r = runner.run([agent] + [RandomAgent(i) for i in range(n - 1)], n_games=200)
        r_h = runner.run([agent] + [HeuristicAgent() for _ in range(n - 1)], n_games=200)
        wr_r = r_r.win_rates().get("ReBeL", 0.0)
        wr_h = r_h.win_rates().get("ReBeL", 0.0)
        avg_r = r_r.avg_scores().get("ReBeL", 0.0)
        avg_h = r_h.avg_scores().get("ReBeL", 0.0)
        print(
            f"  [EVAL iter={t}]"
            f"  vs_random={wr_r:.1%} ({avg_r:+.0f})"
            f"  vs_heuristic={wr_h:.1%} ({avg_h:+.0f})",
            flush=True,
        )

    def _save(self, base_path: str) -> None:
        torch.save(self.value_net.state_dict(), f"{base_path}_value.pt")
        torch.save(self.policy_net.state_dict(), f"{base_path}_policy.pt")
