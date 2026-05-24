"""ReBeL trainer — self-play + subgame solving + network training."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from skull_king.agents import HeuristicAgent, RandomAgent
from skull_king.cards import DECK_TOTAL
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
from skull_king.training.rebel.subgame import SubgameSolver, _build_action_mask

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


class RebelTrainer:
    """Orchestrates ReBeL self-play training.

    Each iteration:
    1. Play N full games using the current policy net (with subgame solving
       at each decision node to get the actual move strategy).
    2. Collect (PBS, strategy, values) tuples from each game.
    3. Train the value network on (PBS, terminal_utility) pairs.
    4. Train the policy network on (PBS, subgame_strategy) pairs.
    5. Evaluate and checkpoint periodically.
    """

    def __init__(self, cfg: "RebelConfig") -> None:
        self.cfg = cfg
        self.device = _pick_device()

        n = cfg.n_players
        pbs_size = pbs_encoding_size(n)

        value_hidden = tuple(cfg.value_hidden)
        policy_hidden = tuple(cfg.policy_hidden)

        self.value_net = RebelValueNet(n, hidden=value_hidden).to(self.device)
        self.policy_net = RebelPolicyNet(n, hidden=policy_hidden).to(self.device)
        self.value_net.eval()
        self.policy_net.eval()

        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)

        self.value_buf = ValueBuffer(cfg.buffer_capacity, pbs_size, n, seed=cfg.seed)
        self.policy_buf = PolicyBuffer(cfg.buffer_capacity, pbs_size, ACTION_SPACE_SIZE,
                                       seed=cfg.seed + 1)

        self._scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self._rng = np.random.default_rng(cfg.seed)

        os.makedirs(cfg.model_dir, exist_ok=True)

        self.solver = SubgameSolver(
            value_net=self.value_net,
            device=self.device,
            n_cfr_iters=cfg.n_cfr_iters_per_subgame,
            max_depth=cfg.max_depth,
            n_samples=cfg.n_subgame_samples,
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
        _p(f"  {cfg.n_iterations} iters  x  {cfg.games_per_iter} games/iter")
        _p(f"  subgame: {cfg.n_subgame_samples} samples × {cfg.n_cfr_iters_per_subgame} CFR iters")
        _p(f"  device: {self.device}")
        if self.device.type == "cuda":
            _p(f"  GPU: {torch.cuda.get_device_name(0)}")
        _p(f"{'='*66}\n")

        for t in range(1, cfg.n_iterations + 1):
            t0 = time.time()

            # Self-play: collect training data
            n_val, n_pol = self._self_play()
            t_play = time.time() - t0

            # Train networks
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

        final = os.path.join(cfg.model_dir, f"{cfg.run_name}_final")
        self._save(final)
        print(f"\nTraining complete. Saved -> {final}_{{value,policy}}.pt", flush=True)

    # ------------------------------------------------------------------
    # Self-play data collection
    # ------------------------------------------------------------------

    def _self_play(self) -> tuple[int, int]:
        """Play games and collect (PBS, strategy, values) for training."""
        cfg = self.cfg
        n_val_samples = 0
        n_pol_samples = 0

        for _ in range(cfg.games_per_iter):
            seed = int(self._rng.integers(0, 2**31))
            engine = GameEngine(n_players=cfg.n_players, seed=seed)
            engine.start()

            game_pbs_encs: list[np.ndarray] = []
            game_masks: list[np.ndarray] = []
            game_strategies: list[np.ndarray] = []

            while engine._phase != GamePhase.GAME_OVER:
                acting_player = engine._current_player_index()
                pbs = PublicBeliefState.from_engine(engine, acting_player)

                # Use subgame solver to get strategy at this node
                result = self.solver.solve(engine, pbs, acting_player)
                strategy = result["strategy"]
                mask = result["mask"]
                pbs_enc = result["pbs_enc"]

                # Store for policy training
                game_pbs_encs.append(pbs_enc)
                game_masks.append(mask)
                game_strategies.append(strategy)

                # Sample action from strategy and apply
                legal = np.where(mask)[0]
                probs = strategy[legal]
                probs = np.maximum(probs, 0)
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(len(legal)) / len(legal)

                action = legal[np.random.choice(len(legal), p=probs)]

                # Apply action to engine
                if engine._phase == GamePhase.BIDDING:
                    engine.place_bid_no_state(acting_player, int(action))
                else:
                    from skull_king.training.rebel.subgame import _action_to_card
                    card, tigress_mode = _action_to_card(action, engine)
                    engine.play_card_no_state(acting_player, card, tigress_mode)

            # Game over: compute terminal utilities
            final_scores = [p.total_score for p in engine._players]
            from skull_king.training.cfr.traversal import _utility_from_scores
            terminal_values = np.array([
                _utility_from_scores(final_scores, i) for i in range(cfg.n_players)
            ], dtype=np.float32)

            # Add to value buffer (all positions → terminal values)
            for enc in game_pbs_encs:
                self.value_buf.add(enc, terminal_values)
                n_val_samples += 1

            # Add to policy buffer
            for enc, mask, strat in zip(game_pbs_encs, game_masks, game_strategies):
                self.policy_buf.add(enc, mask, strat)
                n_pol_samples += 1

        return n_val_samples, n_pol_samples

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
                # Cross-entropy against subgame-solved strategy
                loss = -(strat_t * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

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
        agent = RebelAgent(self.policy_net, n_players=n, name="ReBeL")
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
