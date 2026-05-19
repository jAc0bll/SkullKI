"""NFSP training callback: reservoir filling + SL updates + weight sync."""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from skull_king.training.nfsp_reservoir import ReservoirBuffer
from skull_king.training.nfsp_sl import SLTrainer


class NFSPCallback(BaseCallback):
    """Interleaves supervised-learning updates with PPO rollout updates.

    After every PPO rollout:
      1. Extracts (obs, action_mask, action) from the rollout buffer and
         adds them to the reservoir (Algorithm R sampling).
      2. Trains the SL network on a random minibatch from the reservoir.
      3. Pushes the updated SL weights to all training envs.

    The SL training happens *every* rollout once the reservoir is warm
    (>= min_buffer_size samples).  The weight sync uses vec_env.env_method,
    which is safe with both DummyVecEnv and SubprocVecEnv.
    """

    def __init__(
        self,
        reservoir: ReservoirBuffer,
        sl_trainer: SLTrainer,
        sl_n_updates: int = 8,
        min_buffer_size: int = 5_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.reservoir = reservoir
        self.sl_trainer = sl_trainer
        self.sl_n_updates = sl_n_updates
        self.min_buffer_size = min_buffer_size

    def _on_rollout_end(self) -> bool:
        buf = self.model.rollout_buffer

        # ── 1. Harvest (obs, mask, action) from rollout buffer ─────────────
        obs = buf.observations.reshape(-1, buf.observations.shape[-1])
        acts = buf.actions.reshape(-1).astype(np.int64)

        if hasattr(buf, "action_masks") and buf.action_masks is not None:
            masks = buf.action_masks.reshape(-1, buf.action_masks.shape[-1]).astype(bool)
        else:
            # Fallback: allow all actions (shouldn't happen with MaskablePPO).
            masks = np.ones((len(obs), self.model.action_space.n), dtype=bool)

        self.reservoir.add_batch(obs, masks, acts)

        # ── 2. Train SL ─────────────────────────────────────────────────────
        if len(self.reservoir) < self.min_buffer_size:
            return True

        loss = self.sl_trainer.update(self.reservoir, n_updates=self.sl_n_updates)

        # ── 3. Sync SL weights to all envs ──────────────────────────────────
        state_dict = {
            k: v.cpu() for k, v in self.sl_trainer.net.state_dict().items()
        }
        self.training_env.env_method("set_sl_weights", state_dict)

        if self.verbose >= 1:
            print(
                f"[NFSP] SL loss={loss:.4f}  "
                f"reservoir={len(self.reservoir):,}  "
                f"step={self.num_timesteps:,}"
            )

        self.logger.record("nfsp/sl_loss", loss)
        self.logger.record("nfsp/reservoir_size", len(self.reservoir))

        return True

    def _on_training_end(self) -> None:
        # Final sync so envs end up with the best weights.
        state_dict = {
            k: v.cpu() for k, v in self.sl_trainer.net.state_dict().items()
        }
        self.training_env.env_method("set_sl_weights", state_dict)

    def _on_step(self) -> bool:
        return True
