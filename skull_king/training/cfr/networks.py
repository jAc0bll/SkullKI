"""Advantage network, strategy network, and regret matching for Deep CFR."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, N_BID_ACTIONS, OBS_SIZE

# Action-space sizes for the split-network architecture.
# BID_ACTION_SIZE: bids 0..10 (same indices as global action space).
# PLAY_ACTION_SIZE: card slots 0..68 + Tigress-as-ESCAPE (69) + Tigress-as-PIRATE (70).
BID_ACTION_SIZE = N_BID_ACTIONS   # 11
PLAY_ACTION_SIZE = 71             # 69 cards + 2 Tigress modes


def regret_match(advantages: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CFR regret matching: strategy[a] ∝ max(0, advantage[a]) for legal actions.

    Falls back to uniform over legal actions when all advantages are ≤ 0.
    """
    pos = np.where(mask, np.maximum(advantages, 0.0), 0.0)
    total = pos.sum()
    if total > 1e-12:
        return pos / total
    uniform = mask.astype(np.float32)
    return uniform / uniform.sum()


class _MLP(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = obs_size
        for h in hidden:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class AdvantageNet(_MLP):
    """Maps observation → counterfactual advantage per action (real-valued).

    Positive advantage for action a means: taking a is better than the
    mixed strategy baseline.  Used exclusively during training (regret
    matching derives the current strategy from these estimates).
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        hidden: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__(obs_size, action_size, hidden)

    @torch.inference_mode()
    def predict(self, obs_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Return advantage vector; illegal actions are zeroed.

        ``inference_mode`` is slightly faster than ``no_grad`` because it skips
        the autograd version counter bookkeeping entirely — meaningful when this
        function is called ~80k times per CFR iteration.
        """
        obs_t = torch.from_numpy(obs_np).unsqueeze(0)
        adv = self.forward(obs_t)[0].numpy()
        adv = adv.copy()  # inference_mode returns a read-only view; we need writable
        adv[~mask_np] = 0.0
        return adv


class StrategyNet(_MLP):
    """Maps observation → probability distribution over legal actions.

    Trained to imitate the average CFR strategy (the Nash approximation).
    This is the network used at inference time by CFRAgent.
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        hidden: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__(obs_size, action_size, hidden)

    @torch.inference_mode()
    def predict(
        self,
        obs_np: np.ndarray,
        mask_np: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        dev = next(self.parameters()).device
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(dev)
        logits = self.forward(obs_t)[0].clone()  # clone so masked_fill is writable
        mask_t = torch.from_numpy(mask_np).to(dev)
        logits[~mask_t] = float("-inf")
        if deterministic:
            return int(logits.argmax().item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.inference_mode()
    def predict_probs(self, obs_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Return softmax probability distribution over all actions (illegal=0)."""
        dev = next(self.parameters()).device
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(dev)
        logits = self.forward(obs_t)[0].clone()
        mask_t = torch.from_numpy(mask_np).to(dev)
        logits[~mask_t] = float("-inf")
        probs = torch.softmax(logits, dim=-1).cpu().numpy().copy()
        probs[~mask_np] = 0.0
        return probs


# ---------------------------------------------------------------------------
# Split-network variants
# ---------------------------------------------------------------------------


class BiddingAdvNet(AdvantageNet):
    """Advantage network for bidding decisions only (output size = 11)."""

    def __init__(self, obs_size: int = OBS_SIZE, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__(obs_size=obs_size, action_size=BID_ACTION_SIZE, hidden=hidden)


class BiddingStratNet(StrategyNet):
    """Strategy network for bidding decisions only (output size = 11)."""

    def __init__(self, obs_size: int = OBS_SIZE, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__(obs_size=obs_size, action_size=BID_ACTION_SIZE, hidden=hidden)


class PlayingAdvNet(AdvantageNet):
    """Advantage network for card-play decisions only (output size = 71)."""

    def __init__(self, obs_size: int = OBS_SIZE, hidden: tuple[int, ...] = (512, 512)) -> None:
        super().__init__(obs_size=obs_size, action_size=PLAY_ACTION_SIZE, hidden=hidden)


class PlayingStratNet(StrategyNet):
    """Strategy network for card-play decisions only (output size = 71)."""

    def __init__(self, obs_size: int = OBS_SIZE, hidden: tuple[int, ...] = (512, 512)) -> None:
        super().__init__(obs_size=obs_size, action_size=PLAY_ACTION_SIZE, hidden=hidden)
