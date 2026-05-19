"""Advantage network, strategy network, and regret matching for Deep CFR."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE


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

    @torch.no_grad()
    def predict(self, obs_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Return advantage vector; illegal actions are zeroed."""
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        adv = self.forward(obs_t)[0].numpy()
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

    @torch.no_grad()
    def predict(
        self,
        obs_np: np.ndarray,
        mask_np: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        logits = self.forward(obs_t)[0]
        mask_t = torch.BoolTensor(mask_np)
        logits[~mask_t] = float("-inf")
        if deterministic:
            return int(logits.argmax().item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())
