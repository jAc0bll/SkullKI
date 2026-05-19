"""Average-strategy network and supervised-learning trainer for NFSP.

The AverageStrategyNet imitates the best-response (PPO) policy by minimising
cross-entropy loss on the reservoir of past BR decisions.  Over time its
output distribution converges to the *average* of all BR policies seen during
training — which is the NFSP Nash approximation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE, OBS_SIZE
from skull_king.training.nfsp_reservoir import ReservoirBuffer


class AverageStrategyNet(nn.Module):
    """MLP that maps an observation to a distribution over actions."""

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        hidden: tuple[int, ...] = (256, 256),
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

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        mask_np: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        logits = self.forward(obs_t)[0]
        mask_t = torch.BoolTensor(mask_np)
        logits[~mask_t] = float("-inf")
        if deterministic:
            return int(logits.argmax().item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


class SLTrainer:
    """Trains AverageStrategyNet via cross-entropy on reservoir samples."""

    def __init__(
        self,
        net: AverageStrategyNet,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> None:
        self.net = net
        self.batch_size = batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def update(self, reservoir: ReservoirBuffer, n_updates: int = 8) -> float:
        if len(reservoir) < self.batch_size:
            return 0.0
        self.net.train()
        total_loss = 0.0
        for _ in range(n_updates):
            obs, masks, actions = reservoir.sample(self.batch_size)
            obs_t = torch.FloatTensor(obs).to(self.device)
            mask_t = torch.BoolTensor(masks).to(self.device)
            act_t = torch.LongTensor(actions).to(self.device)

            logits = self.net(obs_t)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            loss = nn.functional.cross_entropy(logits, act_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.net.eval()
        return total_loss / n_updates
