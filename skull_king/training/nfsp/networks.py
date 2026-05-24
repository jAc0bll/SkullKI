"""NFSP networks: Q-net (best response) + Avg-net (average strategy)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.training.rebel.public_belief_state import pbs_encoding_size


def _mlp(in_size: int, hidden: tuple[int, ...], out_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


class NfspQNet(nn.Module):
    """Q-network: predicts expected return for each action given state."""

    def __init__(self, n_players: int, hidden: tuple[int, ...] = (512, 512, 256)):
        super().__init__()
        self.net = _mlp(pbs_encoding_size(n_players), hidden, ACTION_SPACE_SIZE)

    def forward(self, enc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.net(enc)
        return q.masked_fill(~mask, -1e9)


class NfspAvgNet(nn.Module):
    """Average-strategy network: log-softmax over legal actions."""

    def __init__(self, n_players: int, hidden: tuple[int, ...] = (512, 512, 256)):
        super().__init__()
        self.net = _mlp(pbs_encoding_size(n_players), hidden, ACTION_SPACE_SIZE)

    def forward(self, enc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(enc)
        logits = logits.masked_fill(~mask, -1e9)
        return F.log_softmax(logits, dim=-1)
