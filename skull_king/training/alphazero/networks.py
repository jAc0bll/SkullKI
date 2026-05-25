"""AlphaZero policy+value network for Skull King.

Single shared MLP backbone with LayerNorm + ReLU. Two heads:
  policy: log-softmax over ACTION_SPACE_SIZE legal actions
  value:  tanh-bounded scalar in [-1, +1]

The value range [-1, +1] is the standard AlphaZero convention. The
self-play runner scales raw round scores into this range before storing
them as training targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.training.rebel.public_belief_state import pbs_encoding_size


def _safe_mask_value(x: torch.Tensor) -> float:
    """fp16/fp32-safe sentinel for masked-fill (avoid -inf under AMP)."""
    return torch.finfo(x.dtype).min


class AlphaZeroNet(nn.Module):
    """Backbone MLP + (policy, value) heads."""

    def __init__(
        self,
        n_players: int,
        hidden: tuple[int, ...] = (1024, 1024, 512),
        value_hidden: int = 128,
    ) -> None:
        super().__init__()
        in_size = pbs_encoding_size(n_players)

        backbone: list[nn.Module] = []
        prev = in_size
        for h in hidden:
            backbone.append(nn.Linear(prev, h))
            backbone.append(nn.LayerNorm(h))
            backbone.append(nn.ReLU(inplace=True))
            prev = h
        self.backbone = nn.Sequential(*backbone)

        self.policy_head = nn.Linear(prev, ACTION_SPACE_SIZE)

        self.value_head = nn.Sequential(
            nn.Linear(prev, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        enc: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log_probs over actions, scalar value in [-1, +1])."""
        x = self.backbone(enc)
        logits = self.policy_head(x)
        logits = logits.masked_fill(~mask, _safe_mask_value(logits))
        log_probs = F.log_softmax(logits, dim=-1)
        value = self.value_head(x).squeeze(-1)
        return log_probs, value
