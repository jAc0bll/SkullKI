"""Q-network for DouZero-Skull-King.

Single shared MLP across all four seats. Inputs the public-belief-state
encoding from the existing PBS module; outputs Q-values over the full
action space, with illegal actions masked to a dtype-safe sentinel.

LayerNorm + ReLU between layers — value-network training is sensitive to
activation scale, LayerNorm makes it stable across LR + batch-size choices.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.training.rebel.public_belief_state import pbs_encoding_size


def _safe_mask_value(x: torch.Tensor) -> float:
    """Return the largest negative finite value in x's dtype.

    -1e9 would overflow fp16 to -inf and break log_softmax under AMP.
    finfo(dtype).min is finite and small enough that softmax/argmax over
    legal actions ignores illegal ones.
    """
    return torch.finfo(x.dtype).min


class DouZeroQNet(nn.Module):
    """Deep MLP Q-network with LayerNorm regularization."""

    def __init__(
        self,
        n_players: int,
        hidden: tuple[int, ...] = (1024, 1024, 512, 256),
    ) -> None:
        super().__init__()
        in_size = pbs_encoding_size(n_players)

        layers: list[nn.Module] = []
        prev = in_size
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, ACTION_SPACE_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, enc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.net(enc)
        return q.masked_fill(~mask, _safe_mask_value(q))
