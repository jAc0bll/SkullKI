"""Skull King policy/value network — small MLP for Phase 4a behaviour cloning.

Architecture:
    Input  : (B, ENC_DIM=760)
    Trunk  : Linear 760 -> 512 -> 512  (LayerNorm + GELU between)
    Policy : Linear 512 -> ACTION_DIM=83  (raw logits; caller masks illegal)
    Value  : Linear 512 -> 1             (scalar; predicts normalized round value)

The architecture is deliberately small so CPU training is feasible during
pipeline bring-up. Phase 4b / GPU will swap in something larger.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Mirrors the C++ engine. Keep in sync if the binding constants change.
ENC_DIM    = 760
ACTION_DIM = 83
N_PLAYERS  = 4


class PolicyValueNet(nn.Module):
    """Outputs:
        policy_logits : (B, ACTION_DIM)  raw logits over the unified action space
        values        : (B, N_PLAYERS)   per-player expected round score (normalized by ~200)
    The value head is per-player rather than scalar because Skull King is multi-agent
    and per-player rewards are required for clean PUCT-style MCTS backup.
    """
    def __init__(self, hidden: int = 512, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(ENC_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk  = nn.Sequential(*layers)
        self.policy = nn.Linear(hidden, ACTION_DIM)
        self.value  = nn.Linear(hidden, N_PLAYERS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h)


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax with hard mask: illegal entries -> -inf before softmax.

    logits     : (B, A)
    legal_mask : (B, A) bool
    Returns log-probabilities of shape (B, A), with -inf on illegal positions.
    """
    very_negative = torch.finfo(logits.dtype).min
    masked = torch.where(legal_mask, logits, torch.full_like(logits, very_negative))
    return masked.log_softmax(dim=-1)
