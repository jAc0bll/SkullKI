"""Skull King policy/value network.

Two architectures live here:
    PolicyValueNet   — small MLP (Phase 4a BC bring-up, ~1.9M params at hidden=1024).
    PolicyValueNetV2 — residual trunk (Phase 8 large-model upgrade, ~25M params
                       at hidden=2048, num_blocks=3).

The value head is per-player (N_PLAYERS) rather than scalar because Skull King
is multi-agent and per-player rewards are required for clean PUCT-style MCTS
backup.
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


class _ResBlock(nn.Module):
    """Pre-norm residual block: x + Drop(W2 · GELU(LN2( W1 · GELU(LN1(x)) )))"""
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.nn.functional.gelu(self.ln1(x)))
        h = self.drop(h)
        h = self.fc2(torch.nn.functional.gelu(self.ln2(h)))
        return x + h


class PolicyValueNetV2(nn.Module):
    """Residual MLP — Phase 8 upgrade from the 1.9M-param baseline.

    Defaults give ~25M params (hidden=2048, num_blocks=3, dropout=0.1).
    Use --arch v2 in train.py to select.
    """
    def __init__(self, hidden: int = 2048, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input    = nn.Linear(ENC_DIM, hidden)
        self.input_ln = nn.LayerNorm(hidden)
        self.blocks   = nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(num_blocks)])
        self.out_ln   = nn.LayerNorm(hidden)
        self.policy   = nn.Linear(hidden, ACTION_DIM)
        self.value    = nn.Linear(hidden, N_PLAYERS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_ln(self.input(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.policy(h), self.value(h)


def build_model(arch: str, hidden: int, num_blocks: int = 3, dropout: float = 0.0) -> nn.Module:
    """Factory used by train.py / export.py so the choice of arch flows from
    the checkpoint metadata, not from imports scattered through the codebase."""
    if arch == "v1":
        return PolicyValueNet(hidden=hidden, num_layers=2, dropout=dropout)
    if arch == "v2":
        return PolicyValueNetV2(hidden=hidden, num_blocks=num_blocks, dropout=dropout)
    raise ValueError(f"unknown arch '{arch}' (expected 'v1' or 'v2')")


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax with hard mask: illegal entries -> -inf before softmax.

    logits     : (B, A)
    legal_mask : (B, A) bool
    Returns log-probabilities of shape (B, A), with -inf on illegal positions.
    """
    very_negative = torch.finfo(logits.dtype).min
    masked = torch.where(legal_mask, logits, torch.full_like(logits, very_negative))
    return masked.log_softmax(dim=-1)
