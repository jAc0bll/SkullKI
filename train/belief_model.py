"""Belief network: predicts opponent hand contents from an observation.

For each (player, card) pair the model outputs an independent probability
that this player holds this card. Loss is binary cross-entropy on the
non-self rows (the perspective player's hand is trivially derivable
from the observation, so we skip that row).
"""
from __future__ import annotations

import torch
import torch.nn as nn

ENC_DIM   = 760
N_PLAYERS = 4
N_CARDS   = 70


class BeliefNet(nn.Module):
    """Output shape: (B, N_PLAYERS, N_CARDS) — raw logits.
    Apply sigmoid externally to get probabilities; mask known-played cards
    before normalising in the determinizer.
    """
    def __init__(self, hidden: int = 512, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(ENC_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(hidden, N_PLAYERS * N_CARDS)
        # Stored as buffer-like attributes so TorchScript sees them as ints, not globals.
        self.n_players: int = N_PLAYERS
        self.n_cards:   int = N_CARDS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return self.head(h).view(-1, self.n_players, self.n_cards)
