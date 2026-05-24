"""ReBeL value and policy networks.

Value network  V(pbs_enc) → [v_0, ..., v_{n-1}]
    Expected utility for each player given the public belief state.
    Trained on actual game outcomes from self-play.

Policy network π(pbs_enc, mask) → action log-probabilities
    Strategy for the current player given the public belief state.
    Trained via imitation of subgame-solved strategies.

Both networks condition on the full PBS encoding (public info + belief),
unlike the CFR strategy nets which only see the acting player's private
observation.  This allows the agent to reason about opponent card probabilities
explicitly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skull_king.training.rebel.public_belief_state import pbs_encoding_size
from skull_king.env.skull_king_env import ACTION_SPACE_SIZE


def _mlp(in_size: int, hidden: tuple[int, ...], out_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


class RebelValueNet(nn.Module):
    """Maps a PBS encoding to per-player expected utilities.

    Uses LayerNorm for stability since PBS inputs span very different scales
    (binary masks, normalized scores, probability distributions).
    """

    def __init__(
        self,
        n_players: int,
        hidden: tuple[int, ...] = (512, 512, 256),
    ) -> None:
        super().__init__()
        in_size = pbs_encoding_size(n_players)
        self.net = _mlp(in_size, hidden, n_players)
        self.n_players = n_players

    def forward(self, pbs_enc: torch.Tensor) -> torch.Tensor:
        """pbs_enc: [B, pbs_size] → [B, n_players]"""
        return self.net(pbs_enc)


class RebelPolicyNet(nn.Module):
    """Maps a PBS encoding to a masked action distribution.

    Separate from the CFR strategy net: conditions on belief about all
    players' cards rather than just the acting player's private observation.
    """

    def __init__(
        self,
        n_players: int,
        hidden: tuple[int, ...] = (512, 512, 256),
    ) -> None:
        super().__init__()
        in_size = pbs_encoding_size(n_players)
        self.net = _mlp(in_size, hidden, ACTION_SPACE_SIZE)
        self.n_players = n_players

    def forward(
        self,
        pbs_enc: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pbs_enc:     [B, pbs_size]
        action_mask: [B, ACTION_SPACE_SIZE] bool — True = legal action
        Returns:     [B, ACTION_SPACE_SIZE] log-softmax probabilities
        """
        logits = self.net(pbs_enc).masked_fill(~action_mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def strategy(
        self,
        pbs_enc: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities (not log). For inference/rollouts."""
        logits = self.net(pbs_enc).masked_fill(~action_mask, float("-inf"))
        return torch.softmax(logits, dim=-1)
