"""ReBeL inference agent — uses the trained policy network for decisions."""
from __future__ import annotations

import numpy as np
import torch

from skull_king.agents.base_agent import BaseAgent
from skull_king.env.skull_king_env import (
    ACTION_SPACE_SIZE,
    N_BID_ACTIONS,
    TIGRESS_AS_ESCAPE_ACTION,
    TIGRESS_AS_PIRATE_ACTION,
    _build_obs,
    _build_action_mask,
)


class RebelAgent(BaseAgent):
    """Greedy policy-network agent for tournament evaluation."""

    def __init__(
        self,
        policy_net,
        n_players: int = 4,
        name: str = "ReBeL",
        device: torch.device | None = None,
    ) -> None:
        self.policy_net = policy_net
        self.n_players = n_players
        self._name = name
        self.device = device or next(policy_net.parameters()).device
        self.policy_net.eval()

    @property
    def name(self) -> str:
        return self._name

    def act(self, observation: np.ndarray, action_mask: np.ndarray, **kwargs) -> int:
        """Choose action greedily from policy network given observation + mask."""
        from skull_king.training.rebel.public_belief_state import pbs_encoding_size

        # Build a minimal PBS encoding from the existing observation vector
        # The policy net expects a PBS encoding; as a fallback we zero-pad.
        pbs_size = pbs_encoding_size(self.n_players)
        pbs_enc = np.zeros(pbs_size, dtype=np.float32)
        # Copy the observation into the first OBS_SIZE slots (approximate PBS)
        from skull_king.env.skull_king_env import OBS_SIZE
        copy_len = min(OBS_SIZE, pbs_size)
        pbs_enc[:copy_len] = observation[:copy_len]

        enc_t = torch.from_numpy(pbs_enc).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.policy_net(enc_t, mask_t)
            probs = torch.exp(log_probs).squeeze(0).cpu().numpy()

        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            return 0
        best = legal[np.argmax(probs[legal])]
        return int(best)
