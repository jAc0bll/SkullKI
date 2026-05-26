"""Python-side agent that wraps a trained PolicyValueNet for play.

Encodes the observation, masks illegal actions, and chooses by argmax
(deterministic) or temperature-sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
sys.path.insert(0, str(REPO_ROOT / "train"))

import skullking as sk  # noqa: E402
from model import PolicyValueNet, masked_log_softmax  # noqa: E402


class NNAgent:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu",
                 temperature: float = 0.0):
        """
        temperature == 0  → deterministic argmax.
        temperature > 0   → categorical sample from softmax(logits/T) over legal actions.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.model = PolicyValueNet(hidden=ckpt.get("hidden", 512)).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.temperature = float(temperature)
        self.name = "nn"

    @torch.no_grad()
    def select_action(self, state, rng) -> "sk.Action":
        legals = sk.legal_actions(state)
        legal_idxs = [sk.action_to_index(a) for a in legals]

        obs   = sk.observe(state, state.current_player)
        feats = sk.encode(obs)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        logits, _ = self.model(x)
        logits = logits.squeeze(0).cpu().numpy()

        mask = np.full(sk.ACTION_DIM, -np.inf, dtype=np.float32)
        for i in legal_idxs:
            mask[i] = 0.0
        masked = logits + mask

        if self.temperature <= 0.0:
            chosen_idx = int(np.argmax(masked))
        else:
            scaled = masked / self.temperature
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            chosen_idx = int(np.random.choice(sk.ACTION_DIM, p=probs))

        # The chosen idx must be one of the legals.
        return sk.index_to_action(chosen_idx)
