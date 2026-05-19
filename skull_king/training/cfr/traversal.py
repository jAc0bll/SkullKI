"""Deep CFR traversal — outcome-sampling Monte Carlo CFR.

One call to ``traverse`` plays a single full game.  At each decision node
of the *traversing player* the instantaneous counterfactual regret is
recorded; at all other players' nodes a single action is sampled according
to the current strategy.

The collected samples are used to train:
  - AdvantageNet  (regret targets  → MSE on taken action)
  - StrategyNet   (strategy targets → cross-entropy at every visited node)

Workers are self-contained (no shared state) so they run safely in
separate processes via multiprocessing.Pool.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from skull_king.cards import Card, TigressMode
from skull_king.engine import GameEngine
from skull_king.env.skull_king_env import (
    ACTION_SPACE_SIZE,
    N_BID_ACTIONS,
    SkullKingEnv,
    TIGRESS_AS_ESCAPE_ACTION,
    TIGRESS_AS_PIRATE_ACTION,
    _CANONICAL_DECK,
)
from skull_king.game_state import GamePhase
from skull_king.training.cfr.networks import AdvantageNet, StrategyNet, regret_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_utility(state, player: int) -> float:
    """Rank-based utility normalised to roughly [-1, 1]."""
    scores = [ps.total_score for ps in state.player_states]
    my_score = scores[player]
    sorted_unique = sorted(set(scores), reverse=True)
    rank = sorted_unique.index(my_score)
    _RANK_BONUS = (0.50, 0.15, -0.10, -0.30, -0.40, -0.45)
    return float(my_score) / 300.0 + _RANK_BONUS[min(rank, len(_RANK_BONUS) - 1)]


def _decode_action(
    action: int,
) -> tuple[Optional[Card], Optional[TigressMode], Optional[int]]:
    """Returns (card, tigress_mode, bid).  Exactly one of bid / card is set."""
    if action < N_BID_ACTIONS:
        return None, None, action
    if action == TIGRESS_AS_ESCAPE_ACTION:
        return _CANONICAL_DECK[69], TigressMode.ESCAPE, None
    if action == TIGRESS_AS_PIRATE_ACTION:
        return _CANONICAL_DECK[69], TigressMode.PIRATE, None
    return _CANONICAL_DECK[action - N_BID_ACTIONS], None, None


# ---------------------------------------------------------------------------
# Worker entry point (must be a top-level function for multiprocessing)
# ---------------------------------------------------------------------------

# Module-level weights set once per worker process via Pool initializer.
_ADV_WEIGHTS: dict = {}
_STRAT_WEIGHTS: dict = {}


def worker_init(adv_weights: dict, strat_weights: dict) -> None:
    """Called once per worker process to cache network weights."""
    import torch
    # After fork(), only the calling thread survives → MKL/OpenMP pools are dead.
    # Force single-threaded torch to avoid deadlock on any linear layer call.
    torch.set_num_threads(1)
    global _ADV_WEIGHTS, _STRAT_WEIGHTS
    _ADV_WEIGHTS = adv_weights
    _STRAT_WEIGHTS = strat_weights


def worker_task(args: tuple) -> tuple[list, list]:
    """Unpack args and run one traversal using cached weights."""
    traverser, seed, n_players = args
    return traverse(traverser, _ADV_WEIGHTS, _STRAT_WEIGHTS, seed, n_players)


# ---------------------------------------------------------------------------
# Core traversal
# ---------------------------------------------------------------------------

def traverse(
    traverser: int,
    adv_weights: dict,
    strat_weights: dict,
    seed: int,
    n_players: int = 4,
) -> tuple[list, list]:
    """Run one outcome-sampling CFR traversal.

    Parameters
    ----------
    traverser:
        The player whose regrets are updated this traversal.
    adv_weights / strat_weights:
        CPU state-dicts for the current advantage and strategy networks.
    seed:
        RNG seed for both card dealing and action sampling.

    Returns
    -------
    adv_samples:
        list of (obs, mask, adv_target, action)  — one per traverser decision
    strat_samples:
        list of (obs, mask, strategy)             — one per ALL players' decisions
    """
    # Build local network instances — no shared state between workers.
    adv_net = AdvantageNet()
    adv_net.load_state_dict(adv_weights)
    adv_net.eval()

    strat_net = StrategyNet()
    strat_net.load_state_dict(strat_weights)
    strat_net.eval()

    # Utility env used only for obs / mask building, never stepped.
    util_env = SkullKingEnv(n_players=n_players)

    rng = np.random.default_rng(int(seed))
    engine = GameEngine(n_players=n_players, seed=int(seed))
    state = engine.start()

    traverser_history: list[tuple] = []  # (obs, mask, action, adv_est)
    strat_samples: list[tuple] = []

    # ── Play one full game ────────────────────────────────────────────────
    while state.phase != GamePhase.GAME_OVER:
        p = state.current_player_index
        completed = engine.completed_tricks_this_round
        obs = util_env._build_observation_for(state, p, completed)
        mask = util_env._action_masks_for(state, p)

        # Advantage network → regret matching → current strategy
        adv_est = adv_net.predict(obs, mask)
        strategy = regret_match(adv_est, mask)

        strat_samples.append((obs.copy(), mask.copy(), strategy.copy()))

        # Sample one action
        legal_idx = np.where(mask)[0]
        probs = strategy[legal_idx]
        probs = probs / probs.sum()  # guard against float rounding
        action = int(rng.choice(legal_idx, p=probs))

        if p == traverser:
            traverser_history.append((obs.copy(), mask.copy(), action, adv_est.copy()))

        # Advance game state
        card, mode, bid = _decode_action(action)
        if bid is not None:
            state = engine.place_bid(p, bid)
        else:
            state = engine.play_card(p, card, mode)

    # ── Compute utility for traverser ─────────────────────────────────────
    utility = _compute_utility(state, traverser)

    # ── Compute advantage targets ──────────────────────────────────────────
    # For each traverser decision: instantaneous regret = utility - baseline.
    # baseline = expected value under the mixed strategy at that node.
    adv_samples: list[tuple] = []
    for obs, mask, action, adv_est in traverser_history:
        strat = regret_match(adv_est, mask)
        legal_idx = np.where(mask)[0]
        baseline = float(np.dot(strat[legal_idx], adv_est[legal_idx]))

        adv_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        adv_target[action] = utility - baseline

        adv_samples.append((obs, mask, adv_target, action))

    return adv_samples, strat_samples
