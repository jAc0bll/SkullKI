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
from skull_king.training.cfr.networks import AdvantageNet, regret_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Normalisation for the score-vs-field utility.  Empirically the typical gap
# between winner and field-average is ~30·n_players points in Skull King
# (a 4-player game has ~120 gap, a 6-player game has ~180 gap).  This keeps
# utilities roughly in [-1, +1] regardless of player count.
_UTILITY_SCALE_PER_PLAYER = 30.0


def _compute_utility(state, player: int) -> float:
    """Utility = score relative to opponents' average, scaled by n_players.

    Incentivises maximising absolute score AND beating the field.
    Dividing by ``30·n_players`` keeps values roughly in [-1, +1] for any
    player count.  The old rank-only signal caused the agent to converge on
    bid=2 as a 'safe' strategy that avoids losing badly but never scores well.
    """
    scores = [ps.total_score for ps in state.player_states]
    return _utility_from_scores(scores, player)


def _compute_utility_from_engine(engine, player: int) -> float:
    """Fast variant that reads engine internals directly (no GameState freeze)."""
    scores = [p.total_score for p in engine._players]
    return _utility_from_scores(scores, player)


def _utility_from_scores(scores: list[float], player: int) -> float:
    my_score = float(scores[player])
    n = len(scores)
    avg_others = sum(scores[i] for i in range(n) if i != player) / (n - 1)
    return (my_score - avg_others) / (_UTILITY_SCALE_PER_PLAYER * n)


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

# Network objects built once per worker process via Pool initializer.
# Avoids rebuilding + load_state_dict on every traversal call.
#
# Note: only AdvantageNet is needed for traversal.  StrategyNet is trained
# from regret-matched targets and used only at inference time outside the
# worker, so we never ship strat weights through IPC (saves ~3.4 MB × N workers
# per iteration, ~150 MB on a 44-worker server).
_ADV_NET: Optional[AdvantageNet] = None

# Utility env cached per worker for obs/mask building.
# Allocating a fresh SkullKingEnv per traversal is wasteful — only the player
# count differs, so we initialise once at worker startup.
_UTIL_ENV: Optional[SkullKingEnv] = None

# Shared-memory weight broadcast:
#   _SHARED_W   — Manager().dict() holding the current adv-net state dict
#   _SHARED_V   — ctx.Value('Q', 0), bumped each iteration the main process
#                 publishes new weights.  Reads are atomic (lockless) and fast.
#   _LOCAL_V    — per-worker copy of the version we last loaded; reloads only
#                 happen when the shared version moves ahead of ours.
#
# This replaces the older `pool.map(worker_update_nets, …)` broadcast which
# could leave some workers with stale weights when one worker pulled multiple
# update tasks before peers pulled any.
_SHARED_W = None         # type: ignore[assignment]
_SHARED_V = None         # type: ignore[assignment]
_LOCAL_V: int = -1

# Heuristic opponent mixing — set during worker_init.
_HEURISTIC_FRAC: float = 0.0       # fraction of opponent decisions using HeuristicAgent
_HEURISTIC = None                   # type: ignore[assignment]

# Card._hash → first canonical play-action slot index.
# Precomputed at module load so _heuristic_action() avoids per-call dict lookups.
# Tigress (_hash=60) is excluded — it uses TIGRESS_AS_* dedicated actions.
_HASH_TO_FIRST_SLOT: tuple[int, ...] = tuple(
    N_BID_ACTIONS + next(
        (slot for slot, c in enumerate(_CANONICAL_DECK) if c._hash == h and slot < 69),
        -1
    )
    for h in range(61)
)


def _hidden_from_weights(weights: dict) -> tuple[int, ...]:
    """Infer MLP hidden layer sizes from a state dict.

    Layer pattern: net.0.weight [h1, in], net.2.weight [h2, h1], net.4.weight [out, h2].
    All layers except the last are hidden layers.
    """
    hidden = []
    i = 0
    while True:
        key = f"net.{i * 2}.weight"
        next_key = f"net.{(i + 1) * 2}.weight"
        if key not in weights:
            break
        if next_key in weights:  # not the output layer
            hidden.append(weights[key].shape[0])
        i += 1
    return tuple(hidden)


def worker_init(
    adv_weights: dict,
    n_players: int,
    shared_w=None,
    shared_v=None,
    heuristic_frac: float = 0.0,
) -> None:
    """Called once per worker process: build adv-net, util env, install shared-
    memory handles for the lock-free weight broadcast.

    ``adv_weights`` is the iter-0 state dict (used to initialise the net before
    the first broadcast arrives).  ``shared_w`` / ``shared_v`` are the manager-
    backed dict and Value('Q') used for subsequent broadcasts.  Single-process
    mode (called directly without a pool) passes ``None`` for the shared args.
    """
    import torch
    # After fork(), only the calling thread survives → MKL/OpenMP pools are dead.
    # Force single-threaded torch to avoid deadlock on any linear layer call.
    torch.set_num_threads(1)
    global _ADV_NET, _UTIL_ENV, _SHARED_W, _SHARED_V, _LOCAL_V
    global _HEURISTIC, _HEURISTIC_FRAC
    _HEURISTIC_FRAC = heuristic_frac
    if heuristic_frac > 0.0:
        from skull_king.agents import HeuristicAgent as _HA
        _HEURISTIC = _HA()
    else:
        _HEURISTIC = None
    hidden = _hidden_from_weights(adv_weights)
    _ADV_NET = AdvantageNet(hidden=hidden)
    _ADV_NET.load_state_dict(adv_weights)
    _ADV_NET.eval()
    _UTIL_ENV = SkullKingEnv(n_players=n_players)
    _SHARED_W = shared_w
    _SHARED_V = shared_v
    _LOCAL_V = 0  # iter-0 weights already loaded


def _maybe_reload_weights() -> None:
    """Pull the latest adv-weights from shared memory if our local copy is stale.

    Fast path (no broadcast pending) is a single atomic int read on the Value;
    the slow path (one IPC fetch + load_state_dict) runs at most once per
    worker per iteration.
    """
    global _LOCAL_V
    if _SHARED_V is None or _SHARED_W is None or _ADV_NET is None:
        return
    v = _SHARED_V.value
    if v > _LOCAL_V:
        weights = _SHARED_W["weights"]
        _ADV_NET.load_state_dict(weights)
        _LOCAL_V = v


def worker_task(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray]:
    """Unpack args and run one traversal using cached network + env.

    Calls ``_maybe_reload_weights`` first so each worker is guaranteed to see
    the latest adv-net weights before the very first traversal of an iteration
    — no broadcast race possible.
    """
    _maybe_reload_weights()
    traverser, seed, n_players = args
    return traverse(traverser, _ADV_NET, _UTIL_ENV, seed, n_players)


# ---------------------------------------------------------------------------
# Core traversal
# ---------------------------------------------------------------------------

def _heuristic_action(engine, p: int) -> int:
    """Ask the cached HeuristicAgent to pick an action for opponent player p.

    Calls engine.get_state() once (O(n_players) freeze) then delegates to
    the rule-based heuristic.  Only called for the fraction of opponent
    decisions controlled by _HEURISTIC_FRAC, so the amortised overhead is
    acceptable.
    """
    from skull_king.game_state import GamePhase as _GP
    state = engine.get_state()
    if engine._phase == _GP.BIDDING:
        return _HEURISTIC.bid(state, p)   # bid value == action index directly
    card, mode = _HEURISTIC.play(state, p)
    if mode == TigressMode.PIRATE:
        return TIGRESS_AS_PIRATE_ACTION
    if mode == TigressMode.ESCAPE:
        return TIGRESS_AS_ESCAPE_ACTION
    return _HASH_TO_FIRST_SLOT[card._hash]


def traverse(
    traverser: int,
    adv_net: AdvantageNet,
    util_env: SkullKingEnv,
    seed: int,
    n_players: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Run one outcome-sampling CFR traversal.

    Parameters
    ----------
    traverser:
        The player whose regrets are updated this traversal.
    adv_net:
        Advantage network (built once per worker, not per traversal).
    util_env:
        Pre-allocated SkullKingEnv used only for obs/mask building.
    seed:
        RNG seed for both card dealing and action sampling.

    Returns
    -------
    A tuple of 7 stacked ndarrays:
        adv_obs, adv_masks, adv_targets, adv_actions   (traverser decisions)
        strat_obs, strat_masks, strat_strategies       (all players' decisions)

    Returning ndarrays directly (instead of Python tuple lists) lets the main
    process write them into the replay buffer with one slice assignment per
    field instead of millions of per-sample Python loops.
    """
    rng = np.random.default_rng(int(seed))
    engine = GameEngine(n_players=n_players, seed=int(seed))
    state = engine.start()

    # Collected per decision.  We keep both observations + masks + strategies
    # for the strategy buffer, plus indices into that list for traverser-only
    # decisions (so we don't copy obs/mask twice).
    s_obs: list[np.ndarray] = []
    s_masks: list[np.ndarray] = []
    s_strats: list[np.ndarray] = []
    # For each traverser decision we record (index_into_s_lists, action, adv_est).
    # adv_est is needed later to compute the strategy baseline; obs/mask are
    # shared with s_obs/s_masks (no double-copy).
    t_indices: list[int] = []
    t_actions: list[int] = []
    t_adv_ests: list[np.ndarray] = []

    # ── Play one full game ────────────────────────────────────────────────
    # Loop reads engine internals directly via the env's fast-path methods
    # instead of going through engine.get_state() each step — saves ~10% of
    # CFR time by skipping FrozenPlayerState construction for every player
    # on every move (the env methods are pure-read, single-threaded worker,
    # safe to access mutable engine state).
    while engine._phase != GamePhase.GAME_OVER:
        p = engine._current_player_index()
        obs = util_env._build_observation_from_engine(engine, p)
        mask = util_env._action_masks_from_engine(engine, p)

        # Advantage network → regret matching → current strategy
        adv_est = adv_net.predict(obs, mask)
        strategy = regret_match(adv_est, mask)

        s_obs.append(obs)
        s_masks.append(mask)
        s_strats.append(strategy)

        # Sample one legal action.  searchsorted on cumsum is ~3-5x faster than
        # rng.choice(arr, p=...) which validates probs and allocates per call.
        legal_idx = np.where(mask)[0]
        probs = strategy[legal_idx]
        probs = probs / probs.sum()  # guard against float rounding
        cum = probs.cumsum()
        r = float(rng.random())
        idx_in_legal = int(np.searchsorted(cum, r, side="right"))
        if idx_in_legal >= legal_idx.size:
            idx_in_legal = legal_idx.size - 1  # guard r == 1.0 edge case
        action = int(legal_idx[idx_in_legal])

        if p == traverser:
            t_indices.append(len(s_obs) - 1)
            t_actions.append(action)
            t_adv_ests.append(adv_est)
        elif _HEURISTIC is not None and float(rng.random()) < _HEURISTIC_FRAC:
            # Replace opponent action with heuristic policy for richer training signal.
            try:
                action = _heuristic_action(engine, p)
            except Exception:
                pass  # fall back to strategy-sampled action on any error

        # Advance game state via the *_no_state variants — they skip the
        # ``return self.get_state()`` step inside the engine and avoid
        # freezing all player states every move.  Engine internals are still
        # mutated correctly; the next loop iteration reads them directly.
        card, mode, bid = _decode_action(action)
        if bid is not None:
            engine.place_bid_no_state(p, bid)
        else:
            engine.play_card_no_state(p, card, mode)

    # ── Compute utility for traverser ─────────────────────────────────────
    utility = _compute_utility_from_engine(engine, traverser)

    # ── Stack strategy samples (one per decision, all players) ────────────
    n_strat = len(s_obs)
    strat_obs = np.stack(s_obs) if n_strat else np.empty((0, util_env.observation_space.shape[0]), dtype=np.float32)
    strat_masks = np.stack(s_masks) if n_strat else np.empty((0, ACTION_SPACE_SIZE), dtype=bool)
    strat_strategies = np.stack(s_strats) if n_strat else np.empty((0, ACTION_SPACE_SIZE), dtype=np.float32)

    # ── Compute advantage targets for traverser decisions ─────────────────
    # baseline = expected value under the regret-matched strategy at that node.
    n_adv = len(t_indices)
    if n_adv == 0:
        adv_obs = np.empty((0, strat_obs.shape[1]), dtype=np.float32)
        adv_masks = np.empty((0, ACTION_SPACE_SIZE), dtype=bool)
        adv_targets = np.empty((0, ACTION_SPACE_SIZE), dtype=np.float32)
        adv_actions = np.empty((0,), dtype=np.int64)
    else:
        adv_obs = strat_obs[t_indices]
        adv_masks = strat_masks[t_indices]
        adv_targets = np.zeros((n_adv, ACTION_SPACE_SIZE), dtype=np.float32)
        adv_actions = np.array(t_actions, dtype=np.int64)
        # Reuse the strategy we already computed (s_strats) — no second
        # regret_match call.
        for k in range(n_adv):
            idx      = t_indices[k]
            adv_est  = t_adv_ests[k]
            strat    = s_strats[idx]
            mask_k   = s_masks[idx]
            legal_idx = np.where(mask_k)[0]
            baseline = float(np.dot(strat[legal_idx], adv_est[legal_idx]))
            a_taken  = t_actions[k]
            # IS correction: avoid extreme targets when prob is tiny
            prob_taken = max(float(strat[a_taken]), 0.05)
            # Fill counterfactual estimate for ALL legal actions (Fix A)
            adv_targets[k, legal_idx] = adv_est[legal_idx] - baseline
            # Override taken action with real IS-corrected utility signal (Fix B)
            adv_targets[k, a_taken] = (utility - baseline) / prob_taken

    return adv_obs, adv_masks, adv_targets, adv_actions, strat_obs, strat_masks, strat_strategies
