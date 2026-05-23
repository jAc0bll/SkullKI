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

import os
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
from skull_king.training.cfr.networks import (
    AdvantageNet,
    BiddingAdvNet,
    BID_ACTION_SIZE,
    PlayingAdvNet,
    PLAY_ACTION_SIZE,
    regret_match,
)


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

# C engine backend (None when extension not built or disabled).
# When set, worker_task delegates traversal to C instead of Python.
_C_ENGINE = None         # type: ignore[assignment]

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
    global _HEURISTIC, _HEURISTIC_FRAC, _C_ENGINE
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

    # Try to initialise the C engine backend (fails gracefully if not built).
    try:
        from skull_king.cfr_engine import CEngine
        if CEngine.available:
            _C_ENGINE = CEngine(n_players=n_players, heuristic_frac=heuristic_frac)
            _C_ENGINE.load_adv_weights(adv_weights)
    except Exception:
        _C_ENGINE = None


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
        if _C_ENGINE is not None:
            _C_ENGINE.load_adv_weights(weights)
        _LOCAL_V = v


def worker_task(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray]:
    """Unpack args and run one traversal using cached network + env.

    Calls ``_maybe_reload_weights`` first so each worker is guaranteed to see
    the latest adv-net weights before the very first traversal of an iteration
    — no broadcast race possible.

    When the C engine is available (skull_king_engine extension built), delegates
    to the full C traversal (game loop + MLP inference in C, ~3-4× faster).
    Falls back to Python traversal automatically if C engine is absent.
    """
    _maybe_reload_weights()
    traverser, seed, n_players = args
    if _C_ENGINE is not None:
        return _C_ENGINE.traverse(traverser, seed)
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


# ---------------------------------------------------------------------------
# Split-network traversal (separate bidding and playing networks)
# ---------------------------------------------------------------------------

# Global vars for split workers (analogous to _ADV_NET for the unified worker).
_BID_ADV_NET: Optional[BiddingAdvNet] = None
_PLAY_ADV_NET: Optional[PlayingAdvNet] = None


def _global_to_play_mask(global_mask: np.ndarray) -> np.ndarray:
    """Extract 71-action play mask from 82-action global mask."""
    play_mask = np.empty(PLAY_ACTION_SIZE, dtype=bool)
    play_mask[:69] = global_mask[N_BID_ACTIONS:N_BID_ACTIONS + 69]
    play_mask[69] = global_mask[TIGRESS_AS_ESCAPE_ACTION]
    play_mask[70] = global_mask[TIGRESS_AS_PIRATE_ACTION]
    return play_mask


def _play_local_to_global(local: int) -> int:
    """Convert 71-action local play index to 82-action global index."""
    if local < 69:
        return local + N_BID_ACTIONS
    if local == 69:
        return TIGRESS_AS_ESCAPE_ACTION
    return TIGRESS_AS_PIRATE_ACTION


def _play_global_to_local(global_action: int) -> int:
    """Convert 82-action global play index to 71-action local index."""
    if global_action == TIGRESS_AS_ESCAPE_ACTION:
        return 69
    if global_action == TIGRESS_AS_PIRATE_ACTION:
        return 70
    return global_action - N_BID_ACTIONS


def worker_init_split(
    bid_adv_weights: dict,
    play_adv_weights: dict,
    n_players: int,
    shared_w=None,
    shared_v=None,
    heuristic_frac: float = 0.0,
) -> None:
    """Worker initialiser for split-network training.

    Builds BiddingAdvNet and PlayingAdvNet from their respective state dicts.
    Weight broadcast uses shared_w["bid_weights"] and shared_w["play_weights"].
    """
    import torch
    torch.set_num_threads(1)
    global _BID_ADV_NET, _PLAY_ADV_NET, _UTIL_ENV
    global _SHARED_W, _SHARED_V, _LOCAL_V
    global _HEURISTIC, _HEURISTIC_FRAC, _C_ENGINE

    _HEURISTIC_FRAC = heuristic_frac
    _HEURISTIC = None
    if heuristic_frac > 0.0:
        from skull_king.agents import HeuristicAgent as _HA
        _HEURISTIC = _HA()

    bid_hidden = _hidden_from_weights(bid_adv_weights)
    play_hidden = _hidden_from_weights(play_adv_weights)
    _BID_ADV_NET = BiddingAdvNet(hidden=bid_hidden)
    _BID_ADV_NET.load_state_dict(bid_adv_weights)
    _BID_ADV_NET.eval()
    _PLAY_ADV_NET = PlayingAdvNet(hidden=play_hidden)
    _PLAY_ADV_NET.load_state_dict(play_adv_weights)
    _PLAY_ADV_NET.eval()

    _UTIL_ENV = SkullKingEnv(n_players=n_players)
    _SHARED_W = shared_w
    _SHARED_V = shared_v
    _LOCAL_V = 0
    try:
        from skull_king.cfr_engine import SplitCEngine
        if SplitCEngine.available:
            _C_ENGINE = SplitCEngine(n_players=n_players, heuristic_frac=heuristic_frac)
            _C_ENGINE.load_weights(bid_adv_weights, play_adv_weights)
        else:
            _C_ENGINE = None
    except Exception:
        _C_ENGINE = None


def _maybe_reload_weights_split() -> None:
    """Pull bid+play adv weights from shared memory if stale."""
    global _LOCAL_V
    if _SHARED_V is None or _SHARED_W is None:
        return
    v = _SHARED_V.value
    if v > _LOCAL_V:
        w = _SHARED_W["weights"]
        if _BID_ADV_NET is not None:
            _BID_ADV_NET.load_state_dict(w["bid_weights"])
        if _PLAY_ADV_NET is not None:
            _PLAY_ADV_NET.load_state_dict(w["play_weights"])
        if _C_ENGINE is not None:
            _C_ENGINE.load_weights(w["bid_weights"], w["play_weights"])
        _LOCAL_V = v


def worker_task_split(args: tuple) -> tuple:
    """Split-network variant of worker_task. Returns 14 arrays."""
    _maybe_reload_weights_split()
    traverser, seed, n_players = args
    if _C_ENGINE is not None:
        return _C_ENGINE.traverse(traverser, int(seed))
    return traverse_split(traverser, _BID_ADV_NET, _PLAY_ADV_NET, _UTIL_ENV, seed, n_players)


_WORKER_TIMED = False   # print timing once per worker process
_WORKER_BATCH_IDX = 0  # counter for unique filenames per worker
_SHM_AVAILABLE = os.path.exists("/dev/shm")

_ARRAY_KEYS = [
    'b_ao', 'b_am', 'b_at', 'b_aa',
    'b_so', 'b_sm', 'b_ss',
    'p_ao', 'p_am', 'p_at', 'p_aa',
    'p_so', 'p_sm', 'p_ss',
]


def worker_batch_task_split(args_list: list):
    """Process a batch of traversals; return 14 concatenated arrays or /dev/shm path.

    When /dev/shm is available, writes results there and returns a path string
    (~200 bytes IPC) instead of ~43 MB of pickled numpy arrays. The main
    process reads the file and deletes it. This eliminates the sequential
    pickle bottleneck in multiprocessing.Queue (100 × 43 MB → 100 × 200 B).
    """
    global _WORKER_TIMED, _WORKER_BATCH_IDX
    import time, sys
    _maybe_reload_weights_split()
    t0 = time.time()
    results = []
    for traverser, seed, n_players in args_list:
        if _C_ENGINE is not None:
            results.append(_C_ENGINE.traverse(traverser, int(seed)))
        else:
            results.append(
                traverse_split(traverser, _BID_ADV_NET, _PLAY_ADV_NET,
                               _UTIL_ENV, seed, n_players)
            )
    trav = time.time() - t0
    if not _WORKER_TIMED:
        backend = "C" if _C_ENGINE is not None else "Python"
        ms = trav / len(args_list) * 1000
        print(f"  [worker pid={os.getpid()}] {backend} {len(args_list)} trav "
              f"in {trav:.2f}s = {ms:.1f}ms each", file=sys.stderr, flush=True)
        _WORKER_TIMED = True

    arrays = tuple(
        np.concatenate([r[i] for r in results], axis=0)
        for i in range(14)
    )

    if _SHM_AVAILABLE:
        _WORKER_BATCH_IDX += 1
        fname = f"/dev/shm/cfr_{os.getpid()}_{_WORKER_BATCH_IDX}.npz"
        np.savez(fname, **dict(zip(_ARRAY_KEYS, arrays)))
        return fname   # tiny string through IPC instead of 43 MB

    return arrays      # fallback: direct pickle (slow, but safe)


def traverse_split(
    traverser: int,
    bid_adv_net: BiddingAdvNet,
    play_adv_net: PlayingAdvNet,
    util_env: SkullKingEnv,
    seed: int,
    n_players: int = 4,
) -> tuple:
    """Outcome-sampling CFR traversal with separate bidding and playing networks.

    Returns 14 stacked ndarrays grouped as:
        bid_adv_obs, bid_adv_masks, bid_adv_targets, bid_adv_actions,
        bid_strat_obs, bid_strat_masks, bid_strat_strategies,
        play_adv_obs, play_adv_masks, play_adv_targets, play_adv_actions,
        play_strat_obs, play_strat_masks, play_strat_strategies

    Bid samples use local action indices 0..10 (bid value == action index).
    Play samples use local action indices 0..70 (0..68 = card slots,
    69 = Tigress-ESCAPE, 70 = Tigress-PIRATE).
    """
    rng = np.random.default_rng(int(seed))
    engine = GameEngine(n_players=n_players, seed=int(seed))
    engine.start()

    # Strategy samples (all players)
    bid_s_obs: list[np.ndarray] = []
    bid_s_masks: list[np.ndarray] = []
    bid_s_strats: list[np.ndarray] = []
    play_s_obs: list[np.ndarray] = []
    play_s_masks: list[np.ndarray] = []
    play_s_strats: list[np.ndarray] = []

    # Traverser advantage tracking
    bid_t_idx: list[int] = []
    bid_t_act: list[int] = []
    bid_t_adv: list[np.ndarray] = []
    play_t_idx: list[int] = []
    play_t_act: list[int] = []
    play_t_adv: list[np.ndarray] = []

    while engine._phase != GamePhase.GAME_OVER:
        p = engine._current_player_index()
        obs = util_env._build_observation_from_engine(engine, p)
        global_mask = util_env._action_masks_from_engine(engine, p)

        if engine._phase == GamePhase.BIDDING:
            bid_mask = global_mask[:BID_ACTION_SIZE]
            adv_est = bid_adv_net.predict(obs, bid_mask)
            strategy = regret_match(adv_est, bid_mask)

            bid_s_obs.append(obs)
            bid_s_masks.append(bid_mask)
            bid_s_strats.append(strategy)

            legal_idx = np.where(bid_mask)[0]
            probs = strategy[legal_idx]
            probs = probs / probs.sum()
            cum = probs.cumsum()
            r = float(rng.random())
            i_legal = int(np.searchsorted(cum, r, side="right"))
            if i_legal >= len(legal_idx):
                i_legal = len(legal_idx) - 1
            action_local = int(legal_idx[i_legal])  # bid value == local index

            if p == traverser:
                bid_t_idx.append(len(bid_s_obs) - 1)
                bid_t_act.append(action_local)
                bid_t_adv.append(adv_est)
            elif _HEURISTIC is not None and float(rng.random()) < _HEURISTIC_FRAC:
                try:
                    action_local = _heuristic_action(engine, p)  # returns bid (=local)
                except Exception:
                    pass

            engine.place_bid_no_state(p, action_local)

        else:  # PLAYING
            play_mask = _global_to_play_mask(global_mask)
            adv_est = play_adv_net.predict(obs, play_mask)
            strategy = regret_match(adv_est, play_mask)

            play_s_obs.append(obs)
            play_s_masks.append(play_mask)
            play_s_strats.append(strategy)

            legal_idx = np.where(play_mask)[0]
            probs = strategy[legal_idx]
            probs = probs / probs.sum()
            cum = probs.cumsum()
            r = float(rng.random())
            i_legal = int(np.searchsorted(cum, r, side="right"))
            if i_legal >= len(legal_idx):
                i_legal = len(legal_idx) - 1
            action_local = int(legal_idx[i_legal])
            action_global = _play_local_to_global(action_local)

            if p == traverser:
                play_t_idx.append(len(play_s_obs) - 1)
                play_t_act.append(action_local)
                play_t_adv.append(adv_est)
            elif _HEURISTIC is not None and float(rng.random()) < _HEURISTIC_FRAC:
                try:
                    heur_global = _heuristic_action(engine, p)
                    if heur_global >= N_BID_ACTIONS:
                        action_local = _play_global_to_local(heur_global)
                        action_global = heur_global
                except Exception:
                    pass

            card, mode, _ = _decode_action(action_global)
            engine.play_card_no_state(p, card, mode)

    utility = _compute_utility_from_engine(engine, traverser)

    # ── Build bid advantage targets ────────────────────────────────────────
    obs_dim = util_env.observation_space.shape[0]
    n_bid_adv = len(bid_t_idx)
    if n_bid_adv == 0:
        bid_adv_obs = np.empty((0, obs_dim), dtype=np.float32)
        bid_adv_masks = np.empty((0, BID_ACTION_SIZE), dtype=bool)
        bid_adv_targets = np.empty((0, BID_ACTION_SIZE), dtype=np.float32)
        bid_adv_actions = np.empty((0,), dtype=np.int64)
    else:
        bid_adv_obs = np.stack([bid_s_obs[i] for i in bid_t_idx])
        bid_adv_masks = np.stack([bid_s_masks[i] for i in bid_t_idx])
        bid_adv_targets = np.zeros((n_bid_adv, BID_ACTION_SIZE), dtype=np.float32)
        bid_adv_actions = np.array(bid_t_act, dtype=np.int64)
        for k in range(n_bid_adv):
            adv_est = bid_t_adv[k]
            strat = bid_s_strats[bid_t_idx[k]]
            mask_k = bid_s_masks[bid_t_idx[k]]
            legal = np.where(mask_k)[0]
            baseline = float(np.dot(strat[legal], adv_est[legal]))
            a = bid_t_act[k]
            prob = max(float(strat[a]), 0.05)
            bid_adv_targets[k, legal] = adv_est[legal] - baseline
            bid_adv_targets[k, a] = (utility - baseline) / prob

    # ── Build bid strategy arrays ──────────────────────────────────────────
    n_bid_s = len(bid_s_obs)
    if n_bid_s:
        bid_strat_obs = np.stack(bid_s_obs)
        bid_strat_masks = np.stack(bid_s_masks)
        bid_strat_strats = np.stack(bid_s_strats)
    else:
        bid_strat_obs = np.empty((0, obs_dim), dtype=np.float32)
        bid_strat_masks = np.empty((0, BID_ACTION_SIZE), dtype=bool)
        bid_strat_strats = np.empty((0, BID_ACTION_SIZE), dtype=np.float32)

    # ── Build play advantage targets ───────────────────────────────────────
    n_play_adv = len(play_t_idx)
    if n_play_adv == 0:
        play_adv_obs = np.empty((0, obs_dim), dtype=np.float32)
        play_adv_masks = np.empty((0, PLAY_ACTION_SIZE), dtype=bool)
        play_adv_targets = np.empty((0, PLAY_ACTION_SIZE), dtype=np.float32)
        play_adv_actions = np.empty((0,), dtype=np.int64)
    else:
        play_adv_obs = np.stack([play_s_obs[i] for i in play_t_idx])
        play_adv_masks = np.stack([play_s_masks[i] for i in play_t_idx])
        play_adv_targets = np.zeros((n_play_adv, PLAY_ACTION_SIZE), dtype=np.float32)
        play_adv_actions = np.array(play_t_act, dtype=np.int64)
        for k in range(n_play_adv):
            adv_est = play_t_adv[k]
            strat = play_s_strats[play_t_idx[k]]
            mask_k = play_s_masks[play_t_idx[k]]
            legal = np.where(mask_k)[0]
            baseline = float(np.dot(strat[legal], adv_est[legal]))
            a = play_t_act[k]
            prob = max(float(strat[a]), 0.05)
            play_adv_targets[k, legal] = adv_est[legal] - baseline
            play_adv_targets[k, a] = (utility - baseline) / prob

    # ── Build play strategy arrays ─────────────────────────────────────────
    n_play_s = len(play_s_obs)
    if n_play_s:
        play_strat_obs = np.stack(play_s_obs)
        play_strat_masks = np.stack(play_s_masks)
        play_strat_strats = np.stack(play_s_strats)
    else:
        play_strat_obs = np.empty((0, obs_dim), dtype=np.float32)
        play_strat_masks = np.empty((0, PLAY_ACTION_SIZE), dtype=bool)
        play_strat_strats = np.empty((0, PLAY_ACTION_SIZE), dtype=np.float32)

    return (
        bid_adv_obs, bid_adv_masks, bid_adv_targets, bid_adv_actions,
        bid_strat_obs, bid_strat_masks, bid_strat_strats,
        play_adv_obs, play_adv_masks, play_adv_targets, play_adv_actions,
        play_strat_obs, play_strat_masks, play_strat_strats,
    )
