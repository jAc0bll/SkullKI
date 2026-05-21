"""Gymnasium-compatible single-agent environment for Skull King.

The RL agent controls one player (default: player 0). All other players
are auto-played by a random policy on each step.

Reward modes
------------
``"sparse"``
    Single reward at game end: ``total_score / 200``.
    Unbiased — the agent discovers strategy from scratch.
    Credit assignment spans 10 rounds; slow to learn.

``"round"``  (default)
    Score delta at each round boundary: ``Δscore / 200``.
    Captures bid accuracy naturally (the round score already
    encodes whether the bid was hit or missed).
    Zero within a round; 10 non-zero signals per game.

``"shaped"``
    Round reward + small bid-conditioned trick hints each time
    the controlled player's trick count changes intra-round.

    Skull King has *inverted* incentives: winning a trick is bad
    for bid=0, and going one trick over the bid fails it entirely.
    Plain trick-win rewards (always positive) teach the wrong policy.

    Signals (small — round reward dominates at ≈±0.5):
      bid=0, trick won        → −0.10  (ruined bid-0 round)
      bid>0, tricks ≤ bid     → +0.03  (on track toward bid)
      bid>0, tricks > bid     → −0.07  (overshot bid)

    Faster early learning, but the agent needs reasonable bids first.
    Potential reward-hacking risk if bid quality is very poor.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from skull_king.cards import (
    Card, CardType, TigressMode, build_deck,
    MAX_PLAYERS, NUM_ROUNDS, DECK_TOTAL,
)
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase, GameState
from skull_king.resolver import TrickResolver

REWARD_MODES = frozenset({"sparse", "round", "shaped"})

# Shaped-mode intra-round hint magnitudes (small — round signal dominates).
_HINT_BID0_TRICK_WON = -0.10   # bid=0 trick win: catastrophic for the round
_HINT_ON_TRACK       = +0.03   # bid>0, tricks_won ≤ bid: making progress
_HINT_OVERSHOT       = -0.07   # bid>0, tricks_won > bid: already failed


def _bid_conditioned_signal(bid: int, prev_tricks: int, delta_tricks: int) -> float:
    """Small intra-round reward aligned with Skull King's inverted incentives.

    For bid=0 any trick win is catastrophic, so we penalise it.
    For bid>0 we reward progress toward the bid and penalise overshoot.
    The round-end score delta (~±0.5) always dominates these hints (~±0.10 max).
    """
    if bid == 0:
        return _HINT_BID0_TRICK_WON * delta_tricks

    signal = 0.0
    for i in range(delta_tricks):
        trick_num = prev_tricks + i + 1
        signal += _HINT_ON_TRACK if trick_num <= bid else _HINT_OVERSHOT
    return signal


# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------

N_BID_ACTIONS = 11           # actions 0..10 → bid that amount
N_PLAY_SLOTS = 69            # canonical deck slots 0..68 (all cards except Tigress)
TIGRESS_AS_ESCAPE_ACTION = N_BID_ACTIONS + N_PLAY_SLOTS       # = 80
TIGRESS_AS_PIRATE_ACTION = N_BID_ACTIONS + N_PLAY_SLOTS + 1   # = 81
ACTION_SPACE_SIZE = N_BID_ACTIONS + N_PLAY_SLOTS + 2           # = 82

# ---------------------------------------------------------------------------
# Observation space constants
# ---------------------------------------------------------------------------

OBS_SIZE = 244   # 3×70 + 5×6 + 4

# ---------------------------------------------------------------------------
# Canonical deck (built once at module load, deterministic order)
# Indices 0-55:  numbered cards (BLACK 1-14, YELLOW 1-14, GREEN 1-14, PURPLE 1-14)
# Indices 56-60: Escape ×5
# Indices 61-65: Pirate ×5
# Indices 66-67: Mermaid ×2
# Index 68:      Skull King
# Index 69:      Tigress  (play slots only cover 0-68; Tigress gets 2 dedicated actions)
# ---------------------------------------------------------------------------

_CANONICAL_DECK: list[Card] = build_deck()

# Map each unique Card → list of canonical indices it occupies
_CARD_TO_INDICES: dict[Card, list[int]] = {}
for _i, _c in enumerate(_CANONICAL_DECK):
    _CARD_TO_INDICES.setdefault(_c, []).append(_i)

# Flat tuple lookup indexed by Card._hash (the cached int hash value, 0..60).
# Each entry is a tuple of canonical slot indices for that equality class —
# e.g. Pirate (hash=57) maps to (61, 62, 63, 64, 65).  Looking up slots by
# integer hash is faster than ``dict[Card, list]`` lookup (no Card.__hash__
# call, no bucket walk for collisions).
_HASH_TO_SLOTS: tuple[tuple[int, ...], ...]
_max_hash = max(c._hash for c in _CARD_TO_INDICES) + 1
_temp = [()] * _max_hash
for _c, _slots in _CARD_TO_INDICES.items():
    _temp[_c._hash] = tuple(_slots)
_HASH_TO_SLOTS = tuple(_temp)
del _temp, _max_hash


def _encode_cards_into(cards, out: np.ndarray) -> None:
    """Zero ``out`` (must be length 70) and write 1.0 at each card's canonical slot.

    Hot path called ~150k times per CFR iteration.  Optimisations vs naive:
      * Single pass through ``cards`` (was: count → iterate counts twice).
      * Integer-keyed lookups against the precomputed ``_HASH_TO_SLOTS`` tuple
        instead of ``dict[Card, list]`` — saves a Card.__hash__ call and the
        dict bucket walk for every card.
      * ``bytearray(61)`` counter (zero-init in O(1)) instead of ``dict``.
    """
    out[:] = 0.0
    counters = bytearray(len(_HASH_TO_SLOTS))  # zeroed per call, O(1) alloc
    hash_to_slots = _HASH_TO_SLOTS              # local binding — faster lookup
    for card in cards:
        h = card._hash
        slots = hash_to_slots[h]
        cnt = counters[h]
        if cnt < len(slots):
            out[slots[cnt]] = 1.0
            counters[h] = cnt + 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SkullKingEnv(gym.Env):
    """Single-agent Skull King environment.

    Parameters
    ----------
    n_players:
        Total players (2–6). The RL agent controls *controlled_player*; the
        rest are auto-played randomly.
    controlled_player:
        Which seat the RL agent occupies (default 0).
    reward_mode:
        One of ``"sparse"``, ``"round"`` (default), ``"shaped"``.
        See module docstring for detailed tradeoff analysis.
    seed:
        Master seed for the game RNG and the auto-play random policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_players: int = 3,
        controlled_player: int = 0,
        reward_mode: str = "round",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not (2 <= n_players <= MAX_PLAYERS):
            raise ValueError(f"n_players must be 2–{MAX_PLAYERS}, got {n_players}")
        if not (0 <= controlled_player < n_players):
            raise ValueError(f"controlled_player must be 0–{n_players - 1}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {sorted(REWARD_MODES)}, got {reward_mode!r}"
            )

        self.n_players = n_players
        self._controlled_player = controlled_player
        self._reward_mode = reward_mode
        self._master_seed = seed

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Set during reset()
        self._engine: GameEngine
        self._current_state: GameState
        self._rng: random.Random
        self._prev_score: int = 0
        self._prev_tricks: int = 0
        self._episode_seed: int = seed

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed

        self._rng = random.Random(self._episode_seed)
        engine_seed = self._rng.randint(0, 2**31 - 1)
        self._engine = GameEngine(n_players=self.n_players, seed=engine_seed)
        self._current_state = self._engine.start()
        self._prev_score = 0
        self._prev_tricks = 0

        self._current_state = self._advance_others(self._current_state)
        obs = self._build_observation(self._current_state)
        return obs, self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        state = self._current_state
        cp = self._controlled_player

        if state.phase == GamePhase.BIDDING:
            assert 0 <= action < N_BID_ACTIONS, f"Invalid bid action {action}"
            bid = int(action)
            state = self._engine.place_bid(cp, bid)
        elif state.phase == GamePhase.PLAYING:
            card, tigress_mode = self._decode_play_action(action)
            state = self._engine.play_card(cp, card, tigress_mode)
        else:
            raise RuntimeError("step() called on a GAME_OVER environment")

        state = self._advance_others(state)
        self._current_state = state

        reward = self._compute_reward(state)
        terminated = state.phase == GamePhase.GAME_OVER
        obs = self._build_observation(state)
        return obs, reward, terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        """Boolean mask: True where an action is currently legal."""
        return self._action_masks_for(self._current_state, self._controlled_player)

    def _action_masks_for(self, state: GameState, player_index: int) -> np.ndarray:
        """Parameterised mask builder — used for self-play opponents."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        cp = player_index

        if state.phase == GamePhase.BIDDING:
            for b in range(state.round_number + 1):
                mask[b] = True

        elif state.phase == GamePhase.PLAYING and state.current_player_index == cp:
            hand = list(state.player_states[cp].hand)
            legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)

            counts: dict[Card, int] = {}
            for card in legal:
                counts[card] = counts.get(card, 0) + 1

            for card, count in counts.items():
                if card.card_type == CardType.TIGRESS:
                    mask[TIGRESS_AS_ESCAPE_ACTION] = True
                    mask[TIGRESS_AS_PIRATE_ACTION] = True
                else:
                    slots = _CARD_TO_INDICES[card]
                    for i in range(min(count, len(slots))):
                        mask[N_BID_ACTIONS + slots[i]] = True

        return mask

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance_others(self, state: GameState) -> GameState:
        """Auto-play non-controlled players until it's the controlled player's
        turn to act, or the game ends."""
        while state.phase not in (GamePhase.GAME_OVER,):
            if state.phase == GamePhase.BIDDING:
                cur = state.current_player_index
                if cur == self._controlled_player:
                    break
                bid = self._rng.randint(0, state.round_number)
                state = self._engine.place_bid(cur, bid)

            elif state.phase == GamePhase.PLAYING:
                cur = state.current_player_index
                if cur == self._controlled_player:
                    break
                hand = list(state.player_states[cur].hand)
                legal = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
                card = self._rng.choice(legal)
                mode: Optional[TigressMode] = None
                if card.card_type == CardType.TIGRESS:
                    mode = self._rng.choice([TigressMode.PIRATE, TigressMode.ESCAPE])
                state = self._engine.play_card(cur, card, mode)

        return state

    def _decode_play_action(self, action: int) -> tuple[Card, Optional[TigressMode]]:
        if action == TIGRESS_AS_ESCAPE_ACTION:
            return _CANONICAL_DECK[69], TigressMode.ESCAPE
        if action == TIGRESS_AS_PIRATE_ACTION:
            return _CANONICAL_DECK[69], TigressMode.PIRATE
        slot = action - N_BID_ACTIONS
        return _CANONICAL_DECK[slot], None

    def _compute_reward(self, state: GameState) -> float:
        cp = self._controlled_player
        current_score = state.player_states[cp].total_score
        current_tricks = state.player_states[cp].tricks_won_this_round
        reward = 0.0

        # ── sparse ──────────────────────────────────────────────────────────
        if self._reward_mode == "sparse":
            if state.phase == GamePhase.GAME_OVER:
                reward = current_score / 200.0
                self._prev_score = current_score

        # ── round ───────────────────────────────────────────────────────────
        elif self._reward_mode == "round":
            if state.phase in (GamePhase.GAME_OVER, GamePhase.BIDDING):
                delta = current_score - self._prev_score
                if delta != 0:
                    reward = delta / 200.0
                    self._prev_score = current_score

        # ── shaped ──────────────────────────────────────────────────────────
        # Round-end component (identical to "round").
        # Intra-round: small bid-conditioned signals so trick hints never
        # contradict the terminal outcome (bid=0 penalises winning tricks).
        elif self._reward_mode == "shaped":
            if state.phase in (GamePhase.GAME_OVER, GamePhase.BIDDING):
                delta = current_score - self._prev_score
                if delta != 0:
                    reward = delta / 200.0
                    self._prev_score = current_score

            # Per-trick hint — only during active play (never at boundaries).
            if state.phase == GamePhase.PLAYING:
                bid = state.player_states[cp].bid
                if bid is not None and current_tricks != self._prev_tricks:
                    reward += _bid_conditioned_signal(
                        bid, self._prev_tricks, current_tricks - self._prev_tricks
                    )

        # ── rank bonus (all modes) ───────────────────────────────────────────
        # Skull King is a competitive game — the goal is to RANK first, not to
        # maximise absolute score.  Without this bonus, two runs scoring 200 and
        # 190 look the same to the agent even though one won and one lost.
        # The bonus is large enough to be the dominant end-of-game signal (~±0.5)
        # while staying on the same scale as the accumulated round rewards (~±1.0).
        if state.phase == GamePhase.GAME_OVER:
            scores = [ps.total_score for ps in state.player_states]
            my_score = scores[cp]
            # Rank among unique scores (ties share the same rank).
            sorted_unique = sorted(set(scores), reverse=True)
            rank = sorted_unique.index(my_score)  # 0 = best
            _RANK_BONUS = (0.50, 0.15, -0.10, -0.30, -0.40, -0.45)
            reward += _RANK_BONUS[min(rank, len(_RANK_BONUS) - 1)]

        # Keep tracker current; reset at round boundaries.
        if state.phase == GamePhase.BIDDING:
            self._prev_tricks = 0
        elif state.phase == GamePhase.PLAYING:
            self._prev_tricks = current_tricks

        return float(reward)

    @staticmethod
    def _encode_cards(cards) -> np.ndarray:
        vec = np.zeros(DECK_TOTAL, dtype=np.float32)
        _encode_cards_into(cards, vec)
        return vec

    def _build_observation(self, state: GameState) -> np.ndarray:
        return self._build_observation_for(
            state,
            self._controlled_player,
            self._engine.completed_tricks_this_round,
        )

    # ------------------------------------------------------------------
    # Fast paths for CFR worker — bypass GameState freezing
    # ------------------------------------------------------------------
    #
    # The CFR traversal calls _build_observation_for and _action_masks_for
    # ~50× per game.  Going through engine.get_state() creates a fresh
    # FrozenPlayerState per player per call — ~10% of total CFR time was
    # spent in that freeze.  The methods below read engine internals
    # directly (single-thread CFR worker, no shared state, safe).

    def _build_observation_from_engine(
        self, engine: GameEngine, player_index: int
    ) -> np.ndarray:
        # Local bindings of engine internals — each one saves a Python
        # LOAD_ATTR instruction per access.  The per-player loop hits
        # engine._players up to 6 times, players, rounds & trick_leader once
        # each per iteration of that loop.
        players = engine._players
        rn = engine._round
        completed = engine._completed_tricks
        current_trick = engine._current_trick
        trick_leader = engine._trick_leader
        phase = engine._phase
        trick_in_round = engine._trick_in_round

        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        cp = player_index
        ps = players[cp]

        _encode_cards_into(ps.hand, obs[0:70])
        trick_cards = current_trick.played_cards
        _encode_cards_into([pc.card for pc in trick_cards], obs[70:140])

        # Pre-size the seen list — avoids amortised list-grow allocations
        # (4 trick.played_cards × up to 9 completed tricks per round).
        seen: list[Card] = []
        seen_append = seen.append
        for trick in completed:
            for pc in trick.played_cards:
                seen_append(pc.card)
        _encode_cards_into(seen, obs[140:210])

        n = self.n_players
        rn_f = float(rn)
        # Flatten the MAX_PLAYERS loop: the i ≥ n branch sets only one slot to
        # -1, so handle it in two phases instead of a conditional per iter.
        for i in range(n):
            actual = (cp + i) % n
            aps = players[actual]
            bid = aps.bid
            if bid is None:
                obs[210 + i] = -1.0
            else:
                obs[210 + i] = bid / rn_f
                obs[228 + i] = 1.0
            obs[216 + i] = aps.tricks_won_this_round / rn_f
            # Manual clip is faster than np.clip for a single scalar.
            s = aps.total_score / 300.0
            if s > 1.0:
                s = 1.0
            elif s < -1.0:
                s = -1.0
            obs[222 + i] = s
            if actual == trick_leader:
                obs[234 + i] = 1.0
        # Inactive seats: bid slot stays at -1.0, all others remain at 0.0.
        for i in range(n, MAX_PLAYERS):
            obs[210 + i] = -1.0

        obs[240] = (rn - 1) / (NUM_ROUNDS - 1)
        obs[241] = (trick_in_round - 1) / max(rn - 1, 1)
        obs[242] = 1.0 if phase == GamePhase.BIDDING else 0.0
        obs[243] = len(trick_cards) / n
        return obs

    def _action_masks_from_engine(
        self, engine: GameEngine, player_index: int
    ) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        cp = player_index
        phase = engine._phase

        if phase == GamePhase.BIDDING:
            mask[: engine._round + 1] = True
        elif phase == GamePhase.PLAYING and engine._current_player_index() == cp:
            hand = engine._players[cp].hand
            legal = TrickResolver.legal_plays(
                list(engine._current_trick.played_cards), list(hand)
            )

            counts: dict[Card, int] = {}
            for card in legal:
                counts[card] = counts.get(card, 0) + 1

            for card, count in counts.items():
                if card.card_type == CardType.TIGRESS:
                    mask[TIGRESS_AS_ESCAPE_ACTION] = True
                    mask[TIGRESS_AS_PIRATE_ACTION] = True
                else:
                    slots = _CARD_TO_INDICES[card]
                    for i in range(min(count, len(slots))):
                        mask[N_BID_ACTIONS + slots[i]] = True

        return mask

    def _build_observation_for(
        self,
        state: GameState,
        player_index: int,
        completed_tricks: list,
    ) -> np.ndarray:
        """Parameterised obs builder — used for self-play opponents."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        cp = player_index
        ps = state.player_states[cp]
        rn = state.round_number

        obs[0:70] = self._encode_cards(list(ps.hand))
        obs[70:140] = self._encode_cards([pc.card for pc in state.current_trick_cards])

        seen: list[Card] = []
        for trick in completed_tricks:
            for pc in trick.played_cards:
                seen.append(pc.card)
        obs[140:210] = self._encode_cards(seen)

        n = self.n_players
        for i in range(MAX_PLAYERS):
            actual = (cp + i) % n if i < n else -1
            bid_slot, tricks_slot = 210 + i, 216 + i
            score_slot, revealed_slot, leader_slot = 222 + i, 228 + i, 234 + i

            if actual == -1:
                obs[bid_slot] = -1.0
            else:
                aps = state.player_states[actual]
                if aps.bid is not None:
                    obs[bid_slot] = aps.bid / rn
                    obs[revealed_slot] = 1.0
                else:
                    obs[bid_slot] = -1.0
                obs[tricks_slot] = aps.tricks_won_this_round / rn
                obs[score_slot] = float(np.clip(aps.total_score / 300.0, -1.0, 1.0))
                obs[leader_slot] = 1.0 if actual == state.trick_leader_index else 0.0

        obs[240] = (rn - 1) / (NUM_ROUNDS - 1)
        obs[241] = (state.trick_number - 1) / max(rn - 1, 1)
        obs[242] = 1.0 if state.phase == GamePhase.BIDDING else 0.0
        obs[243] = len(state.current_trick_cards) / n
        return obs

    def _info(self) -> dict[str, Any]:
        state = self._current_state
        cp = self._controlled_player
        ps = state.player_states[cp]
        return {
            "round": state.round_number,
            "trick": state.trick_number,
            "phase": state.phase.value,
            "score": ps.total_score,
            "bid": ps.bid,
            "tricks_won": ps.tricks_won_this_round,
        }
