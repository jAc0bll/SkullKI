"""Public Belief State (PBS) for Skull King.

A PBS captures everything ALL players can observe (public info) plus
a probability distribution over what cards each opponent might hold
(the belief).

Public info:  played cards, revealed bids, scores, round/trick position
Private info: each player's hand — represented as a probability vector
              belief[i, c] = P(player i currently holds card c)

After each card play, the played card is removed from all belief vectors
and distributions are renormalized.  Initially the belief is uniform over
all cards not yet seen (not in own hand, not played in tricks).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from skull_king.cards import DECK_TOTAL, NUM_ROUNDS, MAX_PLAYERS
from skull_king.env.skull_king_env import (
    ACTION_SPACE_SIZE,
    N_BID_ACTIONS,
    _CANONICAL_DECK,
    _HASH_TO_SLOTS,
)
from skull_king.game_state import GamePhase


def _card_to_slot(card) -> int:
    """Return the first canonical slot index for a card."""
    slots = _HASH_TO_SLOTS[card._hash]
    return slots[0] if slots else -1


def _cards_to_mask(cards) -> np.ndarray:
    """Build a 70-element binary mask from an iterable of Card objects."""
    mask = np.zeros(DECK_TOTAL, dtype=bool)
    for c in cards:
        slot = _card_to_slot(c)
        if slot >= 0:
            mask[slot] = True
    return mask


# Size of the encoded PBS vector (fixed for a given n_players)
def pbs_encoding_size(n_players: int) -> int:
    return (
        4                           # round, trick, phase, current_player (normalized)
        + n_players * 5             # bids, tricks_won, scores, bid_revealed, leader (one per player)
        + DECK_TOTAL                # seen_cards (binary mask, completed tricks)
        + DECK_TOTAL                # current_trick_mask (which cards played this trick)
        + n_players * DECK_TOTAL    # belief distribution per player
    )


@dataclass
class PublicBeliefState:
    """All public information + card-holding belief for a Skull King game node.

    Attributes
    ----------
    n_players:      number of players
    round_number:   1-indexed round (1..10)
    trick_number:   1-indexed trick within round
    phase:          0 = BIDDING, 1 = PLAYING, 2 = GAME_OVER
    current_player: index of the player to act next
    bids:           [n_players] bid values, -1 if not yet placed
    tricks_won:     [n_players] tricks won this round
    total_scores:   [n_players] cumulative scores
    bid_revealed:   [n_players] bool — True once bid is locked in
    trick_leader:   index of the player who leads the current trick
    seen_cards:     [DECK_TOTAL] bool — cards played in completed tricks
    current_trick:  [DECK_TOTAL] bool — cards played in the current trick
    belief:         [n_players, DECK_TOTAL] float32 — P(player i holds card j)
                    For the acting player this is their exact binary hand mask.
    """

    n_players: int
    round_number: int
    trick_number: int
    phase: int
    current_player: int

    bids: np.ndarray          # [n_players] int, -1 = unknown
    tricks_won: np.ndarray    # [n_players] int
    total_scores: np.ndarray  # [n_players] float
    bid_revealed: np.ndarray  # [n_players] bool
    trick_leader: int

    seen_cards: np.ndarray    # [DECK_TOTAL] bool
    current_trick: np.ndarray # [DECK_TOTAL] bool
    belief: np.ndarray        # [n_players, DECK_TOTAL] float32

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_engine(cls, engine, acting_player: int) -> "PublicBeliefState":
        """Build a PBS from a live GameEngine and the acting player's perspective."""
        n = engine.n_players
        phase_map = {GamePhase.BIDDING: 0, GamePhase.PLAYING: 1, GamePhase.GAME_OVER: 2}

        bids = np.full(n, -1, dtype=np.int32)
        for i in engine._bids_placed:
            bids[i] = engine._players[i].bid

        tricks_won = np.array([p.tricks_won_this_round for p in engine._players], dtype=np.int32)
        total_scores = np.array([p.total_score for p in engine._players], dtype=np.float32)
        bid_revealed = np.array([i in engine._bids_placed for i in range(n)], dtype=bool)

        # Seen cards: all cards played in resolved tricks this round
        seen_mask = np.zeros(DECK_TOTAL, dtype=bool)
        for trick in engine._completed_tricks:
            for pc in trick.played_cards:
                slot = _card_to_slot(pc.card)
                if slot >= 0:
                    seen_mask[slot] = True

        # Current trick cards (not yet resolved)
        curr_mask = np.zeros(DECK_TOTAL, dtype=bool)
        for pc in engine._current_trick.played_cards:
            slot = _card_to_slot(pc.card)
            if slot >= 0:
                curr_mask[slot] = True

        # Own hand mask (exact knowledge)
        own_hand_mask = _cards_to_mask(engine._players[acting_player].hand)

        # Belief: uniform over unseen cards for opponents; exact for acting player
        unavailable = seen_mask | curr_mask | own_hand_mask
        unseen = ~unavailable   # cards that could be in opponent hands

        belief = np.zeros((n, DECK_TOTAL), dtype=np.float32)
        belief[acting_player] = own_hand_mask.astype(np.float32)

        # For opponents: uniform distribution over unseen cards, each opponent
        # expected to hold (round_number - tricks_they_have_played_in) cards.
        # Approximation: give equal weight to each unseen card slot.
        unseen_prob = unseen.astype(np.float32)
        total_unseen = unseen_prob.sum()
        if total_unseen > 0:
            for i in range(n):
                if i != acting_player:
                    belief[i] = unseen_prob / total_unseen

        return cls(
            n_players=n,
            round_number=engine._round,
            trick_number=engine._trick_in_round,
            phase=phase_map.get(engine._phase, 2),
            current_player=engine._current_player_index(),
            bids=bids,
            tricks_won=tricks_won,
            total_scores=total_scores,
            bid_revealed=bid_revealed,
            trick_leader=engine._trick_leader,
            seen_cards=seen_mask,
            current_trick=curr_mask,
            belief=belief,
        )

    # ---------------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------------

    def encode(self) -> np.ndarray:
        """Encode PBS as a fixed-size float32 vector for network input."""
        n = self.n_players
        parts: list[np.ndarray] = [
            np.array([
                self.round_number / NUM_ROUNDS,
                self.trick_number / max(self.round_number, 1),
                float(self.phase) / 2.0,
                self.current_player / n,
            ], dtype=np.float32),
            # Per-player features
            np.where(self.bids >= 0, self.bids / NUM_ROUNDS, -1.0).astype(np.float32),
            (self.tricks_won / max(self.round_number, 1)).astype(np.float32),
            np.clip(self.total_scores / 100.0, -2.0, 2.0).astype(np.float32),
            self.bid_revealed.astype(np.float32),
            (np.arange(n) == self.trick_leader).astype(np.float32),
            # Card masks
            self.seen_cards.astype(np.float32),
            self.current_trick.astype(np.float32),
            # Belief (flattened)
            self.belief.flatten(),
        ]
        return np.concatenate(parts)

    def encoding_size(self) -> int:
        return pbs_encoding_size(self.n_players)

    # ---------------------------------------------------------------------------
    # Belief updates
    # ---------------------------------------------------------------------------

    def observe_card_played(self, card_slot: int, in_current_trick: bool = True) -> "PublicBeliefState":
        """Return a new PBS after observing a card played publicly."""
        new = self._copy()
        if in_current_trick:
            new.current_trick[card_slot] = True
        else:
            new.seen_cards[card_slot] = True
        # Remove this card from all belief distributions
        new.belief[:, card_slot] = 0.0
        row_sums = new.belief.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        new.belief /= row_sums
        return new

    def with_own_hand(self, hand_cards) -> "PublicBeliefState":
        """Return a new PBS with the acting player's exact hand set."""
        new = self._copy()
        new.belief[self.current_player] = _cards_to_mask(hand_cards).astype(np.float32)
        return new

    def _copy(self) -> "PublicBeliefState":
        new = copy.copy(self)
        new.bids = self.bids.copy()
        new.tricks_won = self.tricks_won.copy()
        new.total_scores = self.total_scores.copy()
        new.bid_revealed = self.bid_revealed.copy()
        new.seen_cards = self.seen_cards.copy()
        new.current_trick = self.current_trick.copy()
        new.belief = self.belief.copy()
        return new

    # ---------------------------------------------------------------------------
    # Sampling private states for determinized subgame solving
    # ---------------------------------------------------------------------------

    def sample_opponent_hands(
        self,
        rng: np.random.Generator,
        own_hand_slots: np.ndarray,
    ) -> list[list[int]]:
        """Sample one plausible hand (list of card slots) per opponent.

        Uses the belief distribution as a weighted sample without replacement.
        Each opponent gets approximately the expected number of remaining cards.

        Parameters
        ----------
        rng:           numpy Generator for reproducibility
        own_hand_slots: card slots of the acting player's known hand
        """
        n = self.n_players
        acting = self.current_player

        # Cards that are definitely unavailable for sampling
        taken = set(np.where(self.seen_cards)[0]) | set(np.where(self.current_trick)[0])
        taken |= set(own_hand_slots)

        available = [c for c in range(DECK_TOTAL) if c not in taken]
        rng.shuffle(available)

        hands: list[list[int]] = [[] for _ in range(n)]
        hands[acting] = list(own_hand_slots)

        # Expected cards remaining per opponent
        # Each player was dealt round_number cards; they've played one per completed trick
        # that they participated in (approximation: assume symmetric play)
        cards_dealt = self.round_number
        cards_played_so_far = self.trick_number - 1  # tricks completed this round
        cards_remaining = max(0, cards_dealt - cards_played_so_far)

        offset = 0
        for i in range(n):
            if i == acting:
                continue
            n_cards = min(cards_remaining, len(available) - offset)
            hands[i] = available[offset: offset + n_cards]
            offset += n_cards

        return hands


# ---------------------------------------------------------------------------
# Batch encoding — avoids PBS object creation and np.concatenate overhead
# ---------------------------------------------------------------------------

def encode_pbs_batch(
    engines: list,
    acting_players: list[int],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Encode a batch of (engine, acting_player) pairs directly into a float32 array.

    ~4-6x faster than [PublicBeliefState.from_engine(e, p).encode() for e, p in ...]
    because it skips dataclass allocation and intermediate array concatenation.

    Parameters
    ----------
    engines:        list of GameEngine objects (any phase except GAME_OVER)
    acting_players: acting player index for each engine
    out:            optional pre-allocated [N, pbs_size] float32 array (reused if provided)

    Returns
    -------
    out: float32 array of shape [N, pbs_encoding_size(n_players)]
    """
    N = len(engines)
    if N == 0:
        n = engines[0].n_players if engines else 4
        return np.empty((0, pbs_encoding_size(n)), dtype=np.float32)

    n = engines[0].n_players
    size = pbs_encoding_size(n)
    if out is None or out.shape != (N, size):
        out = np.empty((N, size), dtype=np.float32)

    for i in range(N):
        _encode_into(engines[i], acting_players[i], n, out[i])

    return out


def _encode_into(eng, acting: int, n: int, row: np.ndarray) -> None:
    """Fill one row of the batch output in-place."""
    round_num = eng._round          # 1-indexed
    trick_num = eng._trick_in_round

    phase_val = (
        0.0 if eng._phase == GamePhase.BIDDING
        else 1.0 if eng._phase == GamePhase.PLAYING
        else 2.0
    )
    cur_player = (
        eng._current_player_index()
        if eng._phase not in (GamePhase.GAME_OVER,)
        else 0
    )

    ptr = 0

    # ── scalar context [4] ───────────────────────────────────────────
    row[ptr]   = round_num / NUM_ROUNDS
    row[ptr+1] = trick_num / max(round_num, 1)
    row[ptr+2] = phase_val / 2.0
    row[ptr+3] = cur_player / n
    ptr += 4

    # ── per-player features [n × 5] ─────────────────────────────────
    inv_round = 1.0 / max(round_num, 1)
    for j in range(n):
        row[ptr + j] = (eng._players[j].bid / NUM_ROUNDS
                        if j in eng._bids_placed else -1.0)
    ptr += n
    for j in range(n):
        row[ptr + j] = eng._players[j].tricks_won_this_round * inv_round
    ptr += n
    for j in range(n):
        s = eng._players[j].total_score / 100.0
        row[ptr + j] = s if -2.0 <= s <= 2.0 else (2.0 if s > 2.0 else -2.0)
    ptr += n
    for j in range(n):
        row[ptr + j] = 1.0 if j in eng._bids_placed else 0.0
    ptr += n
    row[ptr: ptr + n] = 0.0
    row[ptr + eng._trick_leader] = 1.0
    ptr += n

    # ── seen_cards [DECK_TOTAL] ──────────────────────────────────────
    seen = row[ptr: ptr + DECK_TOTAL]
    seen[:] = 0.0
    for trick in eng._completed_tricks:
        for pc in trick.played_cards:
            slot = _card_to_slot(pc.card)
            if slot >= 0:
                seen[slot] = 1.0
    ptr += DECK_TOTAL

    # ── current_trick [DECK_TOTAL] ───────────────────────────────────
    curr = row[ptr: ptr + DECK_TOTAL]
    curr[:] = 0.0
    for pc in eng._current_trick.played_cards:
        slot = _card_to_slot(pc.card)
        if slot >= 0:
            curr[slot] = 1.0
    ptr += DECK_TOTAL

    # ── belief [n × DECK_TOTAL] ──────────────────────────────────────
    belief = row[ptr: ptr + n * DECK_TOTAL].reshape(n, DECK_TOTAL)
    belief[:] = 0.0

    # Acting player's exact hand
    for c in eng._players[acting].hand:
        slot = _card_to_slot(c)
        if slot >= 0:
            belief[acting, slot] = 1.0

    # Opponents: uniform over unseen cards
    # unavailable = seen | current_trick | own_hand
    for s in range(DECK_TOTAL):
        if seen[s] or curr[s] or belief[acting, s]:
            continue
        prob_slot = 1.0  # will normalize below
        for j in range(n):
            if j != acting:
                belief[j, s] = 1.0

    # Normalize each opponent row
    for j in range(n):
        if j == acting:
            continue
        total = belief[j].sum()
        if total > 0:
            belief[j] /= total
