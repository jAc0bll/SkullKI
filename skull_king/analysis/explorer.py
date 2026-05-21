"""GTO strategy explorer for trained Deep CFR models."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from skull_king.cards import Card, CardType, Suit, TigressMode, TRUMP_SUIT
from skull_king.env.skull_king_env import (
    ACTION_SPACE_SIZE, N_BID_ACTIONS, N_PLAY_SLOTS, OBS_SIZE,
    TIGRESS_AS_ESCAPE_ACTION, TIGRESS_AS_PIRATE_ACTION,
    _CANONICAL_DECK,
)
from skull_king.cards import build_deck
from skull_king.training.cfr.networks import StrategyNet


# Maps card → slot index in canonical deck (only the 69 play slots, excluding Tigress at 69)
_CARD_TO_SLOT: dict[Card, int] = {
    card: i for i, card in enumerate(_CANONICAL_DECK) if i < N_PLAY_SLOTS
}
MAX_PLAYERS = 6
N_ROUNDS = 10


@dataclass
class BidResult:
    """Bid recommendation for a hand."""
    round_num: int
    hand: list[Card]
    probabilities: dict[int, float]   # bid → probability
    recommended_bid: int
    hand_strength: float              # heuristic hand strength score

    def summary(self) -> str:
        lines = [f"Round {self.round_num} | Hand strength: {self.hand_strength:.2f}"]
        lines.append(f"Recommended bid: {self.recommended_bid}")
        for bid, prob in sorted(self.probabilities.items()):
            bar = "█" * int(prob * 20)
            lines.append(f"  Bid {bid}: {prob:5.1%}  {bar}")
        return "\n".join(lines)


@dataclass
class PlayResult:
    """Card play recommendation for a situation."""
    probabilities: dict[str, float]   # card description → probability
    recommended: str
    position_in_trick: int            # 0=lead, 1=2nd, 2=3rd, 3=4th

    def summary(self) -> str:
        lines = [f"Position in trick: {self.position_in_trick + 1}st/nd/rd/th to play"]
        lines.append(f"Recommended: {self.recommended}")
        for card, prob in sorted(self.probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            lines.append(f"  {card:<25} {prob:5.1%}  {bar}")
        return "\n".join(lines)


def _encode_cards(cards: list[Card]) -> np.ndarray:
    """Encode a list of cards as a 70-dim binary vector."""
    vec = np.zeros(70, dtype=np.float32)
    for card in cards:
        if card in _CARD_TO_SLOT:
            vec[_CARD_TO_SLOT[card]] = 1.0
        elif card.card_type == CardType.TIGRESS:
            # Tigress is at canonical index 69
            vec[69] = 1.0
    return vec


def _card_description(card: Card, tigress_mode: Optional[TigressMode] = None) -> str:
    if card.card_type == CardType.TIGRESS:
        mode = tigress_mode.value if tigress_mode else "?"
        return f"Tigress ({mode})"
    if card.card_type == CardType.NUMBERED:
        return f"{card.suit.value.capitalize()} {card.value}"  # type: ignore
    return card.card_type.value.replace("_", " ").title()


def hand_strength(hand: list[Card]) -> float:
    """Heuristic hand strength (0-1 scale) matching HeuristicAgent bidding."""
    weights = {
        CardType.SKULL_KING: 0.95,
        CardType.PIRATE: 0.75,
        CardType.TIGRESS: 0.45,
        CardType.MERMAID: 0.25,
        CardType.ESCAPE: 0.0,
    }
    total = 0.0
    for card in hand:
        if card.card_type == CardType.NUMBERED:
            assert card.value is not None
            if card.suit == TRUMP_SUIT:
                total += 0.15 + (card.value / 14) * 0.60
            else:
                total += (card.value / 14) * 0.15
        else:
            total += weights.get(card.card_type, 0.0)
    return total


def hand_features(hand: list[Card]) -> dict:
    """Extract categorical features from a hand for table grouping."""
    specials = sum(1 for c in hand if c.card_type in (
        CardType.SKULL_KING, CardType.PIRATE, CardType.MERMAID, CardType.TIGRESS))
    escapes = sum(1 for c in hand if c.card_type == CardType.ESCAPE)
    blacks = sum(1 for c in hand if c.card_type == CardType.NUMBERED and c.suit == TRUMP_SUIT)
    high = sum(1 for c in hand if c.card_type == CardType.NUMBERED and (c.value or 0) >= 10)
    has_sk = any(c.card_type == CardType.SKULL_KING for c in hand)
    pirates = sum(1 for c in hand if c.card_type == CardType.PIRATE)
    strength = hand_strength(hand)
    return {
        "specials": specials,
        "escapes": escapes,
        "blacks": blacks,
        "high_cards": high,
        "has_skull_king": has_sk,
        "pirates": pirates,
        "strength": round(strength, 2),
        "strength_bucket": "low" if strength < 1 else "medium" if strength < 2.5 else "high",
    }


class StrategyExplorer:
    """Query a trained Deep CFR strategy network for GTO recommendations."""

    def __init__(self, strat_net_path: str, n_players: int = 4) -> None:
        hidden = self._infer_hidden(strat_net_path)
        self.net = StrategyNet(hidden=hidden)
        state_dict = torch.load(strat_net_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.n_players = n_players

    @staticmethod
    def _infer_hidden(path: str) -> tuple[int, ...]:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        hidden = []
        i = 0
        while True:
            key, nxt = f"net.{i*2}.weight", f"net.{(i+1)*2}.weight"
            if key not in sd:
                break
            if nxt in sd:
                hidden.append(sd[key].shape[0])
            i += 1
        return tuple(hidden)

    # ------------------------------------------------------------------
    # Bid query
    # ------------------------------------------------------------------

    def query_bid(
        self,
        hand: list[Card],
        round_num: int,
        player_score: float = 0.0,
        opponent_scores: Optional[list[float]] = None,
    ) -> BidResult:
        """Return bid probability distribution for a hand in a round."""
        obs = self._make_bid_obs(hand, round_num, player_score, opponent_scores)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for b in range(round_num + 1):
            mask[b] = True
        probs = self.net.predict_probs(obs, mask)
        bid_probs = {b: float(probs[b]) for b in range(round_num + 1)}
        recommended = max(bid_probs, key=bid_probs.get)  # type: ignore
        return BidResult(
            round_num=round_num,
            hand=hand,
            probabilities=bid_probs,
            recommended_bid=recommended,
            hand_strength=hand_strength(hand),
        )

    # ------------------------------------------------------------------
    # Play query
    # ------------------------------------------------------------------

    def query_play(
        self,
        hand: list[Card],
        round_num: int,
        trick_num: int,
        my_bid: int,
        tricks_won: int,
        current_trick_cards: list[Card],
        seen_cards: list[Card],
        player_score: float = 0.0,
        opponent_scores: Optional[list[float]] = None,
        is_trick_leader: bool = False,
    ) -> PlayResult:
        """Return card play probability distribution for a situation."""
        obs = self._make_play_obs(
            hand, round_num, trick_num, my_bid, tricks_won,
            current_trick_cards, seen_cards, player_score,
            opponent_scores or [], is_trick_leader,
        )
        mask = self._make_play_mask(hand, current_trick_cards)
        probs = self.net.predict_probs(obs, mask)

        # Map action indices back to card descriptions
        result_probs: dict[str, float] = {}
        tigress_in_hand = any(c.card_type == CardType.TIGRESS for c in hand)

        # Iterate only over the 69 play slots (indices 0..68, excluding Tigress at 69)
        for i in range(N_PLAY_SLOTS):
            card = _CANONICAL_DECK[i]
            action = N_BID_ACTIONS + i
            if mask[action] and probs[action] > 0.001:
                result_probs[_card_description(card)] = float(probs[action])

        if tigress_in_hand:
            if mask[TIGRESS_AS_ESCAPE_ACTION] and probs[TIGRESS_AS_ESCAPE_ACTION] > 0.001:
                result_probs["Tigress (ESCAPE)"] = float(probs[TIGRESS_AS_ESCAPE_ACTION])
            if mask[TIGRESS_AS_PIRATE_ACTION] and probs[TIGRESS_AS_PIRATE_ACTION] > 0.001:
                result_probs["Tigress (PIRATE)"] = float(probs[TIGRESS_AS_PIRATE_ACTION])

        recommended = max(result_probs, key=result_probs.get) if result_probs else "?"  # type: ignore
        position = len(current_trick_cards)
        return PlayResult(
            probabilities=result_probs,
            recommended=recommended,
            position_in_trick=position,
        )

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _make_bid_obs(
        self,
        hand: list[Card],
        round_num: int,
        player_score: float,
        opponent_scores: Optional[list[float]],
    ) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0:70] = _encode_cards(hand)
        # Bids unknown, tricks=0, player 0 is self (relative indexing)
        for i in range(MAX_PLAYERS):
            obs[210 + i] = -1.0  # bid unknown
        obs[222] = float(np.clip(player_score / 300.0, -1.0, 1.0))
        if opponent_scores:
            for i, s in enumerate(opponent_scores[:self.n_players - 1], 1):
                obs[222 + i] = float(np.clip(s / 300.0, -1.0, 1.0))
        obs[234] = 1.0  # player 0 (self) is leader for bidding purposes
        obs[240] = (round_num - 1) / (N_ROUNDS - 1)
        obs[241] = 0.0  # first trick of round
        obs[242] = 1.0  # bidding phase
        obs[243] = 0.0  # no cards in trick
        return obs

    def _make_play_obs(
        self,
        hand: list[Card],
        round_num: int,
        trick_num: int,
        my_bid: int,
        tricks_won: int,
        current_trick: list[Card],
        seen_cards: list[Card],
        player_score: float,
        opponent_scores: list[float],
        is_trick_leader: bool,
    ) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0:70] = _encode_cards(hand)
        obs[70:140] = _encode_cards(current_trick)
        obs[140:210] = _encode_cards(seen_cards)
        # Player 0 = self (relative to observer)
        obs[210] = my_bid / round_num if round_num > 0 else 0.0
        obs[216] = tricks_won / round_num if round_num > 0 else 0.0
        obs[222] = float(np.clip(player_score / 300.0, -1.0, 1.0))
        obs[228] = 1.0  # own bid always revealed when playing
        obs[234] = 1.0 if is_trick_leader else 0.0
        for i in range(MAX_PLAYERS):
            if i > 0:
                obs[210 + i] = -1.0  # opponents' bids unknown in simple query
        if opponent_scores:
            for i, s in enumerate(opponent_scores[:self.n_players - 1], 1):
                obs[222 + i] = float(np.clip(s / 300.0, -1.0, 1.0))
        obs[240] = (round_num - 1) / (N_ROUNDS - 1)
        obs[241] = (trick_num - 1) / max(round_num - 1, 1)
        obs[242] = 0.0  # playing phase
        obs[243] = len(current_trick) / self.n_players
        return obs

    def _make_play_mask(
        self, hand: list[Card], current_trick: list[Card]
    ) -> np.ndarray:
        """Compute legal play mask using follow-suit rules.

        Rules:
        - Special cards (Escape, Pirate, Mermaid, Skull King, Tigress) can always be played.
        - If a non-trump (colored) suit was led, you MUST follow that suit if you have it.
        - Black (trump) cards CANNOT be played if a colored suit was led and you hold that suit.
          (Black is strict-follow: you must follow the led color before playing trump.)
        - If trump (Black) was led, you must follow trump if you have it.
        - If you have no cards of the led suit (including trump if trump was led), you may play anything.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        # Determine the led suit (first numbered card in the trick)
        led_suit: Optional[Suit] = None
        for trick_card in current_trick:
            if trick_card.card_type == CardType.NUMBERED:
                led_suit = trick_card.suit
                break

        # Check what suits the player holds (among numbered cards)
        def has_suit(s: Suit) -> bool:
            return any(
                c.card_type == CardType.NUMBERED and c.suit == s
                for c in hand
            )

        must_follow: Optional[Suit] = None
        if led_suit is not None:
            if has_suit(led_suit):
                must_follow = led_suit

        for card in hand:
            if card.card_type == CardType.TIGRESS:
                mask[TIGRESS_AS_ESCAPE_ACTION] = True
                mask[TIGRESS_AS_PIRATE_ACTION] = True
                continue

            if card.card_type != CardType.NUMBERED:
                # All other specials (Escape, Pirate, Mermaid, Skull King) are always legal
                slot = _CARD_TO_SLOT.get(card)
                if slot is not None:
                    mask[N_BID_ACTIONS + slot] = True
                continue

            # Numbered card: apply follow-suit rules
            legal = True
            if must_follow is not None:
                # Must play the led suit; trump cannot substitute for a colored led suit
                if card.suit != must_follow:
                    legal = False

            if legal:
                slot = _CARD_TO_SLOT.get(card)
                if slot is not None:
                    mask[N_BID_ACTIONS + slot] = True

        return mask
