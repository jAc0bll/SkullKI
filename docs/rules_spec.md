# Skull King — Formal Rules Specification
# Reference: Grandpa Beck's Games, 2021 Edition
# Status: AUTHORITATIVE — use this document, not memory, when coding

> **AMBIGUOUS** tags mark rules where official sources conflict or are silent.
> Resolve each before implementing; default resolutions are noted.

---

## 1. Deck Composition (70 cards total)

| Category    | Count | Details                                         |
|-------------|-------|-------------------------------------------------|
| Numbered    | 56    | Values 1–14 in each of 4 suits: Black, Yellow, Green, Purple |
| Escape      | 5     | Identical; suit-less                            |
| Pirate      | 5     | Identical; suit-less                            |
| Mermaid     | 2     | Identical; suit-less                            |
| Skull King  | 1     | Unique; suit-less                               |
| Tigress     | 1     | Unique; suit-less; dual-mode (declared on play) |
| **Total**   | **70**|                                                 |

### Suit names (canonical identifiers for code)
```
BLACK, YELLOW, GREEN, PURPLE
```
Black is the **trump suit**. The other three are "colored suits" with no inherent ranking between them.

---

## 2. Players and Rounds

- **Players:** 2–6 (standard). 7–8 possible if house rules allow a shorter final round.
- **Rounds:** 10 total.
- **Cards dealt per round:** equal to the round number.
  - Round 1 → 1 card each, Round 10 → 10 cards each.
- **Max cards dealt (6 players, Round 10):** 60. Deck has 70; always sufficient.

> **AMBIGUOUS-01:** Player count above 6. Some sources say 8 players, but 8×10=80 > 70 cards.
> **Default resolution:** Cap at 6 players. For 7–8, truncate the final rounds or draw without replacement and reshuffle; not implemented in v1.

---

## 3. Card Hierarchy (Power Ranking)

Listed strongest → weakest. The **first** applicable rule wins.

```
RANK 1 (conditional): Mermaid   — ONLY beats Skull King; loses to everything else
RANK 2:               Skull King — beats all Pirate(s), all numbered cards, Escapes
RANK 3:               Pirate     — beats all numbered cards, Mermaids; loses to Skull King
RANK 3 (alias):       Tigress-as-Pirate — treated identically to Pirate
RANK 4:               Black 14 > Black 13 > … > Black 1 (trump numbered cards)
RANK 5:               Led-suit card (highest value wins among led-suit cards)
RANK 6:               Off-suit colored cards (cannot win; value irrelevant)
RANK 7 (weakest):     Escape / Tigress-as-Escape — never wins (except all-Escape edge case)
```

### Critical interaction matrix

| Cards in trick          | Winner                     | Notes                              |
|-------------------------|----------------------------|------------------------------------|
| SK only                 | Skull King                 |                                    |
| SK + Pirate(s)          | Skull King                 | +30 per Pirate bonus (see §6)      |
| SK + Mermaid            | Mermaid                    | +40 bonus to Mermaid player (§6)   |
| SK + Mermaid + Pirate   | **Skull King**             | Pirate beats Mermaid (negates her SK counter); SK then beats Pirate |
| Multiple Pirates        | **First Pirate played**    | Earliest in play order wins        |
| Multiple Mermaids       | **First Mermaid played**   | AMBIGUOUS-02 (see below)           |
| Pirate + Mermaid (no SK)| Pirate                     |                                    |
| All Escapes             | **First Escape played**    | Leader wins; see §5.4              |

> **AMBIGUOUS-02:** No official ruling on two Mermaids, no Skull King.
> **Default resolution:** First Mermaid played wins (consistent with Pirate tie-break rule).

---

## 4. Trump and Suit-Following Rules

### 4.1 Led suit determination
- The **first numbered card** played in a trick establishes the **led suit**.
- If the first card is a special card (Escape, Pirate, Mermaid, Skull King, Tigress), **no led suit is established**.

### 4.2 Must-follow rule
A player **must** play a card of the led suit if they hold one, with these exceptions:
- **Special cards** (Escape, Pirate, Mermaid, Skull King, Tigress) may always be played regardless of led suit — they carry no suit.
- If a player holds **only** Black (trump) cards and a colored suit was led, they **must** play Black (they have nothing else). This is not "reneging."

**CONFIRMED (2026-05-17):** Black numbered cards follow the same must-follow rule as any other suit. If a colored suit is led and you hold cards of that suit, you must play one — you may not play Black instead. Black may only be played freely when you hold no cards of the led suit.

### 4.3 Trump (Black) superiority
When multiple suits are present in a trick with a colored led suit, the highest Black card beats all other numbered cards, regardless of quantity or value of led-suit cards.

### 4.4 Tigress declaration
- When playing Tigress, the player **immediately and verbally declares** "Pirate" or "Escape" before placing it on the table.
- The declaration is **irrevocable** for that trick.
- Tigress-as-Pirate: identical to a Pirate for all rules including bonuses.
- Tigress-as-Escape: identical to an Escape, including the all-Escape edge case.

---

## 5. Trick Resolution Algorithm

### Step-by-step (implement in this exact order)

```
function resolve_trick(cards: List[PlayedCard]) -> Player:
    # PlayedCard = {player, card, play_order (1-based)}

    special_cards = [c for c in cards if c.card.is_special]
    has_sk      = any(c.card.type == SKULL_KING for c in special_cards)
    has_mermaid = any(c.card.type == MERMAID    for c in special_cards)
    has_pirate  = any(c.card.type in (PIRATE, TIGRESS_PIRATE) for c in special_cards)

    # STEP 1: Skull King + Mermaid present → Mermaid wins
    if has_sk and has_mermaid and not has_pirate:
        winner = first_played(MERMAID, cards)
        return winner

    # STEP 2: Skull King present (and no Mermaid, or Pirate also present)
    if has_sk and (not has_mermaid or has_pirate):
        winner = first_played(SKULL_KING, cards)
        return winner

    # STEP 3: Pirate(s) present, no Skull King
    if has_pirate:
        winner = first_played([PIRATE, TIGRESS_PIRATE], cards)
        return winner

    # STEP 4: Mermaid(s) present, no Skull King, no Pirate
    if has_mermaid:
        winner = first_played(MERMAID, cards)
        return winner

    # STEP 5: No special cards (or only Escapes)
    led_suit = determine_led_suit(cards)  # None if Escape/special led

    if led_suit is None:
        # Sub-case: all Escapes → leader wins (play_order == 1)
        non_escapes = [c for c in cards if c.card.type != ESCAPE and c.card.type != TIGRESS_ESCAPE]
        if not non_escapes:
            return first_played(ESCAPE, cards)  # first player = trick leader

        # Sub-case: Escape led, then numbered cards played
        # Treat highest Black as winner; if no Black, highest of the first non-escape suit played
        black_cards = [c for c in non_escapes if c.card.suit == BLACK]
        if black_cards:
            return max(black_cards, key=lambda c: c.card.value)
        # No Black: first non-Escape establishes an informal led suit
        first_colored = min(non_escapes, key=lambda c: c.play_order)
        led_suit = first_colored.card.suit
        # fall through to Step 5b

    # Step 5a: Black (trump) cards present, colored suit led
    black_cards = [c for c in cards if c.card.suit == BLACK]
    if black_cards:
        return max(black_cards, key=lambda c: c.card.value)

    # Step 5b: No trump. Highest card of led suit wins.
    led_cards = [c for c in cards if c.card.suit == led_suit]
    return max(led_cards, key=lambda c: c.card.value)
```

> **Note:** `first_played(type, cards)` returns the card with lowest `play_order` among matching type(s).

---

## 6. Scoring

### 6.1 Base score per round

| Condition                          | Points                         |
|------------------------------------|--------------------------------|
| `bid == 0` and `tricks_won == 0`   | `+round_number × 10`           |
| `bid == 0` and `tricks_won >= 1`   | `−round_number × 10`           |
| `bid > 0` and `tricks_won == bid`  | `+bid × 20` (+ any bonuses)    |
| `bid > 0` and `tricks_won != bid`  | `−abs(bid − tricks_won) × 10`  |

### 6.2 Bonus points

Bonuses apply **only when `bid > 0` and `tricks_won == bid`** (bid hit).

**CONFIRMED (2026-05-17):** No bonuses on missed bids. Zero-bid rounds never earn bonuses.

| Event                                                              | Bonus  | Awarded to        |
|--------------------------------------------------------------------|--------|-------------------|
| Skull King wins trick containing N Pirates                         | +30×N  | Skull King player |
| Mermaid wins trick containing Skull King                           | +40    | Mermaid player    |
| Black 14 captured in a trick won by Pirate, Mermaid, or Skull King | +20   | Trick winner      |

**CONFIRMED (2026-05-17):** Black 14 bonus (+20) is awarded only when the trick is won by a special card (Pirate, Tigress-as-Pirate, Mermaid, or Skull King). A numbered card winning a trick containing Black 14 earns no bonus.

> **AMBIGUOUS-06:** Does Tigress-as-Pirate count toward the Skull King +30 bonus?
> **Default resolution:** Yes. Tigress-as-Pirate is treated identically to a Pirate in all respects including bonus calculation.

### 6.3 Bonus computation algorithm

```
function compute_bonuses(trick: Trick, winner: Player) -> int:
    bonus = 0
    cards_in_trick = trick.all_cards()
    winner_type = trick.winning_card.type

    # Black 14 bonus — only if a special card won the trick
    SPECIAL_WINNERS = {SKULL_KING, MERMAID, PIRATE, TIGRESS_PIRATE}
    if winner_type in SPECIAL_WINNERS:
        if any(c.suit == BLACK and c.value == 14 for c in cards_in_trick):
            bonus += 20

    # Skull King won the trick
    if winner_type == SKULL_KING:
        pirate_count = sum(
            1 for c in cards_in_trick
            if c.type in (PIRATE, TIGRESS_PIRATE)
        )
        bonus += 30 * pirate_count

    # Mermaid won the trick (Skull King was present)
    if winner_type == MERMAID:
        if any(c.type == SKULL_KING for c in cards_in_trick):
            bonus += 40

    return bonus  # 0 if player missed bid (applied at round scoring time)
```

---

## 7. Bidding Rules

### 7.1 Procedure
1. All players **simultaneously** select a bid (closed fist over table, extend fingers on "3-2-1-reveal").
2. Valid bids: integer in `[0, round_number]`.
3. **No restrictions** on other players' bids — duplicate bids are legal.
4. Bids are **public and immutable** once revealed.

### 7.2 Bid 0 ("Scoundrel")
- The player is attempting to win **zero** tricks this round.
- If successful: `+round_number × 10` (e.g., Round 5 → +50).
- If they accidentally win any trick: `−round_number × 10`.
- Bonuses **do not apply** to bid-0 rounds.

> **AMBIGUOUS-07:** Can a player who bid 0 intentionally lose a trick they are about to win?
> **Default resolution:** No. A player must play a legal card but is not required to win. They cannot play an illegal card to avoid winning. The game does not force "must win if able" — they can play low legally. This is a strategic, not rule, matter.

---

## 8. Round Structure (Per-Round Flow)

```
1. DEAL
   - Shuffle full 70-card deck.
   - Deal `round_number` cards to each player, face-down.
   - Remaining cards go to a discard pile (not used this round).

2. BID
   - All players examine their hand privately.
   - Simultaneous bid reveal (see §7).
   - Record bids publicly.

3. PLAY TRICKS (repeat `round_number` times)
   a. Trick leader plays any card from hand.
   b. Remaining players in clockwise order each play one card (following suit rules §4).
   c. Resolve trick winner (§5).
   d. Winner takes the trick face-down in front of them.
   e. Winner becomes trick leader for the next trick.

4. SCORE
   - Count each player's tricks_won.
   - Compute base score (§6.1) + bonuses (§6.2) for each player.
   - Add to cumulative score.

5. PASS DEAL clockwise; begin next round.
```

---

## 9. Edge Cases Catalogue

### 9.1 All players play Escape (or Tigress-as-Escape)
- **Winner:** The trick **leader** (first card player) wins the trick.
- Trick counts toward their tricks_won total.
- No bonuses are possible.

### 9.2 Escape leads the trick
- No led suit is established.
- Subsequent players may play **any** card.
- Resolution: highest trump (Black) wins; if no Black, highest card of the first non-Escape suit played wins; if all Escape, trick leader wins.

> **AMBIGUOUS-08:** When Escape leads and multiple colored suits are played (no Black, no specials), which is "led suit"?
> **Default resolution:** The first non-Escape card played establishes an informal led suit. Highest card of that suit wins. Later players who played a different colored suit lose (they weren't forced to follow since no suit was led at trick start).

### 9.3 Skull King played against only Escapes (no Pirates)
- Skull King wins.
- No Pirate bonus (0 Pirates in trick).
- Black 14 bonus applies if Black 14 is one of the Escapes? No — Escapes have no value/suit. Only applies if Black 14 (numbered) is in the trick.

### 9.4 Mermaid wins trick with no Skull King
- No +40 bonus; Mermaid simply wins the trick normally.

### 9.5 Skull King and two Mermaids in same trick, no Pirate
- First Mermaid played wins; +40 bonus awarded to that player.
- Second Mermaid plays no special role.

### 9.6 Black 14 played as trump in a trick already won by a Pirate
- Pirate wins (Rank 3 > Rank 4).
- Black 14 bonus (+20) goes to the **Pirate player** who won the trick (they captured Black 14).

### 9.7 Player has only special cards (no colored cards)
- May play any special card; no suit-following obligation (special cards have no suit).

### 9.8 Round 1 (one card each)
- Each player has exactly 1 card; bid must be 0 or 1.
- The single trick is played; the winner leads (irrelevant since round ends).
- Scoring proceeds normally.

### 9.9 Tigress declared mid-play
- Declaration must happen the moment Tigress leaves the player's hand.
- If player forgets to declare: **Default resolution:** treat as Escape (least advantageous, penalizes the oversight). Flag in UI/game log.

---

## 10. Implementation Constants (use these in code)

```python
SUITS = ["BLACK", "YELLOW", "GREEN", "PURPLE"]
TRUMP_SUIT = "BLACK"

CARD_TYPES = ["NUMBERED", "ESCAPE", "PIRATE", "MERMAID", "SKULL_KING", "TIGRESS"]

DECK_COUNTS = {
    "NUMBERED": 56,   # 14 values × 4 suits
    "ESCAPE":    5,
    "PIRATE":    5,
    "MERMAID":   2,
    "SKULL_KING":1,
    "TIGRESS":   1,
}
DECK_TOTAL = 70

NUM_ROUNDS = 10
MAX_PLAYERS = 6

# Scoring constants
BID_HIT_MULTIPLIER     = 20   # per trick bid
BID_MISS_MULTIPLIER    = -10  # per trick difference (absolute)
BID_ZERO_HIT           = 10   # per round number
BID_ZERO_MISS          = -10  # per round number

# Bonus constants (only on bid hit, bid > 0)
BONUS_PIRATE_CAPTURED  = 30   # per pirate, awarded to SK winner
BONUS_SK_CAPTURED      = 40   # awarded to Mermaid winner
BONUS_BLACK_14         = 20   # awarded to trick winner ONLY if won by Pirate/Mermaid/SK
```

---

## 11. Open Ambiguities — Decision Log

| ID            | Rule area              | Question                                         | Default chosen       | Action needed           |
|---------------|------------------------|--------------------------------------------------|----------------------|-------------------------|
| AMBIGUOUS-01  | Players                | 7–8 player support                               | Cap at 6             | Confirm before adding   |
| AMBIGUOUS-02  | Trick resolution       | Two Mermaids, no SK, no Pirate                   | First played wins    | Low priority            |
| AMBIGUOUS-03  | Suit following         | Can Black override a colored lead?               | **CONFIRMED: No** — strict follow, Black only when void in led suit | Done |
| AMBIGUOUS-04  | Bonuses                | Bonuses on missed bid?                           | **CONFIRMED: No** — bonuses on bid success only | Done |
| AMBIGUOUS-05  | Black 14 bonus         | Any winner or special-card winner only?          | **CONFIRMED: Special card winner only** (Pirate, Mermaid, SK) | Done |
| AMBIGUOUS-06  | Tigress bonus          | Tigress-as-Pirate counts for SK +30?             | Yes                  | Low priority            |
| AMBIGUOUS-07  | Bid 0 strategy         | Forced to win if only winning card?              | Not forced (legal play only) | N/A — strategic |
| AMBIGUOUS-08  | Escape leads           | Which colored suit wins when multiple present?   | First non-Escape suit| Low priority            |

---

*Last updated: 2026-05-17. AMBIGUOUS-03, -04, -05 resolved by user. Safe to implement `trick.py` and `scoring.py`.*
