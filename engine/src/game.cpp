#include "sk/game.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <cassert>

namespace sk {

namespace {

void resetForNextRound(GameState& s) {
    for (int p = 0; p < N_PLAYERS; ++p) {
        s.bids[p]          = -1;
        s.tricksWon[p]     = 0;
        s.captured[p].clear();
        s.pendingBonus[p]  = 0;
        s.hands[p].clear();
    }
    s.bidsSubmitted    = 0;
    s.trickSize        = 0;
    s.tricksPlayed     = 0;
    s.leadSuit         = Suit::None;
    s.freeTrick        = false;
    s.pendingTigress   = false;
    s.tigressAsPirate  = false;
}

void onTrickComplete(GameState& s) {
    const TrickResult r = resolveTrick(
        s.trickCards, s.trickPlayers, s.trickSize, s.tigressAsPirate);

    s.tricksWon[r.winner]    += 1;
    s.pendingBonus[r.winner] += r.bonusForWinner;
    for (int i = 0; i < s.trickSize; ++i) {
        s.captured[r.winner].add(s.trickCards[i]);
    }
    ++s.tricksPlayed;

    s.trickLeader     = r.winner;
    s.currentPlayer   = r.winner;
    s.trickSize       = 0;
    s.leadSuit        = Suit::None;
    s.freeTrick       = false;
    s.pendingTigress  = false;
    s.tigressAsPirate = false;

    if (s.tricksPlayed >= s.roundNumber) {
        // Round end → score, then either advance round or finish game.
        scoreRound(s);

        if (s.roundNumber >= MAX_ROUND) {
            s.phase = Phase::GameEnd;
            return;
        }
        s.roundNumber  = static_cast<std::uint8_t>(s.roundNumber + 1);
        s.startPlayer  = static_cast<std::int8_t>((s.startPlayer + 1) % N_PLAYERS);
        s.currentPlayer = s.startPlayer;
        s.phase        = Phase::Bidding;
        resetForNextRound(s);
        // Caller must dealRound() before next applyAction.
    }
}

void advanceOrResolve(GameState& s) {
    if (s.trickSize < N_PLAYERS) {
        s.currentPlayer = static_cast<std::int8_t>((s.currentPlayer + 1) % N_PLAYERS);
        return;
    }
    onTrickComplete(s);
}

} // namespace

GameState initialState(int startPlayer) {
    GameState s{};
    s.roundNumber    = 1;
    s.phase          = Phase::Bidding;
    s.startPlayer    = static_cast<std::int8_t>(startPlayer);
    s.currentPlayer  = static_cast<std::int8_t>(startPlayer);
    resetForNextRound(s);
    return s;
}

void dealRound(GameState& s, std::mt19937_64& rng) {
    assert(s.phase == Phase::Bidding);
    for (int p = 0; p < N_PLAYERS; ++p) assert(s.hands[p].empty());

    std::array<Card, N_CARDS> deck{};
    std::iota(deck.begin(), deck.end(), Card{0});
    std::shuffle(deck.begin(), deck.end(), rng);

    int idx = 0;
    for (int p = 0; p < N_PLAYERS; ++p) {
        for (int i = 0; i < s.roundNumber; ++i) {
            s.hands[p].add(deck[idx++]);
        }
    }
}

void applyAction(GameState& s, Action a) {
    if (s.phase == Phase::Bidding) {
        assert(a.type == ActionType::Bid);
        assert(a.bid <= s.roundNumber);
        s.bids[s.currentPlayer] = static_cast<std::int8_t>(a.bid);
        ++s.bidsSubmitted;
        s.currentPlayer = static_cast<std::int8_t>((s.currentPlayer + 1) % N_PLAYERS);

        if (s.bidsSubmitted == N_PLAYERS) {
            s.phase         = Phase::Playing;
            s.currentPlayer = s.startPlayer;
            s.trickLeader   = s.startPlayer;
            s.trickSize     = 0;
            s.leadSuit      = Suit::None;
            s.freeTrick     = false;
            s.pendingTigress = false;
            s.tigressAsPirate = false;
        }
        return;
    }

    assert(s.phase == Phase::Playing);

    if (a.type == ActionType::TigressMode) {
        assert(s.pendingTigress);
        s.tigressAsPirate = a.asPirate;
        s.pendingTigress  = false;

        // If Tigress was the lead card, its mode determines whether the trick is
        // character-led (freeTrick) or escape-led.
        if (s.trickSize == 1) {
            if (a.asPirate) {
                s.freeTrick = true;
                s.leadSuit  = Suit::None;
            } else {
                s.freeTrick = false;
                s.leadSuit  = Suit::None;  // next colored card sets it
            }
        }
        advanceOrResolve(s);
        return;
    }

    assert(a.type == ActionType::Play);
    assert(!s.pendingTigress);
    const Card c = a.card;
    const std::int8_t p = s.currentPlayer;
    assert(s.hands[p].has(c));

    s.hands[p].remove(c);
    s.trickCards[s.trickSize]   = c;
    s.trickPlayers[s.trickSize] = p;
    ++s.trickSize;

    if (s.trickSize == 1) {
        // Lead card
        if (isColored(c)) {
            s.leadSuit  = suitOf(c);
            s.freeTrick = false;
        } else if (isEscape(c)) {
            s.leadSuit  = Suit::None;
            s.freeTrick = false;
        } else if (isTigress(c)) {
            // Mode decision deferred — leadSuit/freeTrick set after TigressMode.
            s.pendingTigress = true;
            return;
        } else {
            // Pirate / Mermaid / Skull King — character lead.
            s.leadSuit  = Suit::None;
            s.freeTrick = true;
        }
    } else {
        // Follow card: in escape-led tricks, the first colored card sets leadSuit.
        if (!s.freeTrick && s.leadSuit == Suit::None && isColored(c)) {
            s.leadSuit = suitOf(c);
        }
        if (isTigress(c)) {
            s.pendingTigress = true;
            return;
        }
    }

    advanceOrResolve(s);
}

void step(GameState& s, Action a, std::mt19937_64& rng) {
    const std::uint8_t prevRound = s.roundNumber;
    const Phase       prevPhase  = s.phase;
    applyAction(s, a);
    // If applyAction just transitioned us to a new betting round, deal the new hands.
    if (s.phase == Phase::Bidding &&
        (prevPhase != Phase::Bidding || s.roundNumber != prevRound) &&
        s.hands[0].empty())
    {
        dealRound(s, rng);
    }
}

} // namespace sk
