#include "sk/rules.hpp"

namespace sk {

namespace {

inline bool effPirate(Card c, bool tigressAsPirate) {
    return isPirate(c) || (isTigress(c) && tigressAsPirate);
}

} // namespace

TrickResult resolveTrick(const Card* cards,
                         const std::int8_t* players,
                         int n,
                         bool tigressAsPirate)
{
    int skIdx        = -1;
    int firstMermaid = -1;
    int firstPirate  = -1;
    int pirateCount  = 0;

    for (int i = 0; i < n; ++i) {
        const Card c = cards[i];
        if (isSkullKing(c)) {
            if (skIdx < 0) skIdx = i;
        } else if (isMermaid(c)) {
            if (firstMermaid < 0) firstMermaid = i;
        } else if (effPirate(c, tigressAsPirate)) {
            if (firstPirate < 0) firstPirate = i;
            ++pirateCount;
        }
    }

    TrickResult r{};
    if (skIdx >= 0 && firstMermaid >= 0) {
        // Mermaid catches the Skull King (first-played mermaid).
        r.winner = players[firstMermaid];
        r.bonusForWinner = 40;
        return r;
    }
    if (skIdx >= 0) {
        r.winner = players[skIdx];
        r.bonusForWinner = 30 * pirateCount;
        return r;
    }
    if (firstPirate >= 0) {
        r.winner = players[firstPirate];
        return r;
    }
    if (firstMermaid >= 0) {
        r.winner = players[firstMermaid];
        return r;
    }

    // No pirates / mermaids / SK. Only colored + escapes (+ tigress-as-escape).
    Suit lead = Suit::None;
    for (int i = 0; i < n; ++i) {
        if (isColored(cards[i])) {
            lead = suitOf(cards[i]);
            break;
        }
    }
    if (lead == Suit::None) {
        // All escapes — first card wins.
        r.winner = players[0];
        return r;
    }

    int bestIdx = -1;
    int bestVal = -1;

    // Highest trump (black) wins if any.
    for (int i = 0; i < n; ++i) {
        const Card c = cards[i];
        if (isColored(c) && suitOf(c) == Suit::Black) {
            const int v = valueOf(c);
            if (v > bestVal) { bestVal = v; bestIdx = i; }
        }
    }
    if (bestIdx < 0) {
        // No trump played — highest of lead suit wins.
        for (int i = 0; i < n; ++i) {
            const Card c = cards[i];
            if (isColored(c) && suitOf(c) == lead) {
                const int v = valueOf(c);
                if (v > bestVal) { bestVal = v; bestIdx = i; }
            }
        }
    }

    r.winner = players[bestIdx];
    return r;
}

std::vector<Action> legalActions(const GameState& s) {
    std::vector<Action> out;

    if (s.phase == Phase::Bidding) {
        out.reserve(s.roundNumber + 1);
        for (int b = 0; b <= s.roundNumber; ++b) {
            out.push_back(Action::makeBid(b));
        }
        return out;
    }

    if (s.phase != Phase::Playing) return out;

    if (s.pendingTigress) {
        out.push_back(Action::makeTigressMode(true));
        out.push_back(Action::makeTigressMode(false));
        return out;
    }

    const CardSet& hand = s.hands[s.currentPlayer];
    out.reserve(10);

    const bool firstPlay = (s.trickSize == 0);
    const bool freeChoice = firstPlay || s.freeTrick || s.leadSuit == Suit::None;

    if (freeChoice) {
        hand.forEach([&](Card c) { out.push_back(Action::makePlay(c)); });
        return out;
    }

    // Must follow leadSuit if hand contains any card of that suit.
    bool hasLeadSuit = false;
    hand.forEach([&](Card c) {
        if (isColored(c) && suitOf(c) == s.leadSuit) hasLeadSuit = true;
    });

    if (hasLeadSuit) {
        hand.forEach([&](Card c) {
            if (isSpecial(c) || suitOf(c) == s.leadSuit) {
                out.push_back(Action::makePlay(c));
            }
        });
    } else {
        // No card of lead suit — anything goes.
        hand.forEach([&](Card c) { out.push_back(Action::makePlay(c)); });
    }
    return out;
}

} // namespace sk
