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
    int mermaidCount = 0;

    for (int i = 0; i < n; ++i) {
        const Card c = cards[i];
        if (isSkullKing(c)) {
            if (skIdx < 0) skIdx = i;
        } else if (isMermaid(c)) {
            if (firstMermaid < 0) firstMermaid = i;
            ++mermaidCount;
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
        // +20 per Mermaid captured by a Pirate (incl. Tigress-as-Pirate).
        r.winner = players[firstPirate];
        r.bonusForWinner = 20 * mermaidCount;
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

void legalActionsInto(const GameState& s, ActionList& out) {
    out.n = 0;

    if (s.phase == Phase::Bidding) {
        for (int b = 0; b <= s.roundNumber; ++b) {
            out.push(Action::makeBid(b));
        }
        return;
    }

    if (s.phase != Phase::Playing) return;

    if (s.pendingTigress) {
        out.push(Action::makeTigressMode(true));
        out.push(Action::makeTigressMode(false));
        return;
    }

    const CardSet& hand = s.hands[s.currentPlayer];

    const bool firstPlay = (s.trickSize == 0);
    const bool freeChoice = firstPlay || s.freeTrick || s.leadSuit == Suit::None;

    if (freeChoice) {
        hand.forEach([&](Card c) { out.push(Action::makePlay(c)); });
        return;
    }

    // Must follow leadSuit if hand contains any card of that suit.
    bool hasLeadSuit = false;
    hand.forEach([&](Card c) {
        if (isColored(c) && suitOf(c) == s.leadSuit) hasLeadSuit = true;
    });

    if (hasLeadSuit) {
        hand.forEach([&](Card c) {
            if (isSpecial(c) || suitOf(c) == s.leadSuit) {
                out.push(Action::makePlay(c));
            }
        });
    } else {
        // No card of lead suit — anything goes.
        hand.forEach([&](Card c) { out.push(Action::makePlay(c)); });
    }
}

std::vector<Action> legalActions(const GameState& s) {
    ActionList al;
    legalActionsInto(s, al);
    return std::vector<Action>(al.a, al.a + al.n);
}

} // namespace sk
