#include "sk/heuristic_agent.hpp"
#include "sk/rules.hpp"
#include "sk/cards.hpp"

#include <algorithm>
#include <cmath>

namespace sk {

namespace {

// Per-card trick equity used for bidding (rough P(this card wins a trick)).
double bidEquity(Card c) {
    if (isSkullKing(c)) return 0.95;
    if (isPirate(c))    return 0.80;
    if (isTigress(c))   return 0.70;
    if (isMermaid(c))   return 0.55;
    if (isEscape(c))    return 0.00;
    const int v = valueOf(c);
    if (suitOf(c) == Suit::Black) return 0.10 + 0.06 * v;     // 0.16 .. 0.94
    if (v >= 13) return 0.40;
    if (v >= 11) return 0.22;
    if (v >= 8)  return 0.10;
    return 0.03;
}

// Higher == stronger candidate to win the trick from any position.
double cardStrength(Card c, Suit leadSuit) {
    if (isSkullKing(c)) return 1000.0;
    if (isPirate(c))    return 900.0;
    if (isTigress(c))   return 850.0;
    if (isMermaid(c))   return 800.0;
    if (isEscape(c))    return 0.0;
    const int v = valueOf(c);
    if (suitOf(c) == Suit::Black)   return 600.0 + 15.0 * v;
    if (suitOf(c) == leadSuit)      return 300.0 + 10.0 * v;
    return static_cast<double>(v);  // off-suit non-trump rarely wins
}

bool wouldWinNow(const GameState& s, Card candidate, int player, bool tigressAsPirate) {
    Card cards[N_PLAYERS];
    std::int8_t players[N_PLAYERS];
    for (int i = 0; i < s.trickSize; ++i) {
        cards[i]   = s.trickCards[i];
        players[i] = s.trickPlayers[i];
    }
    cards[s.trickSize]   = candidate;
    players[s.trickSize] = static_cast<std::int8_t>(player);
    const auto r = resolveTrick(cards, players, s.trickSize + 1, tigressAsPirate);
    return r.winner == player;
}

} // namespace

Action HeuristicAgent::selectAction(const GameState& s, std::mt19937_64& /*rng*/) {
    const int p = s.currentPlayer;

    // ---- Bidding ----
    if (s.phase == Phase::Bidding) {
        double e = 0;
        s.hands[p].forEach([&](Card c) { e += bidEquity(c); });
        int bid = static_cast<int>(std::round(e));
        bid = std::clamp(bid, 0, static_cast<int>(s.roundNumber));
        return Action::makeBid(bid);
    }

    // ---- TigressMode ----
    if (s.pendingTigress) {
        const int need = s.bids[p] - s.tricksWon[p];
        return Action::makeTigressMode(need > 0);
    }

    // ---- Play ----
    const auto la = legalActions(s);
    const int need      = s.bids[p] - s.tricksWon[p];
    const int remaining = s.roundNumber - s.tricksPlayed;
    const bool wantWin    = (need > 0);
    const bool mustWinAll = (need >= remaining);
    const bool firstPlay  = (s.trickSize == 0);

    Card bestCard = la.front().card;
    double bestScore = -1e18;

    for (const Action& a : la) {
        const Card c = a.card;
        const double strength = cardStrength(c, s.leadSuit);
        // For Tigress in non-pendingTigress case (played first): the mode will be chosen later.
        // For evaluating "wouldWinNow", treat Tigress as pirate iff we want a win.
        const bool tigMode = wantWin;

        bool wins;
        if (firstPlay) {
            // Leading: trick outcome depends on opponents. Use strength heuristic.
            wins = (strength >= 500.0);   // rough threshold: trumps & specials likely win
        } else {
            wins = wouldWinNow(s, c, p, /*tigressAsPirate=*/tigMode);
        }

        double score;
        if (mustWinAll) {
            // Spend our strongest cards to maximize win probability.
            score = strength;
        } else if (wantWin) {
            if (wins) {
                // Win as cheaply as possible (save strong cards for later tricks).
                score = 1e4 - strength;
            } else {
                // Cannot win this one — discard weakest.
                score = -strength;
            }
        } else {
            // Want to lose this trick.
            if (!wins) {
                // Dump weakest (keep strong cards for tricks we DO want).
                score = -strength;
            } else {
                // Forced to play a winning card — pick the weakest winner.
                score = -strength - 1e4;
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestCard  = c;
        }
    }

    return Action::makePlay(bestCard);
}

} // namespace sk
