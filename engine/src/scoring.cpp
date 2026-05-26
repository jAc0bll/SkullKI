#include "sk/scoring.hpp"
#include <cstdlib>

namespace sk {

std::int32_t bidPoints(int bid, int won, int roundSize) {
    if (bid == 0) {
        return (won == 0) ? (10 * roundSize) : (-10 * roundSize);
    }
    if (bid == won) {
        return 20 * won;
    }
    return -10 * std::abs(bid - won);
}

std::int32_t bonusPoints(const CardSet& captured, std::int32_t pendingBonus) {
    std::int32_t bonus = pendingBonus;
    captured.forEach([&](Card c) {
        if (is14(c)) {
            bonus += isTrump(c) ? 20 : 10;
        } else if (isMermaid(c)) {
            bonus += 20;
        }
    });
    return bonus;
}

std::array<RoundScore, N_PLAYERS> scoreRound(GameState& s) {
    std::array<RoundScore, N_PLAYERS> out{};
    for (int p = 0; p < N_PLAYERS; ++p) {
        const int bid       = s.bids[p];
        const int won       = s.tricksWon[p];
        const int roundSize = s.roundNumber;

        const std::int32_t bp = bidPoints(bid, won, roundSize);
        // Bonus only awarded when the player hit their bid exactly.
        const std::int32_t bo = (bid == won) ? bonusPoints(s.captured[p], s.pendingBonus[p]) : 0;

        out[p].bidPoints   = bp;
        out[p].bonusPoints = bo;
        out[p].total       = bp + bo;
        s.scores[p] += bp + bo;
    }
    return out;
}

} // namespace sk
