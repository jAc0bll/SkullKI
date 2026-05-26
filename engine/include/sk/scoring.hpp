#pragma once

#include "state.hpp"

namespace sk {

// Per-player breakdown produced by scoreRound (useful for tests / analytics).
struct RoundScore {
    std::int32_t bidPoints   = 0;  // ±10*delta, ±10*roundSize, or +20*tricks
    std::int32_t bonusPoints = 0;  // 14s, mermaids, pirate-by-SK, SK-by-mermaid (only if bid hit)
    std::int32_t total       = 0;  // bidPoints + bonusPoints
};

// Computes scores for the just-finished round, updates s.scores[], and returns the breakdown.
// Does NOT mutate any other state — caller is responsible for clearing round state.
std::array<RoundScore, N_PLAYERS> scoreRound(GameState& s);

// Pure helper: bid-only points (no bonus). Useful for testing.
std::int32_t bidPoints(int bid, int won, int roundSize);

// Pure helper: bonus points for a player given their captured set + pendingBonus.
std::int32_t bonusPoints(const CardSet& captured, std::int32_t pendingBonus);

} // namespace sk
