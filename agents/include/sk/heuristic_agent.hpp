#pragma once

#include "sk/agent.hpp"

namespace sk {

// Simple deterministic baseline:
//  - Bid: estimate expected tricks from per-card "trick equity".
//  - Play: based on need = (bid - tricksWon) vs tricks remaining,
//          either chase wins (cheapest winning card) or dump (weakest card).
//  - TigressMode: Pirate if still need tricks, Escape otherwise.
class HeuristicAgent final : public Agent {
public:
    Action selectAction(const GameState& s, std::mt19937_64& rng) override;
    const char* name() const override { return "heuristic"; }
};

} // namespace sk
