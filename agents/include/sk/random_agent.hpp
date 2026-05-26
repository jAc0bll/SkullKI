#pragma once

#include "sk/agent.hpp"

namespace sk {

// Picks uniformly from legalActions(s).
class RandomAgent final : public Agent {
public:
    Action selectAction(const GameState& s, std::mt19937_64& rng) override;
    const char* name() const override { return "random"; }
};

} // namespace sk
