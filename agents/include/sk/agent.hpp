#pragma once

#include "sk/state.hpp"
#include "sk/action.hpp"
#include <random>

namespace sk {

// Abstract decision-maker. Receives the full GameState by reference but is
// expected to only access information legitimately available to the
// current player (own hand, public history). Hidden-info enforcement is
// done at the Observation layer (added in Phase 3 with the NN encoder).
class Agent {
public:
    virtual ~Agent() = default;
    virtual Action selectAction(const GameState& s, std::mt19937_64& rng) = 0;
    virtual const char* name() const = 0;
};

} // namespace sk
