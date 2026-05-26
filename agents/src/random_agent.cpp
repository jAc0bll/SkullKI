#include "sk/random_agent.hpp"
#include "sk/rules.hpp"

namespace sk {

Action RandomAgent::selectAction(const GameState& s, std::mt19937_64& rng) {
    const auto la = legalActions(s);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
    return la[dist(rng)];
}

} // namespace sk
