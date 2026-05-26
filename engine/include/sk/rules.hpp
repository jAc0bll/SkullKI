#pragma once

#include "state.hpp"
#include "action.hpp"
#include <vector>

namespace sk {

// All legal actions in the current state for state.currentPlayer.
std::vector<Action> legalActions(const GameState& s);

// Pure trick-resolution helper.
struct TrickResult {
    std::int8_t  winner;          // player index of trick winner
    std::int32_t bonusForWinner;  // pirate-by-SK (+30/pirate) or SK-by-mermaid (+40)
};

// `cards`/`players` arrays of length `n` in play order.
// `tigressAsPirate` is the resolved mode of any Tigress in the trick.
TrickResult resolveTrick(const Card* cards,
                         const std::int8_t* players,
                         int n,
                         bool tigressAsPirate);

} // namespace sk
