#pragma once

#include "state.hpp"
#include "action.hpp"
#include <vector>

namespace sk {

// Fixed-capacity action list (no heap allocation) for hot search loops.
// Max: 11 bids (round 10), 10 cards, or 2 Tigress modes.
struct ActionList {
    static constexpr int CAPACITY = 16;
    Action a[CAPACITY];
    int    n = 0;
    void push(Action x) { a[n++] = x; }
    const Action& operator[](int i) const { return a[i]; }
};

// All legal actions in the current state for state.currentPlayer.
void legalActionsInto(const GameState& s, ActionList& out);
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
