#pragma once

#include "state.hpp"
#include "action.hpp"
#include "rules.hpp"
#include "scoring.hpp"
#include <random>

namespace sk {

// Fresh GameState ready to be dealt (round 1, bidding phase, empty hands).
GameState initialState(int startPlayer = 0);

// Deal the current round into s.hands[]. Resets per-round state.
// Must only be called when s.phase == Bidding and hands are empty.
void dealRound(GameState& s, std::mt19937_64& rng);

// State machine step.
// - In Bidding: a must be ActionType::Bid.
// - In Playing: a must be Play (or TigressMode if pendingTigress).
// Updates state; transitions phases. Does NOT deal new rounds — when a round
// finishes and the game isn't over, phase returns to Bidding with empty hands
// and the caller must invoke dealRound() before continuing.
void applyAction(GameState& s, Action a);

// Convenience: applyAction then deal if a new round just started.
void step(GameState& s, Action a, std::mt19937_64& rng);

inline bool isTerminal(const GameState& s) { return s.phase == Phase::GameEnd; }

} // namespace sk
