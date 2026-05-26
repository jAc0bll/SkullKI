#pragma once

#include "sk/state.hpp"
#include <random>

namespace sk {

// Sample a "possible world" from the MCTS player's information set.
// Inputs:
//   src         : the true GameState (the determinizer treats the
//                 perspective player's hand as known and IGNORES the
//                 actual hands of the others, redistributing uniformly).
//   perspective : player index (0..N_PLAYERS-1) whose information set we sample from.
//   rng         : random generator.
// Output:
//   A GameState identical to src except opponents' hands are replaced by
//   a random assignment of the unseen cards, respecting each opponent's
//   publicly known hand size.
GameState determinize(const GameState& src, int perspective, std::mt19937_64& rng);

} // namespace sk
