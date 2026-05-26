#pragma once

#include "state.hpp"
#include <array>

namespace sk {

// What a single player can legitimately see.
// - `s` is a copy of the underlying GameState with opponents' hands wiped to empty,
//   so an agent operating on Observation cannot accidentally peek at hidden info.
// - `handSizes` keeps the publicly-known hand size for each player (always derivable
//   from the action history, but cached here for convenience).
// - `perspective` is the index of the observing player.
struct Observation {
    GameState s;
    std::array<std::int8_t, N_PLAYERS> handSizes{};
    std::int8_t perspective = 0;
};

// Build an Observation from `src` for player `perspective`.
// The returned struct contains only public information plus the perspective
// player's own hand. Opponents' hands are cleared.
Observation observe(const GameState& src, int perspective);

} // namespace sk
