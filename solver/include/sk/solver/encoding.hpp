#pragma once

#include "sk/solver/round.hpp"

#include <cstdint>

namespace sk::solver {

// ---------------------------------------------------------------------------
// Network input/output layout for the neural (Deep CFR family) solver.
//
// Actions: one output per action kind
//   [0..10]  bid 0..10
//   [11..71] play card kind 0..60
//   [72]     Tigress declared as Pirate
//   [73]     Tigress declared as Escape
//
// Features are small non-negative integers stored as uint8 (compact replay
// buffers); the network converts them to floats. Everything is relative to
// the observing player ("me" = 0, then clockwise).
// ---------------------------------------------------------------------------
constexpr int ACT_DIM = 11 + 61 + 2;

int actionIndex(const Action& a);

// What `player` knows (never other hands, never hidden bids).
constexpr int INFO_DIM =
      10        // round one-hot
    + 3         // phase: bidding / playing / Tigress mode pending
    + 4         // my seat relative to the round's start player
    + 4         // my position in the current trick
    + 2         // tricks played, tricks left
    + 61        // my hand (kind counts)
    + 61        // cards played in completed tricks (kind counts)
    + 4 * 62    // current trick by position from leader: kind one-hot + "present"
    + 2         // Tigress in trick declared Pirate / Escape
    + 4 * 12    // bids per player (one-hot 0..10, 11 = unknown)
    + 4         // tricks won per player
    + 4         // bid - won per player, offset by 10 (unknown bid: 0)
    + 4 * 4     // public voids per player and suit
    + 4         // trick leader (one-hot by player)
    + 61        // unseen cards (kind counts)
    + 61;       // per hand kind: would win the trick if played now

// Full history for the value network: `perspective`'s infoset plus the
// other players' hands (relative order).
constexpr int HIST_DIM = INFO_DIM + 3 * 61;

void encodeInfoset(const RoundState& rs, int player, std::uint8_t* out);
void encodeHistory(const RoundState& rs, int perspective, std::uint8_t* out);

// Legal action mask over ACT_DIM.
void legalMask(const ActionList& legal, std::uint8_t* out);

} // namespace sk::solver
