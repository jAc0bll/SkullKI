#pragma once

#include "observation.hpp"
#include <vector>

namespace sk {

// Dimension of the flat feature vector produced by `encode`.
// Layout (offsets are indicative; the implementation is the ground truth):
//   [0..69]    ownHand                            binary, 70
//   [70..73]   handSizes / 10                     float, 4
//   [74..77]   bids / 10  (-0.1 if not submitted) float, 4
//   [78..81]   bid valid flags                    binary, 4
//   [82..85]   tricksWon / 10                     float, 4
//   [86..89]   scores / 200                       float, 4
//   [90..99]   roundNumber one-hot (1..10)        binary, 10
//   [100..102] phase one-hot                      binary, 3
//   [103..106] currentPlayer one-hot              binary, 4
//   [107..110] startPlayer one-hot                binary, 4
//   [111..114] trickLeader one-hot                binary, 4
//   [115..119] leadSuit one-hot (Y/G/P/B/None)    binary, 5
//   [120..123] perspective one-hot                binary, 4
//   [124..127] flags: freeTrick, pendingTigress,
//              tigressAsPirate, bidsSubmitted/4   float, 4
//   [128..129] trickSize/4, tricksPlayed/10       float, 2
//   [130..409] per-player current-trick card      binary, 4 * 70 = 280
//   [410..689] per-player captured pile           binary, 4 * 70 = 280
//   [690..759] union of all cards played this rd  binary, 70
// Total: 760
constexpr int ENC_DIM = 760;

// Append the float encoding of `obs` to `out` (clears first).
void encodeInto(const Observation& obs, std::vector<float>& out);

// Convenience: returns a fresh vector.
std::vector<float> encode(const Observation& obs);

} // namespace sk
