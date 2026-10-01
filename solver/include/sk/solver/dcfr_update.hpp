#pragma once

#include "sk/solver/table.hpp"

#include <array>
#include <unordered_map>
#include <vector>

namespace sk::solver {

// Discounted CFR (Brown & Sandholm 2019). alpha=1.5, beta=0, gamma=2 is the
// recommended default; alpha=beta=gamma=1 gives Linear CFR.
struct DCFRParams {
    double alpha = 1.5;
    double beta  = 0.0;
    double gamma = 2.0;
};

// One iteration's (or batch's) regret / average-strategy contributions for an
// infoset, accumulated per thread so hot infosets are not contended.
struct Delta {
    std::array<double, ActionList::CAPACITY> dR{};
    std::array<double, ActionList::CAPACITY> dS{};
};
using DeltaMap = std::unordered_map<InfoNode*, Delta>;

// Apply the deltas of update step t (1-based) to the whole table:
//   S <- S * (t/(t+1))^gamma + dS
//   R <- (R + dR) * (t^alpha/(t^alpha+1) if positive, t^beta/(t^beta+1) if not)
// then recompute every node's current strategy by regret matching.
void applyDCFRUpdate(InfosetTable& table, const std::vector<const DeltaMap*>& deltas,
                     int t, const DCFRParams& params);

} // namespace sk::solver
