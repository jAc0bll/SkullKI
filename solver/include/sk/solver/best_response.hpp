#pragma once

#include "sk/solver/round.hpp"
#include "sk/solver/table.hpp"

#include <array>

namespace sk::solver {

struct EvalConfig {
    int     round   = 1;
    Utility utility = Utility::Relative;
    int     threads = 0;
};

// Exact expected utilities of a strategy profile (all deals enumerated).
std::array<double, N_PLAYERS> policyValue(const PolicyView& policy, const EvalConfig& cfg);

// Exact best-response value for `player` against the other players' policy.
// Computed in passes from the deepest decision level of `player` up to the
// root: a level's best action only depends on deeper (already fixed) levels.
double bestResponseValue(const PolicyView& policy, int player, const EvalConfig& cfg);

struct ExploitabilityReport {
    std::array<double, N_PLAYERS> value{};    // utility of each player under the profile
    std::array<double, N_PLAYERS> brValue{};  // utility if that player best-responds
    std::array<double, N_PLAYERS> gain{};     // brValue - value (>= 0)
    double nashConv = 0.0;                    // sum of gains; 0 <=> Nash equilibrium
};

ExploitabilityReport exploitability(const PolicyView& policy, const EvalConfig& cfg);

} // namespace sk::solver
