#pragma once

#include "sk/solver/round.hpp"
#include "sk/solver/table.hpp"

#include <array>

namespace sk::solver {

struct CFRConfig {
    int     round   = 1;
    Utility utility = Utility::Relative;
    int     threads = 0;          // 0 = all hardware threads
    // Discounted CFR (Brown & Sandholm 2019). alpha=1.5, beta=0, gamma=2 is
    // the recommended default; alpha=beta=gamma=1 gives Linear CFR.
    double  alpha = 1.5;
    double  beta  = 0.0;
    double  gamma = 2.0;
};

// Full-width (non-sampled) Discounted CFR over one round.
//
// Every iteration enumerates ALL deals (at kind level, with exact
// probabilities) and every action of every player, so all infosets are
// updated with exact counterfactual values. Feasible for tiny rounds only
// (round 1 with 4 players); larger rounds need sampling / function
// approximation. Updates are simultaneous for all players.
class FullWidthCFR {
public:
    explicit FullWidthCFR(CFRConfig cfg);

    void iterate();
    int iteration() const { return t_; }

    const InfosetTable& table() const { return table_; }
    const CFRConfig& config() const { return cfg_; }

    // Expected utilities of the CURRENT (not average) strategy profile during
    // the last iteration — a free by-product of the traversal.
    const std::array<double, N_PLAYERS>& lastCurrentValue() const { return lastValue_; }

private:
    CFRConfig    cfg_;
    InfosetTable table_;
    int          t_ = 0;
    std::array<double, N_PLAYERS> lastValue_{};
};

} // namespace sk::solver
