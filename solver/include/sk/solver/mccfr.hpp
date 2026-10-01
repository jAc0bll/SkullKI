#pragma once

#include "sk/solver/dcfr_update.hpp"
#include "sk/solver/round.hpp"
#include "sk/solver/table.hpp"

#include <cstdint>

namespace sk::solver {

struct MCCFRConfig {
    int           round   = 1;
    Utility       utility = Utility::Relative;
    int           threads = 0;          // 0 = all hardware threads
    std::uint64_t seed    = 1;
    // Deals sampled per update step. Every deal is traversed once per player.
    int           dealsPerBatch = 1 << 16;
    DCFRParams    dcfr{};
};

// External-sampling Monte Carlo CFR (Lanctot et al. 2009) over one round.
//
// Instead of enumerating every deal and every opponent action (full width),
// each traversal samples one deal, explores all actions of the traversing
// player and samples one action at every other node. That makes the cost
// per traversal independent of the number of deals, which is what lets the
// method scale beyond round 1 — at the price of sampling noise.
//
// Traversals run in parallel against a fixed strategy snapshot; their regret
// and strategy deltas are merged after each batch with a DCFR update (one
// update step per batch).
class ExternalSamplingMCCFR {
public:
    explicit ExternalSamplingMCCFR(MCCFRConfig cfg);

    void runBatch();

    int           batches()    const { return batches_; }
    std::uint64_t traversals() const { return traversals_; }
    // Game-tree nodes touched so far (cost measure; grows ~ factorially with
    // the round because every action of the traverser is explored).
    std::uint64_t nodesVisited() const { return nodes_; }
    const InfosetTable& table() const { return table_; }
    const MCCFRConfig&  config() const { return cfg_; }

private:
    MCCFRConfig   cfg_;
    InfosetTable  table_;
    int           batches_    = 0;
    std::uint64_t traversals_ = 0;
    std::uint64_t nodes_      = 0;
};

} // namespace sk::solver
