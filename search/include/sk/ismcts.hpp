#pragma once

#include "sk/agent.hpp"
#include "sk/state.hpp"
#include "sk/action.hpp"
#include <array>
#include <vector>
#include <random>

namespace sk {

struct ISMCTSConfig {
    int    numSimulations  = 800;     // total simulations (= determinizations) per decision
    double explorationC    = 1.41421356;  // sqrt(2)
    double rewardScale     = 200.0;   // divisor so rewards land in roughly [-1, 1] for UCB
};

// Returned by ISMCTSAgent::selectActionWithTargets. Contains the chosen action
// plus the information needed to construct policy and value training targets:
//   - `rootActions[i]` is the i-th child action at the root.
//   - `visits[i]`     is its visit count (use as policy target after normalisation).
//   - `values[p]`     is player p's average reward at the root over all sims
//                     (in the same units as `rewardScale` from the config — typically points).
//                     The agent's own value is `values[state.currentPlayer]`.
struct ISMCTSResult {
    Action                              action;
    std::vector<Action>                 rootActions;
    std::vector<int>                    visits;
    std::array<double, N_PLAYERS>       values{};
};

// Single-Observer ISMCTS (Cowling, Powley, Whitehouse 2012).
// One tree across all determinizations; per rollout we re-determinize the
// hidden information and walk the tree using UCB1 over actions legal in the
// current determinization.
class ISMCTSAgent final : public Agent {
public:
    explicit ISMCTSAgent(ISMCTSConfig cfg = {}) : cfg_(cfg) {}

    Action selectAction(const GameState& s, std::mt19937_64& rng) override;
    const char* name() const override { return "ismcts"; }

    // Like selectAction, but also returns the per-child visit counts and the
    // root value estimate for use as training targets.
    ISMCTSResult selectActionWithTargets(const GameState& s, std::mt19937_64& rng);

    void setConfig(ISMCTSConfig cfg) { cfg_ = cfg; }
    const ISMCTSConfig& config() const { return cfg_; }

private:
    ISMCTSConfig cfg_;
};

} // namespace sk
