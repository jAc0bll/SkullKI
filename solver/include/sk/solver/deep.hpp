#pragma once

#include "sk/solver/encoding.hpp"
#include "sk/solver/mlp.hpp"
#include "sk/solver/table.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sk::solver {

// ---------------------------------------------------------------------------
// Neural CFR (ESCHER-style, McAleer et al. 2023), data generation side.
//
// Per iteration t (orchestrated by train/deep_cfr.py):
//   1. value data: on-policy returns under the current profile sigma_t
//      -> train value net V (input: full history from one perspective)
//   2. regret data: for each traverser i, trajectories where i acts
//      UNIFORMLY at random (a fixed sampling policy, so no importance
//      weights are needed) and opponents follow sigma_t. At each of i's
//      decisions the regret of every legal action is estimated without any
//      subtree search:  r(a) = V(h.a) - sum_b sigma(b) V(h.b)
//      Opponent decisions yield (infoset, sigma) samples for the average
//      strategy.
//   3. train a regret net on all regret samples (weight t, Linear CFR);
//      sigma_{t+1} = regret matching on its output.
// Cost per trajectory is linear in the round length, unlike external
// sampling, whose traversals grow ~6x per round.
// ---------------------------------------------------------------------------

enum class NetMode : std::uint8_t {
    RegretMatching,   // output = regrets; positive part, else best action
    Softmax,          // output = logits of the average strategy
    Argmax,           // output = regrets; always the highest one (best responses
                      // against fixed opponents should be deterministic)
};

// Strategy given by a network over the legal actions.
class NetPolicy final : public PolicyView {
public:
    NetPolicy(std::shared_ptr<const MLP> net, NetMode mode) : net_(std::move(net)), mode_(mode) {}
    void probs(const RoundState& rs, int player, const ActionList& legal,
               double* out) const override;
private:
    std::shared_ptr<const MLP> net_;
    NetMode mode_;
};

// Another policy evaluated once at every infoset of a round (one parallel
// pass over all deals and all actions), then served lock-free from a table.
// Exact evaluation queries the same infosets hundreds of times, so this is
// far cheaper than re-running a network. Tiny rounds only (enumerates deals).
class MaterializedPolicy final : public PolicyView {
public:
    MaterializedPolicy(const PolicyView& inner, int round, int threads = 0);
    void probs(const RoundState& rs, int player, const ActionList& legal,
               double* out) const override;
    std::size_t size() const { return map_.size(); }
private:
    const PolicyView& inner_;
    std::unordered_map<InfoKey, std::vector<double>> map_;
};

struct GenConfig {
    int           round   = 1;
    Utility       utility = Utility::Relative;
    int           count   = 10000;   // games (values) or trajectories per traverser (regrets)
    std::uint64_t seed    = 1;
    int           threads = 0;
    // >= 0: only this player is traversed / explored (best-response
    // training against fixed opponents); -1: all players.
    int           onlyPlayer = -1;
    // Fraction of average-strategy samples kept (they outnumber regret
    // samples ~3:1; thinning keeps files small when generating lots of data).
    double        policyKeep = 1.0;
};

// Player `learner` follows `mine`, everybody else follows `others`.
class MixedPolicy final : public PolicyView {
public:
    MixedPolicy(const PolicyView& mine, const PolicyView& others, int learner)
        : mine_(mine), others_(others), learner_(learner) {}
    void probs(const RoundState& rs, int player, const ActionList& legal,
               double* out) const override {
        (player == learner_ ? mine_ : others_).probs(rs, player, legal, out);
    }
private:
    const PolicyView& mine_;
    const PolicyView& others_;
    int learner_;
};

// Mean utility of `player` (and its standard error) when everybody follows
// `sigma`, estimated from cfg.count sampled deals.
struct MatchResult { double mean = 0.0, stderr_ = 0.0; };
MatchResult playMatch(const PolicyView& sigma, int player, const GenConfig& cfg);

struct ValueSamples {
    std::vector<std::uint8_t> x;   // n x HIST_DIM
    std::vector<float>        y;   // n
    std::size_t size() const { return y.size(); }
};

struct RegretSamples {
    std::vector<std::uint8_t> x;       // n x INFO_DIM
    std::vector<std::uint8_t> mask;    // n x ACT_DIM
    std::vector<float>        target;  // n x ACT_DIM (regrets; 0 where illegal)
    std::size_t size() const { return mask.size() / ACT_DIM; }
};

struct PolicySamples {
    std::vector<std::uint8_t> x;       // n x INFO_DIM
    std::vector<std::uint8_t> mask;    // n x ACT_DIM
    std::vector<float>        target;  // n x ACT_DIM (sigma)
    std::size_t size() const { return mask.size() / ACT_DIM; }
};

// On-policy value targets for the states the regret estimator will query
// (children of a traverser's uniformly sampled actions, continued under sigma).
ValueSamples generateValueSamples(const PolicyView& sigma, const GenConfig& cfg);

// Regret samples for every traverser and average-strategy samples.
// If `value` is null, action values are computed exactly by a full-width
// expectation under sigma (only feasible for tiny rounds; a diagnostic that
// removes value-net error).
void generateRegretSamples(const PolicyView& sigma, const MLP* value, const GenConfig& cfg,
                           RegretSamples& regrets, PolicySamples& policy);

// Minimal .npy writer (little-endian, C order).
void writeNpy(const std::string& path, const std::vector<std::uint8_t>& data,
              const std::vector<std::size_t>& shape);
void writeNpy(const std::string& path, const std::vector<float>& data,
              const std::vector<std::size_t>& shape);

} // namespace sk::solver
