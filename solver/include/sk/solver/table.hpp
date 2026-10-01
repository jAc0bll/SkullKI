#pragma once

#include "sk/solver/abstraction.hpp"
#include "sk/solver/round.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sk::solver {

// Regret / strategy storage for one information set.
struct InfoNode {
    std::uint8_t        nA = 0;
    std::vector<double> regret;     // cumulative (discounted) regret
    std::vector<double> stratSum;   // cumulative (discounted) average-strategy weight
    std::vector<double> current;    // strategy used during the current iteration

    explicit InfoNode(int n)
        : nA(static_cast<std::uint8_t>(n)), regret(n, 0.0), stratSum(n, 0.0),
          current(n, 1.0 / n) {}

    // Normalised average strategy (uniform if never reached).
    void averageStrategy(double* out) const;
};

// Read-only view of a strategy profile, as needed by best response and
// evaluation: the probabilities of `player`'s legal actions (in the order of
// `legal`) at a real decision point. Must depend only on what `player` knows.
class PolicyView {
public:
    virtual ~PolicyView() = default;
    virtual void probs(const RoundState& rs, int player, const ActionList& legal,
                       double* out) const = 0;
};

class UniformPolicy final : public PolicyView {
public:
    void probs(const RoundState&, int, const ActionList& legal, double* out) const override {
        for (int i = 0; i < legal.n; ++i) out[i] = 1.0 / legal.n;
    }
};

// Sharded hash table of InfoNodes. Node addresses are stable for the
// lifetime of the table. While `frozen()` is false, findOrCreate is
// thread-safe; once every infoset exists, freeze() makes lookups lock-free.
class InfosetTable {
public:
    static constexpr int N_SHARDS = 256;

    InfosetTable();

    InfoNode& findOrCreate(const InfoKey& key, int nA);
    const InfoNode* find(const InfoKey& key) const;

    void freeze() { frozen_ = true; }
    bool frozen() const { return frozen_; }

    std::size_t size() const;

    template <class F> void forEach(F&& f) {
        for (auto& sh : shards_) for (auto& [k, v] : sh->map) f(k, v);
    }
    template <class F> void forEach(F&& f) const {
        for (const auto& sh : shards_) for (const auto& [k, v] : sh->map) f(k, v);
    }

private:
    struct Shard {
        std::mutex mu;
        std::unordered_map<InfoKey, InfoNode> map;
    };
    std::array<std::unique_ptr<Shard>, N_SHARDS> shards_;
    std::atomic<bool> frozen_{false};

    static std::size_t shardOf(const InfoKey& key);
};

// The average strategy stored in a table built with abstraction `abs`.
// Not thread-safe against concurrent table writes.
class AveragePolicy final : public PolicyView {
public:
    explicit AveragePolicy(const InfosetTable& t, const Abstraction& abs = exactAbstraction())
        : t_(t), abs_(abs) {}
    void probs(const RoundState& rs, int player, const ActionList& legal,
               double* out) const override;
private:
    const InfosetTable& t_;
    const Abstraction&  abs_;
};

} // namespace sk::solver
