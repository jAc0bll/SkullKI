#include "sk/solver/mccfr.hpp"
#include "sk/solver/parallel.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <thread>

namespace sk::solver {

namespace {

constexpr int CAP = ActionList::CAPACITY;

struct Worker {
    const MCCFRConfig* cfg   = nullptr;
    InfosetTable*      table = nullptr;
    DeltaMap           deltas;
    std::mt19937_64    rng;
    InfosetView        view;
    std::uint64_t      nodes = 0;   // walk() calls, for cost measurements
    bool               prune = false;

    std::array<CardSet, N_PLAYERS> sampleDeal() {
        std::array<Card, N_CARDS> deck;
        std::iota(deck.begin(), deck.end(), Card{0});
        const int need = N_PLAYERS * cfg->round;
        // Partial Fisher-Yates: only the dealt prefix needs to be shuffled.
        for (int i = 0; i < need; ++i) {
            std::uniform_int_distribution<int> d(i, N_CARDS - 1);
            std::swap(deck[i], deck[d(rng)]);
        }
        std::array<CardSet, N_PLAYERS> hands{};
        for (int i = 0; i < need; ++i) hands[i / cfg->round].add(deck[i]);
        return hands;
    }

    void runDeal() {
        const RoundState rs = makeRoundState(cfg->round, sampleDeal());
        for (int i = 0; i < N_PLAYERS; ++i) walk(rs, i);
    }

    // Returns the traverser's sampled counterfactual value.
    double walk(const RoundState& rs, int traverser) {
        ++nodes;
        if (rs.terminal()) return roundUtilities(rs, cfg->utility)[traverser];

        const int p = rs.s.currentPlayer;
        ActionList al;
        legalKindActions(rs.s, al);
        if (al.n == 1) {
            RoundState child = rs;
            applyRound(child, al[0]);
            return walk(child, traverser);
        }

        cfg->abstraction->view(rs, p, al, view);
        if (view.nSlots == 1) {   // all legal actions equivalent: no decision
            RoundState child = rs;
            applyRound(child, al[0]);
            return walk(child, traverser);
        }
        InfoNode& node = table->findOrCreate(view.key, view.nSlots);
        Delta& delta = deltas[&node];
        // The solver works on slots; each slot is played through one
        // representative real action (merged actions are equivalent under
        // the abstraction). `view` is reused by the recursion, so copy.
        const int nS = view.nSlots;
        std::array<int, CAP> rep;
        rep.fill(-1);
        for (int a = 0; a < al.n; ++a)
            if (rep[view.slot[a]] < 0) rep[view.slot[a]] = a;
        const double* sigma = node.current.data();
        std::uniform_real_distribution<double> u(0.0, 1.0);

        if (p == traverser) {
            std::array<double, CAP> v{};
            std::array<bool, CAP> explored{};
            double ev = 0.0;
            for (int k = 0; k < nS; ++k) {
                if (prune && sigma[k] == 0.0 && u(rng) < cfg->pruneProb) continue;
                explored[k] = true;
                RoundState child = rs;
                applyRound(child, al[rep[k]]);
                v[k] = walk(child, traverser);
                ev += sigma[k] * v[k];
            }
            for (int k = 0; k < nS; ++k)
                if (explored[k]) delta.dR[k] += v[k] - ev;
            return ev;
        }

        // Opponent node: stochastically-weighted average-strategy update,
        // then follow one sampled slot.
        for (int k = 0; k < nS; ++k) delta.dS[k] += sigma[k];
        double r = u(rng);
        int pickSlot = nS - 1;
        for (int k = 0; k < nS; ++k) {
            r -= sigma[k];
            if (r < 0.0) { pickSlot = k; break; }
        }
        const int pick = rep[pickSlot];
        RoundState child = rs;
        applyRound(child, al[pick]);
        return walk(child, traverser);
    }
};

} // namespace

ExternalSamplingMCCFR::ExternalSamplingMCCFR(MCCFRConfig cfg) : cfg_(cfg) {}

void ExternalSamplingMCCFR::runBatch() {
    ++batches_;
    const int nThreads = resolveThreads(cfg_.threads);
    std::vector<Worker> workers(nThreads);
    for (int w = 0; w < nThreads; ++w) {
        workers[w].cfg = &cfg_;
        workers[w].table = &table_;
        workers[w].prune = cfg_.pruneProb > 0.0 && batches_ > cfg_.pruneAfterBatch;
        // Distinct, reproducible stream per (batch, worker).
        workers[w].rng.seed(cfg_.seed * 0x9E3779B97F4A7C15ull + batches_ * 1315423911ull + w);
    }

    const int perWorker = (cfg_.dealsPerBatch + nThreads - 1) / nThreads;
    std::vector<std::thread> threads;
    for (int w = 0; w < nThreads; ++w) {
        threads.emplace_back([&, w] {
            for (int i = 0; i < perWorker; ++i) workers[w].runDeal();
        });
    }
    for (auto& t : threads) t.join();
    traversals_ += static_cast<std::uint64_t>(perWorker) * nThreads * N_PLAYERS;
    for (const auto& w : workers) nodes_ += w.nodes;

    std::vector<const DeltaMap*> deltas;
    for (const auto& w : workers) deltas.push_back(&w.deltas);
    applyDCFRUpdate(table_, deltas, batches_, cfg_.dcfr);
}

} // namespace sk::solver
