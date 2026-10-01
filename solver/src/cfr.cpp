#include "sk/solver/cfr.hpp"
#include "sk/solver/parallel.hpp"

#include <cmath>
#include <unordered_map>

namespace sk::solver {

namespace {

constexpr int CAP = ActionList::CAPACITY;
using Values = std::array<double, N_PLAYERS>;

// Per-thread accumulator of this iteration's regret / strategy deltas, so
// hot infosets (e.g. bidding) are not contended between threads.
struct Delta {
    std::array<double, CAP> dR{};
    std::array<double, CAP> dS{};
};

struct Worker {
    const CFRConfig* cfg   = nullptr;
    InfosetTable*    table = nullptr;
    std::unordered_map<InfoNode*, Delta> deltas;
    Values value{};   // sum over deals of chance * value under current profile

    // Bidding infosets depend only on (seat, hand): resolve once per deal.
    std::array<InfoNode*, N_PLAYERS> bidNode{};
    std::array<Delta*,    N_PLAYERS> bidDelta{};
    InfoKey key;

    void runDeal(const KindDeal& d) {
        const RoundState rs = makeRoundState(cfg->round, realizeDeal(d, cfg->round));
        bidNode.fill(nullptr);
        bidDelta.fill(nullptr);
        Values reach;
        reach.fill(1.0);
        Values out{};
        walk(rs, reach, d.prob, out);
        for (int p = 0; p < N_PLAYERS; ++p) value[p] += d.prob * out[p];
    }

    void walk(const RoundState& rs, const Values& reach, double chance, Values& out) {
        if (rs.terminal()) {
            out = roundUtilities(rs, cfg->utility);
            return;
        }

        const int p = rs.s.currentPlayer;
        ActionList al;
        legalKindActions(rs.s, al);

        if (al.n == 1) {
            RoundState child = rs;
            applyRound(child, al[0]);
            walk(child, reach, chance, out);
            return;
        }

        // A subtree where two or more players have zero reach cannot affect
        // any regret (see DCFR derivation): skip it.
        int zeros = 0;
        for (double r : reach) zeros += (r == 0.0);
        if (zeros >= 2) {
            out.fill(0.0);
            return;
        }

        InfoNode* node;
        Delta*    delta;
        if (rs.s.phase == Phase::Bidding) {
            if (!bidNode[p]) {
                biddingKey(rs.round, p, rs.s.hands[p], key);
                bidNode[p]  = &table->findOrCreate(key, al.n);
                bidDelta[p] = &deltas[bidNode[p]];
            }
            node  = bidNode[p];
            delta = bidDelta[p];
        } else {
            infosetKey(rs, p, key);
            node  = &table->findOrCreate(key, al.n);
            delta = &deltas[node];
        }

        const double* sigma = node->current.data();
        std::array<Values, CAP> vals;
        out.fill(0.0);
        for (int a = 0; a < al.n; ++a) {
            RoundState child = rs;
            applyRound(child, al[a]);
            Values r2 = reach;
            r2[p] *= sigma[a];
            walk(child, r2, chance, vals[a]);
            for (int q = 0; q < N_PLAYERS; ++q) out[q] += sigma[a] * vals[a][q];
        }

        double cf = chance;
        for (int q = 0; q < N_PLAYERS; ++q) if (q != p) cf *= reach[q];
        const double own = chance * reach[p];
        for (int a = 0; a < al.n; ++a) {
            delta->dR[a] += cf * (vals[a][p] - out[p]);
            delta->dS[a] += own * sigma[a];
        }
    }
};

void regretMatch(const std::vector<double>& regret, std::vector<double>& out) {
    double pos = 0.0;
    for (double r : regret) pos += (r > 0.0 ? r : 0.0);
    const std::size_t n = regret.size();
    for (std::size_t a = 0; a < n; ++a) {
        out[a] = (pos > 0.0) ? (regret[a] > 0.0 ? regret[a] / pos : 0.0) : 1.0 / n;
    }
}

} // namespace

FullWidthCFR::FullWidthCFR(CFRConfig cfg) : cfg_(cfg) {}

void FullWidthCFR::iterate() {
    ++t_;
    const int nThreads = resolveThreads(cfg_.threads);
    std::vector<Worker> workers(nThreads);
    for (auto& w : workers) { w.cfg = &cfg_; w.table = &table_; }

    parallelOverDeals(cfg_.round, workers, [](Worker& w, const KindDeal& d) { w.runDeal(d); });

    // Full-width traversal visits every infoset, so after the first
    // iteration the table is complete and lookups can go lock-free.
    table_.freeze();

    const double t = static_cast<double>(t_);
    const double posDisc = std::pow(t, cfg_.alpha) / (std::pow(t, cfg_.alpha) + 1.0);
    const double negDisc = std::pow(t, cfg_.beta)  / (std::pow(t, cfg_.beta)  + 1.0);
    const double avgDisc = std::pow(t / (t + 1.0), cfg_.gamma);

    // S <- S * avgDisc + dS ;  R <- disc(R + dR)
    table_.forEach([&](const InfoKey&, InfoNode& n) {
        for (int a = 0; a < n.nA; ++a) n.stratSum[a] *= avgDisc;
    });
    for (auto& w : workers) {
        for (auto& [node, d] : w.deltas) {
            for (int a = 0; a < node->nA; ++a) {
                node->regret[a]   += d.dR[a];
                node->stratSum[a] += d.dS[a];
            }
        }
    }
    table_.forEach([&](const InfoKey&, InfoNode& n) {
        for (int a = 0; a < n.nA; ++a) {
            double& r = n.regret[a];
            r *= (r > 0.0) ? posDisc : negDisc;
        }
        regretMatch(n.regret, n.current);
    });

    lastValue_.fill(0.0);
    for (const auto& w : workers)
        for (int p = 0; p < N_PLAYERS; ++p) lastValue_[p] += w.value[p];
}

} // namespace sk::solver
