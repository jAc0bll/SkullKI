#include "sk/solver/best_response.hpp"
#include "sk/solver/parallel.hpp"

#include <unordered_map>

namespace sk::solver {

namespace {

constexpr int CAP = ActionList::CAPACITY;
using Values = std::array<double, N_PLAYERS>;

// Per-deal cache of each player's bidding-infoset strategy (bidding keys
// depend only on seat + hand, and each player is visited many times).
struct BidCache {
    std::array<bool, N_PLAYERS> have{};
    std::array<std::array<double, CAP>, N_PLAYERS> probs{};

    void reset() { have.fill(false); }

    const double* get(const RoundState& rs, int p, const ActionList& al, const PolicyView& pol) {
        if (!have[p]) {
            pol.probs(rs, p, al, probs[p].data());
            have[p] = true;
        }
        return probs[p].data();
    }
};

// ---------------------------------------------------------------------------
// Value of the profile.
// ---------------------------------------------------------------------------
struct ValueWorker {
    const EvalConfig* cfg = nullptr;
    const PolicyView* pol = nullptr;
    Values  value{};
    BidCache bids;

    void runDeal(const KindDeal& d) {
        const RoundState rs = makeRoundState(cfg->round, realizeDeal(d, cfg->round));
        bids.reset();
        Values v{};
        walk(rs, v);
        for (int p = 0; p < N_PLAYERS; ++p) value[p] += d.prob * v[p];
    }

    void walk(const RoundState& rs, Values& out) {
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
            walk(child, out);
            return;
        }
        std::array<double, CAP> local;
        const double* probs;
        if (rs.s.phase == Phase::Bidding) {
            probs = bids.get(rs, p, al, *pol);
        } else {
            pol->probs(rs, p, al, local.data());
            probs = local.data();
        }
        out.fill(0.0);
        for (int a = 0; a < al.n; ++a) {
            if (probs[a] <= 0.0) continue;
            RoundState child = rs;
            applyRound(child, al[a]);
            Values v{};
            walk(child, v);
            for (int q = 0; q < N_PLAYERS; ++q) out[q] += probs[a] * v[q];
        }
    }
};

// ---------------------------------------------------------------------------
// Best response for one player.
//
// Pass with target level L:
//   - BR player's decisions above L: branch on every action (we must reach
//     all level-L infosets); returned values are not needed.
//   - at level L: accumulate sum_h reach_{-i}(h) * Q(h, a) per infoset.
//   - below L: follow the best actions fixed by earlier (deeper) passes.
// Level = number of the BR player's real decisions (>1 legal action) so far;
// identical for all histories of an infoset by perfect recall.
// Final pass (L = -1) follows the BR everywhere and yields its value.
// ---------------------------------------------------------------------------
struct ActionValues {
    std::uint8_t nA = 0;
    std::array<double, CAP> q{};
};

struct BRWorker {
    const EvalConfig* cfg = nullptr;
    const PolicyView* pol = nullptr;
    int brPlayer = 0;
    int target   = 0;
    const std::unordered_map<InfoKey, std::uint8_t>* best = nullptr;

    std::unordered_map<InfoKey, ActionValues> acc;
    int    maxLevel = -1;
    double value    = 0.0;
    BidCache bids;

    void runDeal(const KindDeal& d) {
        const RoundState rs = makeRoundState(cfg->round, realizeDeal(d, cfg->round));
        bids.reset();
        value += d.prob * walk(rs, d.prob, 0);
    }

    // `reach` = chance * opponents' reach (counterfactual reach of brPlayer).
    double walk(const RoundState& rs, double reach, int level) {
        if (rs.terminal()) return roundUtilities(rs, cfg->utility)[brPlayer];

        const int p = rs.s.currentPlayer;
        ActionList al;
        legalKindActions(rs.s, al);
        if (al.n == 1) {
            RoundState child = rs;
            applyRound(child, al[0]);
            return walk(child, reach, level);
        }

        if (p != brPlayer) {
            std::array<double, CAP> local;
            const double* probs;
            if (rs.s.phase == Phase::Bidding) {
                probs = bids.get(rs, p, al, *pol);
            } else {
                pol->probs(rs, p, al, local.data());
                probs = local.data();
            }
            double v = 0.0;
            for (int a = 0; a < al.n; ++a) {
                if (probs[a] <= 0.0) continue;
                RoundState child = rs;
                applyRound(child, al[a]);
                v += probs[a] * walk(child, reach * probs[a], level);
            }
            return v;
        }

        if (level > maxLevel) maxLevel = level;
        InfoKey k;
        infosetKey(rs, p, k);

        if (level < target) {
            for (int a = 0; a < al.n; ++a) {
                RoundState child = rs;
                applyRound(child, al[a]);
                walk(child, reach, level + 1);
            }
            return 0.0;
        }
        if (level == target) {
            ActionValues& slot = acc[k];   // inserted even at zero reach
            slot.nA = static_cast<std::uint8_t>(al.n);
            for (int a = 0; a < al.n; ++a) {
                RoundState child = rs;
                applyRound(child, al[a]);
                slot.q[a] += reach * walk(child, reach, level + 1);
            }
            return 0.0;
        }
        const auto it = best->find(k);
        const int a = (it != best->end()) ? it->second : 0;
        RoundState child = rs;
        applyRound(child, al[a]);
        return walk(child, reach, level + 1);
    }
};

std::vector<BRWorker> runBRPass(const PolicyView& pol, const EvalConfig& cfg, int player,
                                int target,
                                const std::unordered_map<InfoKey, std::uint8_t>& best)
{
    std::vector<BRWorker> workers(resolveThreads(cfg.threads));
    for (auto& w : workers) {
        w.cfg = &cfg; w.pol = &pol; w.brPlayer = player; w.target = target; w.best = &best;
    }
    parallelOverDeals(cfg.round, workers, [](BRWorker& w, const KindDeal& d) { w.runDeal(d); });
    return workers;
}

} // namespace

std::array<double, N_PLAYERS> policyValue(const PolicyView& policy, const EvalConfig& cfg) {
    std::vector<ValueWorker> workers(resolveThreads(cfg.threads));
    for (auto& w : workers) { w.cfg = &cfg; w.pol = &policy; }
    parallelOverDeals(cfg.round, workers, [](ValueWorker& w, const KindDeal& d) { w.runDeal(d); });
    Values v{};
    for (const auto& w : workers)
        for (int p = 0; p < N_PLAYERS; ++p) v[p] += w.value[p];
    return v;
}

double bestResponseValue(const PolicyView& policy, int player, const EvalConfig& cfg) {
    std::unordered_map<InfoKey, std::uint8_t> best;

    // Discovery: how deep do the BR player's decisions go?
    int maxLevel = -1;
    for (const auto& w : runBRPass(policy, cfg, player, /*target=*/1 << 20, best))
        maxLevel = std::max(maxLevel, w.maxLevel);

    for (int level = maxLevel; level >= 0; --level) {
        auto workers = runBRPass(policy, cfg, player, level, best);
        std::unordered_map<InfoKey, ActionValues> total;
        for (auto& w : workers) {
            for (auto& [k, av] : w.acc) {
                ActionValues& t = total[k];
                t.nA = av.nA;
                for (int a = 0; a < av.nA; ++a) t.q[a] += av.q[a];
            }
        }
        for (const auto& [k, av] : total) {
            int arg = 0;
            for (int a = 1; a < av.nA; ++a) if (av.q[a] > av.q[arg]) arg = a;
            best[k] = static_cast<std::uint8_t>(arg);
        }
    }

    double v = 0.0;
    for (const auto& w : runBRPass(policy, cfg, player, /*target=*/-1, best)) v += w.value;
    return v;
}

ExploitabilityReport exploitability(const PolicyView& policy, const EvalConfig& cfg) {
    ExploitabilityReport r;
    r.value = policyValue(policy, cfg);
    for (int p = 0; p < N_PLAYERS; ++p) {
        r.brValue[p] = bestResponseValue(policy, p, cfg);
        r.gain[p]    = r.brValue[p] - r.value[p];
        r.nashConv  += r.gain[p];
    }
    return r;
}

} // namespace sk::solver
