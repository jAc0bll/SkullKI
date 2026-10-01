#include "sk/solver/deep.hpp"
#include "sk/solver/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace sk::solver {

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------
void NetPolicy::probs(const RoundState& rs, int player, const ActionList& legal,
                      double* out) const
{
    std::uint8_t x[INFO_DIM];
    float o[ACT_DIM];
    encodeInfoset(rs, player, x);
    net_->forward(x, o);

    if (mode_ == NetMode::RegretMatching) {
        double pos = 0.0;
        int best = 0;
        for (int a = 0; a < legal.n; ++a) {
            const float r = o[actionIndex(legal[a])];
            pos += r > 0.0f ? r : 0.0;
            if (r > o[actionIndex(legal[best])]) best = a;
        }
        for (int a = 0; a < legal.n; ++a) {
            const float r = o[actionIndex(legal[a])];
            // No positive regret: play the highest-regret action (Deep CFR).
            out[a] = pos > 0.0 ? (r > 0.0f ? r / pos : 0.0) : (a == best ? 1.0 : 0.0);
        }
        return;
    }
    float mx = -1e30f;
    for (int a = 0; a < legal.n; ++a) mx = std::max(mx, o[actionIndex(legal[a])]);
    double sum = 0.0;
    for (int a = 0; a < legal.n; ++a) {
        out[a] = std::exp(static_cast<double>(o[actionIndex(legal[a])] - mx));
        sum += out[a];
    }
    for (int a = 0; a < legal.n; ++a) out[a] /= sum;
}

namespace {

// Visits every decision node of a deal (all actions of all players) and
// records the policy at each infoset not seen yet by this worker.
struct MaterializeWorker {
    const PolicyView* inner = nullptr;
    int round = 1;
    std::unordered_map<InfoKey, std::vector<double>> map;

    void runDeal(const KindDeal& d) { walk(makeRoundState(round, realizeDeal(d, round))); }

    void walk(const RoundState& rs) {
        if (rs.terminal()) return;
        ActionList al;
        legalKindActions(rs.s, al);
        if (al.n > 1) {
            InfoKey key;
            infosetKey(rs, rs.s.currentPlayer, key);
            if (map.find(key) == map.end()) {
                std::vector<double> pr(al.n);
                inner->probs(rs, rs.s.currentPlayer, al, pr.data());
                map.emplace(std::move(key), std::move(pr));
            }
        }
        for (int a = 0; a < al.n; ++a) {
            RoundState child = rs;
            applyRound(child, al[a]);
            walk(child);
        }
    }
};

} // namespace

MaterializedPolicy::MaterializedPolicy(const PolicyView& inner, int round, int threads)
    : inner_(inner)
{
    std::vector<MaterializeWorker> workers(resolveThreads(threads));
    for (auto& w : workers) { w.inner = &inner; w.round = round; }
    parallelOverDeals(round, workers, [](MaterializeWorker& w, const KindDeal& d) { w.runDeal(d); });
    for (auto& w : workers) {
        for (auto& [k, v] : w.map) map_.emplace(k, std::move(v));
        w.map.clear();
    }
}

void MaterializedPolicy::probs(const RoundState& rs, int player, const ActionList& legal,
                               double* out) const
{
    InfoKey key;
    infosetKey(rs, player, key);
    const auto it = map_.find(key);
    if (it != map_.end()) {
        std::copy(it->second.begin(), it->second.end(), out);
        return;
    }
    inner_.probs(rs, player, legal, out);   // not reached for the round it was built for
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------
namespace {

std::array<CardSet, N_PLAYERS> sampleDeal(int round, std::mt19937_64& rng) {
    std::array<Card, N_CARDS> deck;
    std::iota(deck.begin(), deck.end(), Card{0});
    const int need = N_PLAYERS * round;
    for (int i = 0; i < need; ++i) {
        std::uniform_int_distribution<int> d(i, N_CARDS - 1);
        std::swap(deck[i], deck[d(rng)]);
    }
    std::array<CardSet, N_PLAYERS> hands{};
    for (int i = 0; i < need; ++i) hands[i / round].add(deck[i]);
    return hands;
}

int sampleIndex(const double* p, int n, std::mt19937_64& rng) {
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    for (int a = 0; a < n; ++a) {
        r -= p[a];
        if (r < 0.0) return a;
    }
    return n - 1;
}

int uniformIndex(int n, std::mt19937_64& rng) {
    return std::uniform_int_distribution<int>(0, n - 1)(rng);
}

// Expected utility of `who` from `rs` when everybody follows sigma.
double exactValue(const RoundState& rs, int who, const PolicyView& sigma, Utility u) {
    if (rs.terminal()) return roundUtilities(rs, u)[who];
    ActionList al;
    legalKindActions(rs.s, al);
    std::array<double, ActionList::CAPACITY> pr;
    if (al.n == 1) pr[0] = 1.0;
    else sigma.probs(rs, rs.s.currentPlayer, al, pr.data());
    double v = 0.0;
    for (int a = 0; a < al.n; ++a) {
        if (pr[a] <= 0.0) continue;
        RoundState child = rs;
        applyRound(child, al[a]);
        v += pr[a] * exactValue(child, who, sigma, u);
    }
    return v;
}

template <class Fn>
void runThreads(int nThreads, Fn&& fn) {
    std::vector<std::thread> ts;
    for (int t = 0; t < nThreads; ++t) ts.emplace_back([&, t] { fn(t); });
    for (auto& th : ts) th.join();
}

} // namespace

// ---------------------------------------------------------------------------
// Value samples
// ---------------------------------------------------------------------------
ValueSamples generateValueSamples(const PolicyView& sigma, const GenConfig& cfg) {
    const int nThreads = resolveThreads(cfg.threads);
    std::vector<ValueSamples> parts(nThreads);

    runThreads(nThreads, [&](int tid) {
        std::mt19937_64 rng(cfg.seed * 0x9E3779B97F4A7C15ull + 7919ull * tid + 1);
        ValueSamples& out = parts[tid];
        const int games = cfg.count / nThreads + (tid < cfg.count % nThreads ? 1 : 0);
        std::uint8_t x[HIST_DIM];
        std::array<double, ActionList::CAPACITY> pr;

        for (int g = 0; g < games; ++g) {
            RoundState rs = makeRoundState(cfg.round, sampleDeal(cfg.round, rng));
            // Traverser i explores uniformly for its first K+1 decisions, then
            // everybody follows sigma; states from then on get on-policy targets.
            const int i = cfg.onlyPlayer >= 0 ? cfg.onlyPlayer : uniformIndex(N_PLAYERS, rng);
            const int K = uniformIndex(cfg.round + 2, rng);
            int decisions = 0;
            bool recording = false;
            const std::size_t first = out.y.size();

            while (!rs.terminal()) {
                if (recording) {
                    encodeHistory(rs, i, x);
                    out.x.insert(out.x.end(), x, x + HIST_DIM);
                    out.y.push_back(0.0f);
                }
                ActionList al;
                legalKindActions(rs.s, al);
                const int p = rs.s.currentPlayer;
                int a = 0;
                if (al.n > 1) {
                    if (p == i && !recording) {
                        a = uniformIndex(al.n, rng);
                        if (++decisions > K) recording = true;
                    } else {
                        sigma.probs(rs, p, al, pr.data());
                        a = sampleIndex(pr.data(), al.n, rng);
                    }
                }
                applyRound(rs, al[a]);
            }
            const float u = static_cast<float>(roundUtilities(rs, cfg.utility)[i]);
            for (std::size_t k = first; k < out.y.size(); ++k) out.y[k] = u;
        }
    });

    ValueSamples all;
    for (auto& p : parts) {
        all.x.insert(all.x.end(), p.x.begin(), p.x.end());
        all.y.insert(all.y.end(), p.y.begin(), p.y.end());
    }
    return all;
}

// ---------------------------------------------------------------------------
// Regret + average-strategy samples
// ---------------------------------------------------------------------------
void generateRegretSamples(const PolicyView& sigma, const MLP* value, const GenConfig& cfg,
                           RegretSamples& regrets, PolicySamples& policy)
{
    const int nThreads = resolveThreads(cfg.threads);
    std::vector<RegretSamples> rparts(nThreads);
    std::vector<PolicySamples> pparts(nThreads);

    runThreads(nThreads, [&](int tid) {
        std::mt19937_64 rng(cfg.seed * 0xD1B54A32D192ED03ull + 104729ull * tid + 3);
        RegretSamples& rout = rparts[tid];
        PolicySamples& pout = pparts[tid];
        const int trajs = cfg.count / nThreads + (tid < cfg.count % nThreads ? 1 : 0);
        std::uint8_t x[INFO_DIM], h[HIST_DIM], mask[ACT_DIM];
        std::array<double, ActionList::CAPACITY> pr;
        float target[ACT_DIM];
        float v;

        for (int tr = 0; tr < trajs; ++tr) {
            for (int i = 0; i < N_PLAYERS; ++i) {
                if (cfg.onlyPlayer >= 0 && i != cfg.onlyPlayer) continue;
                RoundState rs = makeRoundState(cfg.round, sampleDeal(cfg.round, rng));
                while (!rs.terminal()) {
                    ActionList al;
                    legalKindActions(rs.s, al);
                    if (al.n == 1) { applyRound(rs, al[0]); continue; }
                    const int p = rs.s.currentPlayer;
                    sigma.probs(rs, p, al, pr.data());
                    encodeInfoset(rs, p, x);
                    legalMask(al, mask);

                    if (p == i) {
                        std::array<double, ActionList::CAPACITY> q{};
                        double ev = 0.0;
                        for (int a = 0; a < al.n; ++a) {
                            RoundState child = rs;
                            applyRound(child, al[a]);
                            if (child.terminal()) {
                                q[a] = roundUtilities(child, cfg.utility)[i];
                            } else if (value) {
                                encodeHistory(child, i, h);
                                value->forward(h, &v);
                                q[a] = v;
                            } else {
                                q[a] = exactValue(child, i, sigma, cfg.utility);
                            }
                            ev += pr[a] * q[a];
                        }
                        std::fill(target, target + ACT_DIM, 0.0f);
                        for (int a = 0; a < al.n; ++a)
                            target[actionIndex(al[a])] = static_cast<float>(q[a] - ev);
                        rout.x.insert(rout.x.end(), x, x + INFO_DIM);
                        rout.mask.insert(rout.mask.end(), mask, mask + ACT_DIM);
                        rout.target.insert(rout.target.end(), target, target + ACT_DIM);
                        applyRound(rs, al[uniformIndex(al.n, rng)]);
                    } else {
                        const bool keep = cfg.policyKeep >= 1.0 ||
                            std::uniform_real_distribution<double>(0.0, 1.0)(rng) < cfg.policyKeep;
                        if (keep) {
                            std::fill(target, target + ACT_DIM, 0.0f);
                            for (int a = 0; a < al.n; ++a)
                                target[actionIndex(al[a])] = static_cast<float>(pr[a]);
                            pout.x.insert(pout.x.end(), x, x + INFO_DIM);
                            pout.mask.insert(pout.mask.end(), mask, mask + ACT_DIM);
                            pout.target.insert(pout.target.end(), target, target + ACT_DIM);
                        }
                        applyRound(rs, al[sampleIndex(pr.data(), al.n, rng)]);
                    }
                }
            }
        }
    });

    for (auto& r : rparts) {
        regrets.x.insert(regrets.x.end(), r.x.begin(), r.x.end());
        regrets.mask.insert(regrets.mask.end(), r.mask.begin(), r.mask.end());
        regrets.target.insert(regrets.target.end(), r.target.begin(), r.target.end());
    }
    for (auto& p : pparts) {
        policy.x.insert(policy.x.end(), p.x.begin(), p.x.end());
        policy.mask.insert(policy.mask.end(), p.mask.begin(), p.mask.end());
        policy.target.insert(policy.target.end(), p.target.begin(), p.target.end());
    }
}

// ---------------------------------------------------------------------------
// Head-to-head simulation
// ---------------------------------------------------------------------------
MatchResult playMatch(const PolicyView& sigma, int player, const GenConfig& cfg) {
    const int nThreads = resolveThreads(cfg.threads);
    std::vector<double> sum(nThreads), sumSq(nThreads);
    runThreads(nThreads, [&](int tid) {
        std::mt19937_64 rng(cfg.seed * 0xA24BAED4963EE407ull + 31337ull * tid + 5);
        const int games = cfg.count / nThreads + (tid < cfg.count % nThreads ? 1 : 0);
        std::array<double, ActionList::CAPACITY> pr;
        for (int g = 0; g < games; ++g) {
            RoundState rs = makeRoundState(cfg.round, sampleDeal(cfg.round, rng));
            while (!rs.terminal()) {
                ActionList al;
                legalKindActions(rs.s, al);
                int a = 0;
                if (al.n > 1) {
                    sigma.probs(rs, rs.s.currentPlayer, al, pr.data());
                    a = sampleIndex(pr.data(), al.n, rng);
                }
                applyRound(rs, al[a]);
            }
            const double u = roundUtilities(rs, cfg.utility)[player];
            sum[tid] += u;
            sumSq[tid] += u * u;
        }
    });
    double s = 0.0, s2 = 0.0;
    for (int t = 0; t < nThreads; ++t) { s += sum[t]; s2 += sumSq[t]; }
    const double n = cfg.count;
    MatchResult r;
    r.mean = s / n;
    r.stderr_ = std::sqrt(std::max(0.0, s2 / n - r.mean * r.mean) / n);
    return r;
}

// ---------------------------------------------------------------------------
// .npy writer
// ---------------------------------------------------------------------------
namespace {

void writeNpyRaw(const std::string& path, const char* descr, const void* data, std::size_t bytes,
                 const std::vector<std::size_t>& shape)
{
    std::string dict = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) dict += ",";
        if (i + 1 < shape.size()) dict += " ";
    }
    dict += "), }";
    // Pad so magic(6)+version(2)+len(2)+header is a multiple of 64; end with '\n'.
    const std::size_t total = 10 + dict.size() + 1;
    dict.append((64 - total % 64) % 64, ' ');
    dict += '\n';

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    const std::uint16_t len = static_cast<std::uint16_t>(dict.size());
    f.write(reinterpret_cast<const char*>(&len), 2);
    f.write(dict.data(), static_cast<std::streamsize>(dict.size()));
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

} // namespace

void writeNpy(const std::string& path, const std::vector<std::uint8_t>& data,
              const std::vector<std::size_t>& shape)
{
    writeNpyRaw(path, "|u1", data.data(), data.size(), shape);
}

void writeNpy(const std::string& path, const std::vector<float>& data,
              const std::vector<std::size_t>& shape)
{
    writeNpyRaw(path, "<f4", data.data(), data.size() * sizeof(float), shape);
}

} // namespace sk::solver
