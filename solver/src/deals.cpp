#include "sk/solver/deals.hpp"

#include <cassert>

namespace sk::solver {

namespace {

double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

// Number of ordered deals of distinct card ids for this round.
double totalDealWeight(int round) {
    double w = 1.0;
    for (int p = 0; p < N_PLAYERS; ++p) w *= binom(N_CARDS - p * round, round);
    return w;
}

using Avail = std::array<int, N_KINDS>;

Avail fullDeck() {
    Avail a{};
    for (int k = 0; k < N_KINDS; ++k) a[k] = kindMultiplicity(static_cast<Kind>(k));
    return a;
}

// Enumerate hands for players [p, endP) as kind multisets.
// `w` accumulates prod C(avail_k, m_k) — the number of id-level deals per kind deal.
template <class F>
void enumPlayers(int p, int endP, int round, Avail& avail, KindDeal& cur, double w, F&& f);

template <class F>
void enumHand(int p, int endP, int round, int kind, int filled,
              Avail& avail, KindDeal& cur, double w, F&& f)
{
    if (filled == round) {
        enumPlayers(p + 1, endP, round, avail, cur, w, f);
        return;
    }
    if (kind >= N_KINDS) return;
    const int need = round - filled;
    const int maxM = std::min(avail[kind], need);
    // m = 0 first, then take 1..maxM copies of this kind.
    enumHand(p, endP, round, kind + 1, filled, avail, cur, w, f);
    for (int m = 1; m <= maxM; ++m) {
        for (int i = 0; i < m; ++i) cur.hand[p][filled + i] = static_cast<Kind>(kind);
        const int before = avail[kind];
        avail[kind] -= m;
        enumHand(p, endP, round, kind + 1, filled + m, avail, cur, w * binom(before, m), f);
        avail[kind] += m;
    }
}

template <class F>
void enumPlayers(int p, int endP, int round, Avail& avail, KindDeal& cur, double w, F&& f) {
    if (p == endP) {
        cur.prob = w;
        f(cur);
        return;
    }
    enumHand(p, endP, round, 0, 0, avail, cur, w, f);
}

Avail availAfter(const KindDeal& d, int players, int round) {
    Avail a = fullDeck();
    for (int p = 0; p < players; ++p)
        for (int i = 0; i < round; ++i) --a[d.hand[p][i]];
    return a;
}

} // namespace

std::array<CardSet, N_PLAYERS> realizeDeal(const KindDeal& d, int round) {
    std::array<CardSet, N_PLAYERS> hands{};
    std::array<int, N_KINDS> used{};
    for (int p = 0; p < N_PLAYERS; ++p) {
        for (int i = 0; i < round; ++i) {
            const Kind k = d.hand[p][i];
            assert(used[k] < kindMultiplicity(k));
            hands[p].add(static_cast<Card>(firstCardOfKind(k) + used[k]++));
        }
    }
    return hands;
}

std::vector<DealTask> makeDealTasks(int round, int prefixPlayers) {
    assert(prefixPlayers >= 0 && prefixPlayers <= N_PLAYERS);
    std::vector<DealTask> tasks;
    Avail avail = fullDeck();
    KindDeal cur;
    enumPlayers(0, prefixPlayers, round, avail, cur, 1.0, [&](const KindDeal& d) {
        DealTask t;
        t.partial = d;
        t.prefixPlayers = prefixPlayers;
        tasks.push_back(t);
    });
    return tasks;
}

void forEachDealInTask(const DealTask& task, int round,
                       const std::function<void(const KindDeal&)>& f)
{
    const double norm = 1.0 / totalDealWeight(round);
    Avail avail = availAfter(task.partial, task.prefixPlayers, round);
    KindDeal cur = task.partial;
    enumPlayers(task.prefixPlayers, N_PLAYERS, round, avail, cur, task.partial.prob,
                [&](KindDeal& d) {
                    d.prob *= norm;
                    f(d);
                });
}

std::size_t countDeals(int round) {
    std::size_t n = 0;
    Avail avail = fullDeck();
    KindDeal cur;
    enumPlayers(0, N_PLAYERS, round, avail, cur, 1.0, [&](const KindDeal&) { ++n; });
    return n;
}

} // namespace sk::solver
