// bench_traversal — how expensive is one external-sampling MCCFR traversal
// in each round?
//
// 1. Knuth's estimator: follow random paths, multiply the branching factor at
//    every node the traversal would fully expand (the traverser's decisions;
//    opponent nodes are sampled = factor 1). The average of the products is an
//    unbiased estimate of the number of nodes one traversal visits.
// 2. Measured throughput (nodes/s) of the real MCCFR for small rounds, used
//    to extrapolate the time per traversal in the large rounds.
//
//   sk_bench_traversal [--max-round 10] [--probes 20000] [--measure-up-to 4]

#include "sk/solver/mccfr.hpp"
#include "sk/solver/round.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>

using namespace sk;
using namespace sk::solver;

namespace {

std::array<CardSet, N_PLAYERS> randomDeal(int round, std::mt19937_64& rng) {
    std::array<Card, N_CARDS> deck;
    std::iota(deck.begin(), deck.end(), Card{0});
    std::shuffle(deck.begin(), deck.end(), rng);
    std::array<CardSet, N_PLAYERS> h{};
    for (int i = 0; i < N_PLAYERS * round; ++i) h[i / round].add(deck[i]);
    return h;
}

// One Knuth probe: estimated node count of an ES traversal for `traverser`.
double knuthProbe(int round, int traverser, std::mt19937_64& rng) {
    RoundState rs = makeRoundState(round, randomDeal(round, rng));
    double weight = 1.0;   // product of branching factors so far
    double total  = 1.0;   // root
    while (!rs.terminal()) {
        ActionList al;
        legalKindActions(rs.s, al);
        std::uniform_int_distribution<int> d(0, al.n - 1);
        const int pick = d(rng);
        if (rs.s.currentPlayer == traverser && al.n > 1) weight *= al.n;
        applyRound(rs, al[pick]);
        total += weight;
    }
    return total;
}

} // namespace

int main(int argc, char** argv) {
    int maxRound = 10, probes = 20000, measureUpTo = 4;
    for (int i = 1; i + 1 < argc; i += 2) {
        if      (!std::strcmp(argv[i], "--max-round"))     maxRound = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--probes"))        probes = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--measure-up-to")) measureUpTo = std::atoi(argv[i + 1]);
    }

    std::mt19937_64 rng(12345);
    std::printf("round | est. nodes per traversal | measured nodes/s (all threads) | est. time per traversal\n");

    double nodesPerSec = 0.0;
    for (int r = 1; r <= maxRound; ++r) {
        double sum = 0.0;
        for (int i = 0; i < probes; ++i) sum += knuthProbe(r, i % N_PLAYERS, rng);
        const double est = sum / probes;

        if (r <= measureUpTo) {
            MCCFRConfig c;
            c.round = r;
            c.dealsPerBatch = 4000;
            ExternalSamplingMCCFR m(c);
            const auto t0 = std::chrono::steady_clock::now();
            m.runBatch();
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            nodesPerSec = m.nodesVisited() / secs;
            std::printf("%5d | %24.3g | %30.3g | %20.3g s\n", r, est, nodesPerSec, est / nodesPerSec);
        } else {
            std::printf("%5d | %24.3g | %30s | %20.3g s (extrapolated)\n", r, est, "-", est / nodesPerSec);
        }
        std::fflush(stdout);
    }
    return 0;
}
