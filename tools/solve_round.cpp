// solve_round — exact equilibrium solver for a single Skull King round.
//
// Runs full-width Discounted CFR over every deal of the round and reports
// exact NashConv (sum over players of best-response gain) as it converges.
// Feasible for round 1 with 4 players.
//
//   sk_solve_round --round 1 --iters 200 --eval-every 20 --out round1.csv

#include "sk/solver/best_response.hpp"
#include "sk/solver/cfr.hpp"
#include "sk/solver/deals.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

using namespace sk;
using namespace sk::solver;

namespace {

struct Args {
    int     round     = 1;
    int     iters     = 200;
    int     evalEvery = 20;
    int     threads   = 0;
    Utility utility   = Utility::Relative;
    std::string out   = "";
    bool    dumpAll   = false;
};

void usage() {
    std::puts(
        "usage: sk_solve_round [--round R] [--iters N] [--eval-every K] [--threads T]\n"
        "                      [--utility relative|absolute] [--out FILE.csv] [--dump-all]");
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--round"))      a.round = std::atoi(next());
        else if (!std::strcmp(argv[i], "--iters"))      a.iters = std::atoi(next());
        else if (!std::strcmp(argv[i], "--eval-every")) a.evalEvery = std::atoi(next());
        else if (!std::strcmp(argv[i], "--threads"))    a.threads = std::atoi(next());
        else if (!std::strcmp(argv[i], "--out"))        a.out = next();
        else if (!std::strcmp(argv[i], "--dump-all"))   a.dumpAll = true;
        else if (!std::strcmp(argv[i], "--utility")) {
            const std::string u = next();
            if (u == "relative")      a.utility = Utility::Relative;
            else if (u == "absolute") a.utility = Utility::Absolute;
            else { usage(); std::exit(2); }
        } else { usage(); std::exit(2); }
    }
    return a;
}

double seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void printReport(int iter, std::size_t infosets, double secs, const ExploitabilityReport& r) {
    std::printf("iter %5d | infosets %9zu | %7.1fs | NashConv %9.4f | gain", iter, infosets, secs,
                r.nashConv);
    for (int p = 0; p < N_PLAYERS; ++p) std::printf(" %7.4f", r.gain[p]);
    std::printf(" | value");
    for (int p = 0; p < N_PLAYERS; ++p) std::printf(" %+8.3f", r.value[p]);
    std::printf("\n");
    std::fflush(stdout);
}

// Actions of an infoset in the same order the solver used.
std::vector<std::string> actionNamesForKey(const InfoKey& key, int nA) {
    std::vector<std::string> names;
    const bool bidding = key[2] == 0;
    if (bidding) {
        for (int b = 0; b < nA; ++b) names.push_back("bid " + std::to_string(b));
        return names;
    }
    // Pending Tigress mode: the solver lists Pirate first, then Escape.
    if (!key.empty() && static_cast<std::uint8_t>(key.back()) == KIND_TIGRESS && nA == 2) {
        return {"Tigress=Pirate", "Tigress=Escape"};
    }
    for (int a = 0; a < nA; ++a) names.push_back("a" + std::to_string(a));
    return names;
}

void printRound1Chart(const InfosetTable& table) {
    std::printf("\nRound 1 equilibrium bidding: P(bid 1) by seat (seat 1 leads the trick)\n");
    std::printf("%-10s", "card");
    for (int seat = 0; seat < N_PLAYERS; ++seat) std::printf("  seat %d", seat + 1);
    std::printf("\n");

    auto row = [&](Kind k, const std::string& label) {
        std::printf("%-10s", label.c_str());
        for (int seat = 0; seat < N_PLAYERS; ++seat) {
            CardSet hand;
            hand.add(firstCardOfKind(k));
            InfoKey key;
            biddingKey(1, seat, hand, key);
            double pr[2] = {0.5, 0.5};
            if (const InfoNode* n = table.find(key)) n->averageStrategy(pr);
            std::printf("  %6.3f", pr[1]);
        }
        std::printf("\n");
    };
    for (int v = 1; v <= 14; ++v)
        row(static_cast<Kind>(makeColored(Suit::Yellow, v)), "Yellow " + std::to_string(v));
    for (int v = 1; v <= 14; ++v)
        row(static_cast<Kind>(makeColored(Suit::Black, v)), "Black " + std::to_string(v));
    row(KIND_ESCAPE, "Escape");
    row(KIND_MERMAID, "Mermaid");
    row(KIND_PIRATE, "Pirate");
    row(KIND_TIGRESS, "Tigress");
    row(KIND_SK, "SkullKing");

    // Green / Purple must equal Yellow by suit symmetry — a free sanity check.
    double maxDiff = 0.0;
    for (int seat = 0; seat < N_PLAYERS; ++seat) {
        for (int v = 1; v <= 14; ++v) {
            double ref[2] = {0, 0};
            for (Suit s : {Suit::Yellow, Suit::Green, Suit::Purple}) {
                CardSet hand;
                hand.add(makeColored(s, v));
                InfoKey key;
                biddingKey(1, seat, hand, key);
                double pr[2] = {0.5, 0.5};
                if (const InfoNode* n = table.find(key)) n->averageStrategy(pr);
                if (s == Suit::Yellow) { ref[0] = pr[0]; ref[1] = pr[1]; }
                else maxDiff = std::max(maxDiff, std::abs(pr[1] - ref[1]));
            }
        }
    }
    std::printf("suit-symmetry check: max |P_green/purple - P_yellow| = %.2e\n", maxDiff);

    // Tigress declaration, weighted by how often each infoset is reached.
    std::map<std::pair<int, int>, std::pair<double, double>> tig;  // (seat, own bid) -> (w, w*pPirate)
    table.forEach([&](const InfoKey& key, const InfoNode& n) {
        if (key[2] == 0 || n.nA != 2) return;
        if (static_cast<std::uint8_t>(key.back()) != KIND_TIGRESS) return;
        const int seat = key[1];
        const int handLen = key[3];
        const int ownBid = static_cast<std::int8_t>(key[4 + handLen + seat]);
        double w = n.stratSum[0] + n.stratSum[1];
        double pr[2];
        n.averageStrategy(pr);
        auto& e = tig[{seat, ownBid}];
        e.first += w;
        e.second += w * pr[0];
    });
    std::printf("\nTigress declared as Pirate (reach-weighted):\n");
    for (const auto& [k, e] : tig) {
        if (e.first <= 0) continue;
        // If the Tigress holder never makes this bid in equilibrium, these
        // infosets are off the equilibrium path and their strategy is arbitrary.
        CardSet hand;
        hand.add(TIGRESS);
        InfoKey key;
        biddingKey(1, k.first, hand, key);
        double bid[2] = {0.5, 0.5};
        if (const InfoNode* n = table.find(key)) n->averageStrategy(bid);
        if (bid[k.second] < 1e-6) {
            std::printf("  seat %d, own bid %d: off-path (never bid with Tigress)\n", k.first + 1,
                        k.second);
            continue;
        }
        std::printf("  seat %d, own bid %d: %.3f\n", k.first + 1, k.second, e.second / e.first);
    }
}

void writeCsv(const InfosetTable& table, const std::string& path, bool dumpAll) {
    std::ofstream f(path);
    f << "infoset,action,probability\n";
    std::size_t rows = 0;
    table.forEach([&](const InfoKey& key, const InfoNode& n) {
        if (!dumpAll && key[2] != 0) return;   // bidding infosets only by default
        std::vector<double> pr(n.nA);
        n.averageStrategy(pr.data());
        const auto names = actionNamesForKey(key, n.nA);
        for (int a = 0; a < n.nA; ++a) {
            f << '"' << describeKey(key) << "\"," << names[a] << ',' << pr[a] << '\n';
            ++rows;
        }
    });
    std::printf("wrote %zu rows to %s\n", rows, path.c_str());
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);

    std::printf("Skull King round %d, %d players, utility=%s, %zu kind-deals\n", args.round,
                N_PLAYERS, args.utility == Utility::Relative ? "relative" : "absolute",
                countDeals(args.round));

    CFRConfig cc;
    cc.round = args.round;
    cc.utility = args.utility;
    cc.threads = args.threads;
    FullWidthCFR cfr(cc);

    EvalConfig ec;
    ec.round = args.round;
    ec.utility = args.utility;
    ec.threads = args.threads;

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 1; it <= args.iters; ++it) {
        cfr.iterate();
        if (it == 1 || it % args.evalEvery == 0 || it == args.iters) {
            const auto rep = exploitability(AveragePolicy(cfr.table()), ec);
            printReport(it, cfr.table().size(), seconds(t0), rep);
        }
    }

    if (args.round == 1) printRound1Chart(cfr.table());
    if (!args.out.empty()) writeCsv(cfr.table(), args.out, args.dumpAll);
    return 0;
}
