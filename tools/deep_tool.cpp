// sk_deep — C++ side of the neural CFR loop (train/deep_cfr.py drives it).
//
//   sk_deep gen-values  --round R [--policy regret.bin] --count N --out PREFIX [--seed S]
//   sk_deep gen-regrets --round R [--policy regret.bin] (--value value.bin | --exact-values)
//                       --count N --out PREFIX [--seed S]
//   sk_deep eval        --round 1 --policy net.bin --mode rm|softmax
//
// Without --policy the current strategy is uniform (first iteration).
// gen-values writes PREFIX_x.npy (uint8, n x HIST_DIM) and PREFIX_y.npy;
// gen-regrets writes PREFIX_rx/_rmask/_rtarget and PREFIX_px/_pmask/_ptarget.
// eval prints the exact NashConv (enumerates every deal; round 1 only).

#include "sk/solver/best_response.hpp"
#include "sk/solver/deep.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

using namespace sk;
using namespace sk::solver;

namespace {

struct Args {
    std::string cmd, policy, value, out, mode = "rm", oppPolicy, oppMode = "softmax";
    int learner = -1;
    std::string improveBase;
    double tau = 0.0;
    int round = 1, count = 10000, threads = 0;
    std::uint64_t seed = 1;
    bool exactValues = false;
    double policyKeep = 1.0;
};

[[noreturn]] void usage() {
    std::puts("usage: sk_deep gen-values|gen-regrets|eval [--round R] [--policy F] [--value F]\n"
              "               [--exact-values] [--count N] [--out PREFIX] [--seed S]\n"
              "               [--mode rm|softmax|argmax|improve] [--improve-base F --tau T]\n"
              "               [--threads T]");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    if (argc < 2) usage();
    Args a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (!std::strcmp(argv[i], "--round"))   a.round = std::atoi(next());
        else if (!std::strcmp(argv[i], "--policy"))  a.policy = next();
        else if (!std::strcmp(argv[i], "--value"))   a.value = next();
        else if (!std::strcmp(argv[i], "--out"))     a.out = next();
        else if (!std::strcmp(argv[i], "--mode"))    a.mode = next();
        else if (!std::strcmp(argv[i], "--count"))   a.count = std::atoi(next());
        else if (!std::strcmp(argv[i], "--threads")) a.threads = std::atoi(next());
        else if (!std::strcmp(argv[i], "--seed"))    a.seed = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(argv[i], "--exact-values")) a.exactValues = true;
        else if (!std::strcmp(argv[i], "--policy-keep")) a.policyKeep = std::atof(next());
        else if (!std::strcmp(argv[i], "--opp-policy")) a.oppPolicy = next();
        else if (!std::strcmp(argv[i], "--opp-mode"))   a.oppMode = next();
        else if (!std::strcmp(argv[i], "--learner"))    a.learner = std::atoi(next());
        else if (!std::strcmp(argv[i], "--improve-base")) a.improveBase = next();
        else if (!std::strcmp(argv[i], "--tau"))        a.tau = std::atof(next());
        else usage();
    }
    return a;
}

std::unique_ptr<PolicyView> loadPolicy(const std::string& path, const std::string& mode,
                                       const std::string& improveBase = "", double tau = 0.0) {
    if (path.empty() || path == "uniform") return std::make_unique<UniformPolicy>();
    auto net = std::make_shared<const MLP>(MLP::load(path));
    if (net->inputDim() != INFO_DIM || net->outputDim() != ACT_DIM)
        throw std::runtime_error("policy net has wrong shape: " + path);
    if (mode == "improve") {
        // `path` holds advantages relative to `improveBase` (softmax net).
        if (improveBase.empty()) throw std::runtime_error("--mode improve needs --improve-base");
        return std::make_unique<ImprovedPolicy>(loadPolicy(improveBase, "softmax"), net, tau);
    }
    const NetMode m = mode == "softmax" ? NetMode::Softmax
                    : mode == "argmax"  ? NetMode::Argmax
                                        : NetMode::RegretMatching;
    return std::make_unique<NetPolicy>(net, m);
}

double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();
    GenConfig g;
    g.round = a.round;
    g.count = a.count;
    g.seed = a.seed;
    g.threads = a.threads;
    g.policyKeep = a.policyKeep;

    try {
        // --learner i --opp-policy F: player i follows --policy, the others
        // the fixed --opp-policy (best-response training / matches).
        const auto own = loadPolicy(a.policy, a.mode, a.improveBase, a.tau);
        std::unique_ptr<PolicyView> opp;
        std::unique_ptr<PolicyView> mixed;
        const PolicyView* policy = own.get();
        if (a.learner >= 0) {
            opp = loadPolicy(a.oppPolicy, a.oppMode);
            mixed = std::make_unique<MixedPolicy>(*own, *opp, a.learner);
            policy = mixed.get();
            g.onlyPlayer = a.learner;
        }
        if (a.cmd == "gen-values") {
            if (a.out.empty()) usage();
            const ValueSamples v = generateValueSamples(*policy, g);
            writeNpy(a.out + "_x.npy", v.x, {v.size(), static_cast<std::size_t>(HIST_DIM)});
            writeNpy(a.out + "_y.npy", v.y, {v.size()});
            std::printf("value samples %zu (%.1fs)\n", v.size(), secondsSince(t0));
        } else if (a.cmd == "gen-regrets") {
            if (a.out.empty() || (a.value.empty() && !a.exactValues)) usage();
            std::unique_ptr<MLP> value;
            if (!a.exactValues) {
                value = std::make_unique<MLP>(MLP::load(a.value));
                if (value->inputDim() != HIST_DIM || value->outputDim() != 1)
                    throw std::runtime_error("value net has wrong shape");
            }
            RegretSamples r;
            PolicySamples p;
            generateRegretSamples(*policy, value.get(), g, r, p);
            const auto I = static_cast<std::size_t>(INFO_DIM), A = static_cast<std::size_t>(ACT_DIM);
            writeNpy(a.out + "_rx.npy", r.x, {r.size(), I});
            writeNpy(a.out + "_rmask.npy", r.mask, {r.size(), A});
            writeNpy(a.out + "_rtarget.npy", r.target, {r.size(), A});
            writeNpy(a.out + "_px.npy", p.x, {p.size(), I});
            writeNpy(a.out + "_pmask.npy", p.mask, {p.size(), A});
            writeNpy(a.out + "_ptarget.npy", p.target, {p.size(), A});
            std::printf("regret samples %zu, policy samples %zu (%.1fs)\n", r.size(), p.size(),
                        secondsSince(t0));
        } else if (a.cmd == "match") {
            // Mean utility of the learner seat with and without its policy.
            if (a.learner < 0) usage();
            const MatchResult m = playMatch(*policy, a.learner, g);
            std::printf("seat %d utility %+.4f +- %.4f over %d deals\n", a.learner + 1, m.mean,
                        m.stderr_, g.count);
        } else if (a.cmd == "chart") {
            // Round-1 bidding of the policy: P(bid 1) per seat and card kind,
            // one line per kind: "kind p1 p2 p3 p4".
            for (int k = 0; k < N_KINDS; ++k) {
                std::printf("%s", kindName(static_cast<Kind>(k)).c_str());
                for (int seat = 0; seat < N_PLAYERS; ++seat) {
                    std::array<CardSet, N_PLAYERS> hands{};
                    hands[seat].add(firstCardOfKind(static_cast<Kind>(k)));
                    RoundState rs = makeRoundState(1, hands);
                    rs.s.currentPlayer = static_cast<std::int8_t>(seat);
                    ActionList al;
                    legalKindActions(rs.s, al);
                    double pr[2];
                    policy->probs(rs, seat, al, pr);
                    std::printf(" %.3f", pr[1]);
                }
                std::printf("\n");
            }
        } else if (a.cmd == "eval") {
            if (a.round != 1) {
                std::fprintf(stderr, "exact evaluation only for round 1\n");
                return 2;
            }
            const MaterializedPolicy cached(*policy, a.round, a.threads);
            std::printf("materialized %zu infosets (%.1fs)\n", cached.size(), secondsSince(t0));
            EvalConfig e;
            e.round = a.round;
            e.threads = a.threads;
            const ExploitabilityReport r = exploitability(cached, e);
            std::printf("NashConv %.4f | gain", r.nashConv);
            for (int p = 0; p < N_PLAYERS; ++p) std::printf(" %.4f", r.gain[p]);
            std::printf(" | value");
            for (int p = 0; p < N_PLAYERS; ++p) std::printf(" %+.3f", r.value[p]);
            std::printf(" (%.1fs)\n", secondsSince(t0));
        } else {
            usage();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fflush(stdout);
    return 0;
}
