#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sk/solver/abstraction.hpp"
#include "sk/solver/best_response.hpp"
#include "sk/solver/cfr.hpp"
#include "sk/solver/deals.hpp"
#include "sk/solver/mccfr.hpp"
#include "sk/solver/round.hpp"

#include <random>

using namespace sk;
using namespace sk::solver;

namespace {

std::array<CardSet, N_PLAYERS> randomHands(int round, std::mt19937_64& rng) {
    GameState s = initialState(0);
    s.roundNumber = static_cast<std::uint8_t>(round);
    dealRound(s, rng);
    std::array<CardSet, N_PLAYERS> h{};
    for (int p = 0; p < N_PLAYERS; ++p) h[p] = s.hands[p];
    return h;
}

// Play uniformly random kind-actions to the end of the round.
RoundState playRandomly(RoundState rs, std::mt19937_64& rng) {
    while (!rs.terminal()) {
        ActionList al;
        legalKindActions(rs.s, al);
        std::uniform_int_distribution<int> d(0, al.n - 1);
        applyRound(rs, al[d(rng)]);
    }
    return rs;
}

} // namespace

TEST_CASE("Kinds: identical cards map to one kind, ids round-trip", "[solver][kinds]") {
    REQUIRE(kindOf(ESCAPE_OFFSET) == kindOf(ESCAPE_OFFSET + 4));
    REQUIRE(kindOf(PIRATE_OFFSET) == kindOf(PIRATE_OFFSET + 4));
    REQUIRE(kindOf(MERMAID_OFFSET) == kindOf(MERMAID_OFFSET + 1));
    int total = 0;
    for (int k = 0; k < N_KINDS; ++k) {
        total += kindMultiplicity(static_cast<Kind>(k));
        REQUIRE(kindOf(firstCardOfKind(static_cast<Kind>(k))) == k);
    }
    REQUIRE(total == N_CARDS);
}

TEST_CASE("Deals: round-1 kind deals cover the deck with total probability 1", "[solver][deals]") {
    double sum = 0.0;
    std::size_t n = 0;
    for (const DealTask& t : makeDealTasks(1, 2)) {
        forEachDealInTask(t, 1, [&](const KindDeal& d) {
            sum += d.prob;
            ++n;
        });
    }
    REQUIRE(n == countDeals(1));
    REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("Deals: realized hands are disjoint and match the kinds", "[solver][deals]") {
    KindDeal d;
    d.hand[0][0] = KIND_PIRATE;  d.hand[0][1] = KIND_PIRATE;
    d.hand[1][0] = KIND_PIRATE;  d.hand[1][1] = KIND_ESCAPE;
    d.hand[2][0] = 3;            d.hand[2][1] = KIND_SK;
    d.hand[3][0] = KIND_ESCAPE;  d.hand[3][1] = KIND_TIGRESS;
    const auto hands = realizeDeal(d, 2);
    CardSet all;
    for (int p = 0; p < N_PLAYERS; ++p) {
        REQUIRE(hands[p].count() == 2);
        hands[p].forEach([&](Card c) {
            REQUIRE(!all.has(c));
            all.add(c);
        });
    }
    int pirates = 0;
    all.forEach([&](Card c) { pirates += isPirate(c); });
    REQUIRE(pirates == 3);
}

TEST_CASE("legalKindActions collapses interchangeable cards", "[solver][actions]") {
    GameState s = initialState(0);
    s.roundNumber = 4;
    s.phase = Phase::Playing;
    s.hands[0].add(PIRATE_OFFSET);
    s.hands[0].add(PIRATE_OFFSET + 3);
    s.hands[0].add(ESCAPE_OFFSET + 1);
    s.hands[0].add(makeColored(Suit::Green, 9));
    ActionList al;
    legalKindActions(s, al);
    REQUIRE(al.n == 3);   // Green 9, Escape, Pirate
}

TEST_CASE("Infoset keys contain only the player's information", "[solver][keys]") {
    std::mt19937_64 rng(5);
    for (int trial = 0; trial < 50; ++trial) {
        const int round = 1 + trial % 4;
        RoundState a = makeRoundState(round, randomHands(round, rng));

        // Bidding: key ignores opponents' hands AND already-submitted bids.
        RoundState b = a;
        applyRound(a, Action::makeBid(0));
        applyRound(b, Action::makeBid(1));
        b.s.hands[2] = a.s.hands[3];
        b.s.hands[3] = a.s.hands[2];
        InfoKey ka, kb;
        infosetKey(a, 1, ka);
        infosetKey(b, 1, kb);
        REQUIRE(ka == kb);

        // Playing: key ignores opponents' hands but reflects public plays.
        RoundState c = makeRoundState(round, randomHands(round, rng));
        for (int i = 0; i < N_PLAYERS; ++i) applyRound(c, Action::makeBid(i % 2));
        RoundState c2 = c;
        const int me = c.s.currentPlayer;
        const int other = (me + 1) % N_PLAYERS;
        std::swap(c2.s.hands[other], c2.s.hands[(me + 2) % N_PLAYERS]);
        infosetKey(c, me, ka);
        infosetKey(c2, me, kb);
        REQUIRE(ka == kb);

        InfoKey before, after;
        infosetKey(c, other, before);
        ActionList al;
        legalKindActions(c.s, al);
        applyRound(c, al[0]);
        infosetKey(c, other, after);
        REQUIRE(after != before);   // the public play changed what `other` knows
    }
}

TEST_CASE("Relative utility is zero-sum", "[solver][utility]") {
    std::mt19937_64 rng(9);
    for (int trial = 0; trial < 200; ++trial) {
        const int round = 1 + trial % 5;
        RoundState rs = playRandomly(makeRoundState(round, randomHands(round, rng)), rng);
        const auto u = roundUtilities(rs, Utility::Relative);
        double sum = 0.0;
        for (double x : u) sum += x;
        REQUIRE(sum == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("Exact policy value matches Monte Carlo simulation (round 1, uniform)", "[solver][eval]") {
    EvalConfig cfg;
    cfg.round = 1;
    cfg.utility = Utility::Absolute;
    const UniformPolicy uniform;
    const auto exact = policyValue(uniform, cfg);

    std::mt19937_64 rng(2024);
    constexpr int N = 400000;
    std::array<double, N_PLAYERS> mc{};
    for (int i = 0; i < N; ++i) {
        RoundState rs = playRandomly(makeRoundState(1, randomHands(1, rng)), rng);
        const auto u = roundUtilities(rs, Utility::Absolute);
        for (int p = 0; p < N_PLAYERS; ++p) mc[p] += u[p] / N;
    }
    for (int p = 0; p < N_PLAYERS; ++p) {
        REQUIRE(mc[p] == Catch::Approx(exact[p]).margin(0.1));   // ~4 sigma
    }
}

TEST_CASE("Best response never loses to the profile it answers", "[solver][br]") {
    EvalConfig cfg;
    cfg.round = 1;
    const UniformPolicy uniform;
    const auto rep = exploitability(uniform, cfg);
    for (int p = 0; p < N_PLAYERS; ++p) {
        REQUIRE(rep.gain[p] >= -1e-9);
        // A round-1 utility is bounded by the scoring (|points| <= 10+40+20...),
        // so a best response can never gain more than a few hundred points.
        REQUIRE(rep.brValue[p] < 200.0);
    }
    REQUIRE(rep.nashConv > 1.0);   // uniform random bidding is very exploitable
}

namespace {
// Uniform for everyone, except `player` always bids `bid` (if legal).
class FixedBidPolicy final : public PolicyView {
public:
    FixedBidPolicy(int player, int bid) : player_(player), bid_(bid) {}
    void probs(const RoundState& rs, int player, const ActionList& legal,
               double* out) const override {
        const bool mine = rs.s.phase == Phase::Bidding && player == player_ && bid_ < legal.n;
        for (int a = 0; a < legal.n; ++a) out[a] = mine ? (a == bid_ ? 1.0 : 0.0) : 1.0 / legal.n;
    }
private:
    int player_, bid_;
};
} // namespace

TEST_CASE("Best response dominates every fixed-bid deviation", "[solver][br]") {
    EvalConfig cfg;
    cfg.round = 1;
    const UniformPolicy uniform;
    for (int p : {0, 3}) {
        const double br = bestResponseValue(uniform, p, cfg);
        for (int bid : {0, 1}) {
            const double dev = policyValue(FixedBidPolicy(p, bid), cfg)[p];
            REQUIRE(br >= dev - 1e-9);
        }
    }
}

TEST_CASE("CFR on round 1 drives NashConv towards zero", "[solver][cfr][.slow]") {
    CFRConfig c;
    c.round = 1;
    FullWidthCFR cfr(c);
    EvalConfig e;
    e.round = 1;
    double first = 0.0;
    for (int i = 1; i <= 30; ++i) {
        cfr.iterate();
        if (i == 2) first = exploitability(AveragePolicy(cfr.table()), e).nashConv;
    }
    const double last = exploitability(AveragePolicy(cfr.table()), e).nashConv;
    REQUIRE(last < first);
    REQUIRE(last < 0.5);
}

TEST_CASE("MCCFR runs on larger rounds and keeps strategies normalised", "[solver][mccfr]") {
    MCCFRConfig c;
    c.round = 4;
    c.dealsPerBatch = 2000;
    ExternalSamplingMCCFR m(c);
    for (int i = 0; i < 3; ++i) m.runBatch();
    REQUIRE(m.traversals() >= 3u * 2000u * N_PLAYERS);
    REQUIRE(m.table().size() > 1000);
    m.table().forEach([&](const InfoKey&, const InfoNode& n) {
        double sum = 0.0;
        for (double x : n.current) {
            REQUIRE(x >= 0.0);
            sum += x;
        }
        REQUIRE(sum == Catch::Approx(1.0));
    });
}

TEST_CASE("MCCFR on round 1 is far less exploitable than uniform", "[solver][mccfr]") {
    MCCFRConfig c;
    c.round = 1;
    c.dealsPerBatch = 50000;
    ExternalSamplingMCCFR m(c);
    for (int i = 0; i < 20; ++i) m.runBatch();
    EvalConfig e;
    e.round = 1;
    const double nc = exploitability(AveragePolicy(m.table()), e).nashConv;
    REQUIRE(nc < 1.0);   // uniform: ~31
}

namespace {
// Relabel Yellow/Green/Purple by perm; Black and specials stay.
Card permuteCard(Card c, const int* perm) {
    if (!isColored(c) || suitOf(c) == Suit::Black) return c;
    return static_cast<Card>(perm[static_cast<int>(suitOf(c))] * CARDS_PER_SUIT + (c % CARDS_PER_SUIT));
}
} // namespace

TEST_CASE("Suit isomorphism: permuted situations share key, actions keep their slot", "[solver][abstraction]") {
    const SuitIsomorphism iso;
    const int perm[3] = {2, 0, 1};
    std::mt19937_64 rng(31);
    for (int trial = 0; trial < 300; ++trial) {
        const int round = 2 + trial % 5;
        const auto hands = randomHands(round, rng);
        std::array<CardSet, N_PLAYERS> permHands{};
        for (int p = 0; p < N_PLAYERS; ++p)
            hands[p].forEach([&](Card c) { permHands[p].add(permuteCard(c, perm)); });

        RoundState a = makeRoundState(round, hands);
        RoundState b = makeRoundState(round, permHands);
        // Play a random number of random actions, mirrored in b.
        std::uniform_int_distribution<int> steps(0, N_PLAYERS + 2 * round);
        const int n = steps(rng);
        for (int i = 0; i < n && !a.terminal(); ++i) {
            ActionList al;
            legalKindActions(a.s, al);
            std::uniform_int_distribution<int> d(0, al.n - 1);
            Action x = al[d(rng)];
            Action y = x;
            if (x.type == ActionType::Play) y.card = permuteCard(x.card, perm);
            applyRound(a, x);
            applyRound(b, y);
        }
        if (a.terminal()) continue;

        const int p = a.s.currentPlayer;
        ActionList la, lb;
        legalKindActions(a.s, la);
        legalKindActions(b.s, lb);
        REQUIRE(la.n == lb.n);
        InfosetView va, vb;
        iso.view(a, p, la, va);
        iso.view(b, p, lb, vb);
        REQUIRE(va.key == vb.key);
        for (int i = 0; i < la.n; ++i) {
            // Find the action in b that corresponds to la[i].
            Action y = la[i];
            if (y.type == ActionType::Play) y.card = permuteCard(y.card, perm);
            int j = -1;
            for (int k = 0; k < lb.n; ++k) {
                const bool same = lb[k].type == y.type &&
                    (y.type != ActionType::Play || kindOf(lb[k].card) == kindOf(y.card)) &&
                    (y.type != ActionType::Bid || lb[k].bid == y.bid) &&
                    (y.type != ActionType::TigressMode || lb[k].asPirate == y.asPirate);
                if (same) j = k;
            }
            REQUIRE(j >= 0);
            // Slots may only differ when the situation is self-symmetric.
            REQUIRE((va.slot[i] == vb.slot[j] || va.symmetric));
        }
    }
}

TEST_CASE("MCCFR with suit isomorphism needs fewer infosets", "[solver][abstraction][mccfr]") {
    MCCFRConfig c;
    c.round = 1;
    c.dealsPerBatch = 50000;
    ExternalSamplingMCCFR exact(c);
    c.abstraction = makeAbstraction("suit-iso");
    ExternalSamplingMCCFR iso(c);
    for (int i = 0; i < 5; ++i) { exact.runBatch(); iso.runBatch(); }
    REQUIRE(iso.table().size() < exact.table().size());
}

TEST_CASE("Feature abstraction: private, well-formed slots", "[solver][abstraction][features]") {
    const FeatureAbstraction fa;
    std::mt19937_64 rng(47);
    for (int trial = 0; trial < 2000; ++trial) {
        const int round = 1 + trial % 10;
        RoundState rs = makeRoundState(round, randomHands(round, rng));
        std::uniform_int_distribution<int> steps(0, N_PLAYERS * (round + 1));
        const int n = steps(rng);
        for (int i = 0; i < n && !rs.terminal(); ++i) {
            ActionList al;
            legalKindActions(rs.s, al);
            std::uniform_int_distribution<int> d(0, al.n - 1);
            applyRound(rs, al[d(rng)]);
        }
        if (rs.terminal()) continue;
        const int p = rs.s.currentPlayer;
        ActionList al;
        legalKindActions(rs.s, al);
        InfosetView v;
        fa.view(rs, p, al, v);

        // Every slot is used, slots are in range.
        REQUIRE(v.nSlots >= 1);
        REQUIRE(v.nSlots <= al.n);
        std::array<bool, ActionList::CAPACITY> used{};
        for (int a = 0; a < al.n; ++a) {
            REQUIRE(v.slot[a] < v.nSlots);
            used[v.slot[a]] = true;
        }
        for (int k = 0; k < v.nSlots; ++k) REQUIRE(used[k]);

        // Swapping two opponents' hidden hands (same sizes) changes nothing.
        const int q1 = (p + 1) % N_PLAYERS, q2 = (p + 2) % N_PLAYERS;
        if (rs.s.hands[q1].count() == rs.s.hands[q2].count()) {
            RoundState other = rs;
            std::swap(other.s.hands[q1], other.s.hands[q2]);
            InfosetView w;
            fa.view(other, p, al, w);
            REQUIRE(w.key == v.key);
        }
        // Hidden bids during bidding do not leak into the key.
        if (rs.s.phase == Phase::Bidding) {
            RoundState other = rs;
            for (int q = 0; q < N_PLAYERS; ++q)
                if (q != p && other.s.bids[q] >= 0) other.s.bids[q] = (other.s.bids[q] + 1) % (round + 1);
            InfosetView w;
            fa.view(other, p, al, w);
            REQUIRE(w.key == v.key);
        }
    }
}

TEST_CASE("Feature abstraction keeps big-round tables small", "[solver][abstraction][features]") {
    MCCFRConfig c;
    c.round = 5;
    c.dealsPerBatch = 2000;
    ExternalSamplingMCCFR exact(c);
    c.abstraction = makeAbstraction("features");
    ExternalSamplingMCCFR feat(c);
    for (int i = 0; i < 2; ++i) { exact.runBatch(); feat.runBatch(); }
    REQUIRE(feat.table().size() * 3 < exact.table().size());
}

// ---------------------------------------------------------------------------
// Neural solver plumbing
// ---------------------------------------------------------------------------
#include "sk/solver/deep.hpp"
#include "sk/solver/encoding.hpp"
#include "sk/solver/mlp.hpp"

#include <cstdio>
#include <fstream>

TEST_CASE("Encoding: action indices are distinct and in range", "[deep][encoding]") {
    std::array<bool, ACT_DIM> seen{};
    auto check = [&](Action a) {
        const int i = actionIndex(a);
        REQUIRE(i >= 0);
        REQUIRE(i < ACT_DIM);
        REQUIRE(!seen[i]);
        seen[i] = true;
    };
    for (int b = 0; b <= 10; ++b) check(Action::makeBid(b));
    for (int k = 0; k < N_KINDS; ++k) check(Action::makePlay(firstCardOfKind(static_cast<Kind>(k))));
    check(Action::makeTigressMode(true));
    check(Action::makeTigressMode(false));
}

TEST_CASE("Encoding: infoset features hide other hands and hidden bids", "[deep][encoding]") {
    std::mt19937_64 rng(13);
    std::uint8_t a[INFO_DIM], b[INFO_DIM];
    for (int trial = 0; trial < 2000; ++trial) {
        const int round = 1 + trial % 10;
        RoundState rs = makeRoundState(round, randomHands(round, rng));
        std::uniform_int_distribution<int> steps(0, N_PLAYERS * (round + 1));
        const int n = steps(rng);
        for (int i = 0; i < n && !rs.terminal(); ++i) {
            ActionList al;
            legalKindActions(rs.s, al);
            applyRound(rs, al[std::uniform_int_distribution<int>(0, al.n - 1)(rng)]);
        }
        if (rs.terminal()) continue;
        const int p = rs.s.currentPlayer;
        encodeInfoset(rs, p, a);
        RoundState other = rs;
        const int q1 = (p + 1) % N_PLAYERS, q2 = (p + 3) % N_PLAYERS;
        if (other.s.hands[q1].count() == other.s.hands[q2].count())
            std::swap(other.s.hands[q1], other.s.hands[q2]);
        if (other.s.phase == Phase::Bidding)
            for (int q = 0; q < N_PLAYERS; ++q)
                if (q != p && other.s.bids[q] >= 0) other.s.bids[q] = static_cast<std::int8_t>((other.s.bids[q] + 1) % (round + 1));
        encodeInfoset(other, p, b);
        REQUIRE(std::equal(a, a + INFO_DIM, b));
    }
}

TEST_CASE("MLP: loads exported weights and computes a ReLU network", "[deep][mlp]") {
    // 3 -> 2 (ReLU) -> 1, hand-made weights.
    const std::string path = "test_mlp_tmp.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("SKMLP001", 8);
        const std::uint32_t n = 2;
        f.write(reinterpret_cast<const char*>(&n), 4);
        auto layer = [&](std::uint32_t in, std::uint32_t out, std::vector<float> w, std::vector<float> b) {
            f.write(reinterpret_cast<const char*>(&in), 4);
            f.write(reinterpret_cast<const char*>(&out), 4);
            f.write(reinterpret_cast<const char*>(w.data()), w.size() * 4);
            f.write(reinterpret_cast<const char*>(b.data()), b.size() * 4);
        };
        layer(3, 2, {1, 2, 3, -1, -1, -1}, {0.5f, 0.0f});
        layer(2, 1, {2, 10}, {-1.0f});
    }
    const MLP m = MLP::load(path);
    std::remove(path.c_str());
    const std::uint8_t x[3] = {1, 0, 2};
    float y = 0;
    m.forward(x, &y);
    // h = relu([1+6+0.5, -1-2]) = [7.5, 0]; y = 2*7.5 + 0 - 1 = 14
    REQUIRE(y == Catch::Approx(14.0f));
}

TEST_CASE("npy writer produces a valid header", "[deep][npy]") {
    const std::string path = "test_npy_tmp.npy";
    writeNpy(path, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    std::ifstream f(path, std::ios::binary);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    std::remove(path.c_str());
    REQUIRE(all.substr(1, 5) == "NUMPY");
    const std::size_t hlen = static_cast<unsigned char>(all[8]) | (static_cast<unsigned char>(all[9]) << 8);
    REQUIRE((10 + hlen) % 64 == 0);
    REQUIRE(all.find("'shape': (2, 3)") != std::string::npos);
    REQUIRE(all.size() == 10 + hlen + 6 * 4);
}

TEST_CASE("Regret samples with exact values are consistent", "[deep][gen]") {
    GenConfig g;
    g.round = 1;
    g.count = 200;
    const UniformPolicy uniform;
    RegretSamples r;
    PolicySamples p;
    generateRegretSamples(uniform, nullptr, g, r, p);
    REQUIRE(r.size() > 0);
    REQUIRE(p.size() > 0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        // Regrets of the legal actions sum to 0 under the uniform strategy.
        double sum = 0.0;
        int legal = 0;
        for (int a = 0; a < ACT_DIM; ++a) {
            if (r.mask[i * ACT_DIM + a]) { sum += r.target[i * ACT_DIM + a]; ++legal; }
            else REQUIRE(r.target[i * ACT_DIM + a] == 0.0f);
        }
        REQUIRE(legal >= 2);
        REQUIRE(sum == Catch::Approx(0.0).margin(1e-3));
    }
}

TEST_CASE("Policy samples can be thinned", "[deep][gen]") {
    GenConfig g;
    g.round = 3;
    g.count = 400;
    const UniformPolicy uniform;
    RegretSamples r0, r1, rh;
    PolicySamples p0, p1, ph;
    generateRegretSamples(uniform, nullptr, g, r1, p1);
    g.policyKeep = 0.0;
    generateRegretSamples(uniform, nullptr, g, r0, p0);
    g.policyKeep = 0.5;
    generateRegretSamples(uniform, nullptr, g, rh, ph);
    REQUIRE(p0.size() == 0);
    // Regret samples are never thinned (counts differ only because the extra
    // random draws change the sampled games).
    REQUIRE(static_cast<double>(r0.size()) > 0.9 * static_cast<double>(r1.size()));
    REQUIRE(static_cast<double>(r0.size()) < 1.1 * static_cast<double>(r1.size()));
    const double frac = static_cast<double>(ph.size()) / static_cast<double>(p1.size());
    REQUIRE(frac > 0.4);
    REQUIRE(frac < 0.6);
}

TEST_CASE("ImprovedPolicy: follows the base unless confidently better", "[deep][improve]") {
    // Advantage net: no hidden layer, zero weights, bias +10 for "bid 1",
    // so the advantage of bid 1 is 10 points everywhere.
    const std::string path = "test_adv_tmp.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("SKMLP001", 8);
        const std::uint32_t n = 1, in = INFO_DIM, out = ACT_DIM;
        f.write(reinterpret_cast<const char*>(&n), 4);
        f.write(reinterpret_cast<const char*>(&in), 4);
        f.write(reinterpret_cast<const char*>(&out), 4);
        std::vector<float> w(static_cast<std::size_t>(in) * out, 0.0f), b(out, 0.0f);
        b[1] = 10.0f;   // action index 1 = bid 1
        f.write(reinterpret_cast<const char*>(w.data()), w.size() * 4);
        f.write(reinterpret_cast<const char*>(b.data()), b.size() * 4);
    }
    auto adv = std::make_shared<const MLP>(MLP::load(path));
    std::remove(path.c_str());

    std::array<CardSet, N_PLAYERS> hands{};
    hands[0].add(makeColored(Suit::Yellow, 3));
    RoundState rs = makeRoundState(1, hands);
    ActionList al;
    legalKindActions(rs.s, al);   // bid 0, bid 1
    double pr[2];

    ImprovedPolicy confident(std::make_unique<UniformPolicy>(), adv, 5.0);
    confident.probs(rs, 0, al, pr);
    REQUIRE(pr[0] == 0.0);
    REQUIRE(pr[1] == 1.0);

    ImprovedPolicy cautious(std::make_unique<UniformPolicy>(), adv, 20.0);
    cautious.probs(rs, 0, al, pr);
    REQUIRE(pr[0] == Catch::Approx(0.5));   // below threshold: the base (uniform)
    REQUIRE(pr[1] == Catch::Approx(0.5));
}

TEST_CASE("MLP: optimised forward matches a naive reference", "[deep][mlp]") {
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<float> uw(-0.5f, 0.5f);
    for (int outDim : {7, 1}) {
        const std::vector<std::pair<int, int>> shapes = {{37, 24}, {24, 24}, {24, outDim}};
        std::vector<std::vector<float>> W, B;
        const std::string path = "test_mlp_ref_tmp.bin";
        {
            std::ofstream f(path, std::ios::binary);
            f.write("SKMLP001", 8);
            const std::uint32_t n = static_cast<std::uint32_t>(shapes.size());
            f.write(reinterpret_cast<const char*>(&n), 4);
            for (auto [in, out] : shapes) {
                std::vector<float> w(static_cast<std::size_t>(in) * out), b(out);
                for (float& v : w) v = uw(rng);
                for (float& v : b) v = uw(rng);
                const std::uint32_t i32 = in, o32 = out;
                f.write(reinterpret_cast<const char*>(&i32), 4);
                f.write(reinterpret_cast<const char*>(&o32), 4);
                f.write(reinterpret_cast<const char*>(w.data()), w.size() * 4);
                f.write(reinterpret_cast<const char*>(b.data()), b.size() * 4);
                W.push_back(w);
                B.push_back(b);
            }
        }
        const MLP m = MLP::load(path);
        std::remove(path.c_str());
        for (int trial = 0; trial < 50; ++trial) {
            std::uint8_t x[37];
            for (auto& v : x) v = (rng() % 3 == 0) ? static_cast<std::uint8_t>(rng() % 4) : 0;
            // Naive reference: row-major, double precision.
            std::vector<double> cur(x, x + 37);
            for (std::size_t li = 0; li < shapes.size(); ++li) {
                const auto [in, out] = shapes[li];
                std::vector<double> nxt(out);
                for (int o = 0; o < out; ++o) {
                    double acc = B[li][o];
                    for (int k = 0; k < in; ++k) acc += W[li][static_cast<std::size_t>(o) * in + k] * cur[k];
                    nxt[o] = (li + 1 < shapes.size() && acc < 0.0) ? 0.0 : acc;
                }
                cur = nxt;
            }
            std::vector<float> y(outDim);
            m.forward(x, y.data());
            for (int o = 0; o < outDim; ++o) REQUIRE(y[o] == Catch::Approx(cur[o]).margin(1e-4));
        }
    }
}
