#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sk/solver/best_response.hpp"
#include "sk/solver/cfr.hpp"
#include "sk/solver/deals.hpp"
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
    void probs(const InfoKey& key, int nA, double* out) const override {
        const bool mine = key[2] == 0 && key[1] == player_ && bid_ < nA;
        for (int a = 0; a < nA; ++a) out[a] = mine ? (a == bid_ ? 1.0 : 0.0) : 1.0 / nA;
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
