#include <catch2/catch_test_macros.hpp>
#include "sk/game.hpp"
#include <random>

using namespace sk;

namespace {

int sumTricks(const GameState& s) {
    int t = 0;
    for (int p = 0; p < N_PLAYERS; ++p) t += s.tricksWon[p];
    return t;
}

int sumHandSizes(const GameState& s) {
    int n = 0;
    for (int p = 0; p < N_PLAYERS; ++p) n += s.hands[p].count();
    return n;
}

int sumCaptured(const GameState& s) {
    int n = 0;
    for (int p = 0; p < N_PLAYERS; ++p) n += s.captured[p].count();
    return n;
}

} // namespace

TEST_CASE("Fuzz: 5000 random games, full invariants", "[fuzz]") {
    for (std::uint64_t seed = 1; seed <= 5000; ++seed) {
        auto s = initialState(static_cast<int>(seed % N_PLAYERS));
        std::mt19937_64 rng(seed);
        dealRound(s, rng);

        int prevRound = s.roundNumber;
        int cardsPerPlayerThisRound = s.roundNumber;

        while (!isTerminal(s)) {
            auto la = legalActions(s);
            REQUIRE_FALSE(la.empty());
            Action a = la[std::uniform_int_distribution<int>(0, static_cast<int>(la.size()) - 1)(rng)];

            // Pre-action invariants
            if (s.phase == Phase::Playing && !s.pendingTigress) {
                // Hand of currentPlayer should contain the card we're about to play.
                if (a.type == ActionType::Play) {
                    REQUIRE(s.hands[s.currentPlayer].has(a.card));
                }
            }

            step(s, a, rng);

            // Round-rollover invariants
            if (s.roundNumber != prevRound) {
                // We just rolled to a new round. Hands should be freshly dealt.
                cardsPerPlayerThisRound = s.roundNumber;
                for (int p = 0; p < N_PLAYERS; ++p) {
                    REQUIRE(s.hands[p].count() == cardsPerPlayerThisRound);
                }
                // tricksWon, captured, etc. reset for the new round.
                REQUIRE(sumTricks(s) == 0);
                REQUIRE(sumCaptured(s) == 0);
                prevRound = s.roundNumber;
            }

            // Mid-round invariant: cards still in play (= sum hands) +
            // cards in current trick + cards captured this round = roundSize * 4
            if (!isTerminal(s) && s.phase == Phase::Playing) {
                int total = sumHandSizes(s) + s.trickSize + sumCaptured(s);
                REQUIRE(total == cardsPerPlayerThisRound * N_PLAYERS);
            }
        }

        REQUIRE(s.roundNumber == MAX_ROUND);
        REQUIRE(s.phase == Phase::GameEnd);
    }
}

TEST_CASE("Fuzz: bid==won implies non-negative bid points", "[fuzz]") {
    // Sanity property at the bid-scoring level.
    for (int r = 1; r <= 10; ++r) {
        for (int b = 0; b <= r; ++b) {
            int p = bidPoints(b, b, r);
            REQUIRE(p >= 0);
            if (b == 0) REQUIRE(p == 10 * r);
            else        REQUIRE(p == 20 * b);
        }
    }
}

TEST_CASE("Fuzz: bid!=won implies non-positive bid points", "[fuzz]") {
    for (int r = 1; r <= 10; ++r) {
        for (int b = 0; b <= r; ++b) {
            for (int w = 0; w <= r; ++w) {
                if (b == w) continue;
                int p = bidPoints(b, w, r);
                REQUIRE(p <= 0);
            }
        }
    }
}
