#include <catch2/catch_test_macros.hpp>
#include "sk/game.hpp"
#include "sk/determinize.hpp"
#include <random>

using namespace sk;

TEST_CASE("Determinizer: perspective hand identical, opponents resampled", "[determinize]") {
    std::mt19937_64 rng(11);
    GameState s = initialState(0);
    s.roundNumber = 5;
    s.phase = Phase::Playing;
    dealRound(s, rng);  // (re-)deals according to roundNumber. Bidding state irrelevant for this test.

    const int me = 1;
    const CardSet myHand = s.hands[me];

    for (int trial = 0; trial < 100; ++trial) {
        GameState d = determinize(s, me, rng);

        // My own hand is unchanged.
        REQUIRE(d.hands[me] == myHand);

        // Each opponent has the same hand-size as before.
        for (int p = 0; p < N_PLAYERS; ++p) {
            if (p == me) continue;
            REQUIRE(d.hands[p].count() == s.hands[p].count());
        }

        // Across all players, total cards in hands + captured + current trick
        // should still equal roundNumber * N_PLAYERS.
        int total = 0;
        for (int p = 0; p < N_PLAYERS; ++p) {
            total += d.hands[p].count();
            total += d.captured[p].count();
        }
        total += d.trickSize;
        REQUIRE(total == s.roundNumber * N_PLAYERS);

        // No card is duplicated across hands.
        CardSet acc;
        for (int p = 0; p < N_PLAYERS; ++p) {
            d.hands[p].forEach([&](Card c) {
                REQUIRE(!acc.has(c));
                acc.add(c);
            });
        }
    }
}

TEST_CASE("Determinizer: respects already-played cards (captured + current trick)", "[determinize]") {
    std::mt19937_64 rng(99);
    GameState s = initialState(0);
    s.roundNumber = 4;
    dealRound(s, rng);

    // Simulate two completed tricks by moving 8 cards into captured[0].
    // We pull them out of the hands accordingly to keep invariants.
    int moved = 0;
    for (int p = 0; p < N_PLAYERS && moved < 8; ++p) {
        std::vector<Card> tmp;
        s.hands[p].forEach([&](Card c) { tmp.push_back(c); });
        for (Card c : tmp) {
            if (moved >= 8) break;
            s.hands[p].remove(c);
            s.captured[0].add(c);
            ++moved;
        }
    }

    for (int trial = 0; trial < 50; ++trial) {
        GameState d = determinize(s, 1, rng);
        // Captured cards must not appear in any hand.
        s.captured[0].forEach([&](Card c) {
            for (int p = 0; p < N_PLAYERS; ++p) {
                REQUIRE(!d.hands[p].has(c));
            }
        });
    }
}
