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

TEST_CASE("Determinizer: hidden bids are un-submitted during bidding", "[determinize][bidding]") {
    std::mt19937_64 rng(21);
    GameState s = initialState(0);
    s.roundNumber = 3;
    dealRound(s, rng);
    applyAction(s, Action::makeBid(3));   // player 0
    applyAction(s, Action::makeBid(2));   // player 1 -> now player 2 to act

    GameState d = determinize(s, 2, rng);
    REQUIRE(d.bids[0] == -1);
    REQUIRE(d.bids[1] == -1);
    REQUIRE(d.bidsSubmitted == 0);
    REQUIRE(d.currentPlayer == 2);

    // The game can be completed from the determinized state.
    applyAction(d, Action::makeBid(1));
    REQUIRE(d.currentPlayer == 3);
    applyAction(d, Action::makeBid(1));
    REQUIRE(d.currentPlayer == 0);
    applyAction(d, Action::makeBid(0));
    REQUIRE(d.currentPlayer == 1);
    applyAction(d, Action::makeBid(0));
    REQUIRE(d.phase == Phase::Playing);
}

TEST_CASE("Determinizer: respects publicly revealed voids", "[determinize][voids]") {
    std::mt19937_64 rng(77);
    GameState s = initialState(0);
    s.roundNumber = 8;
    dealRound(s, rng);
    s.phase = Phase::Playing;
    s.voidSuits[1] = (1u << static_cast<int>(Suit::Yellow)) | (1u << static_cast<int>(Suit::Black));
    s.voidSuits[3] = (1u << static_cast<int>(Suit::Green));
    // Make the true hands consistent with these voids so a valid world exists.
    for (int p : {1, 3}) {
        std::vector<Card> bad;
        s.hands[p].forEach([&](Card c) {
            if (isColored(c) && ((s.voidSuits[p] >> static_cast<int>(suitOf(c))) & 1u)) bad.push_back(c);
        });
        for (Card c : bad) { s.hands[p].remove(c); s.captured[0].add(c); }
    }

    for (int trial = 0; trial < 200; ++trial) {
        GameState d = determinize(s, 0, rng);
        for (int p : {1, 3}) {
            REQUIRE(d.hands[p].count() == s.hands[p].count());
            d.hands[p].forEach([&](Card c) {
                if (isColored(c)) REQUIRE(((s.voidSuits[p] >> static_cast<int>(suitOf(c))) & 1u) == 0);
            });
        }
    }
}
