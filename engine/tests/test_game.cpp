#include <catch2/catch_test_macros.hpp>
#include "sk/game.hpp"
#include <random>

using namespace sk;

TEST_CASE("Initial state and dealing", "[game]") {
    auto s = initialState(0);
    REQUIRE(s.phase == Phase::Bidding);
    REQUIRE(s.roundNumber == 1);
    REQUIRE(s.currentPlayer == 0);

    std::mt19937_64 rng(42);
    dealRound(s, rng);

    // Round 1 → each player has 1 card.
    int total = 0;
    for (int p = 0; p < N_PLAYERS; ++p) {
        REQUIRE(s.hands[p].count() == 1);
        total += s.hands[p].count();
    }
    REQUIRE(total == 4);
}

TEST_CASE("Bidding flow advances and transitions to Playing", "[game]") {
    auto s = initialState(0);
    std::mt19937_64 rng(1);
    dealRound(s, rng);

    REQUIRE(s.phase == Phase::Bidding);
    applyAction(s, Action::makeBid(0));
    REQUIRE(s.currentPlayer == 1);
    applyAction(s, Action::makeBid(1));
    applyAction(s, Action::makeBid(0));
    applyAction(s, Action::makeBid(1));

    REQUIRE(s.phase == Phase::Playing);
    REQUIRE(s.currentPlayer == s.startPlayer);
    REQUIRE(s.bidsSubmitted == 4);
}

TEST_CASE("Full random game completes without errors", "[game]") {
    auto s = initialState(0);
    std::mt19937_64 rng(123);
    dealRound(s, rng);

    int safety = 0;
    while (!isTerminal(s) && safety++ < 10000) {
        auto la = legalActions(s);
        REQUIRE(!la.empty());
        std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
        Action a = la[dist(rng)];
        step(s, a, rng);
    }
    REQUIRE(isTerminal(s));
    REQUIRE(s.roundNumber == MAX_ROUND);
}

TEST_CASE("Tigress played as lead-pirate creates free trick", "[game]") {
    // Use a 2-trick round so we can inspect tricksWon after the first trick
    // without the round-end reset wiping it out.
    GameState s = initialState(0);
    s.phase = Phase::Playing;
    s.currentPlayer = 0;
    s.trickLeader = 0;
    s.roundNumber = 2;
    s.bidsSubmitted = 4;
    for (int p = 0; p < N_PLAYERS; ++p) s.bids[p] = 0;  // bids must be set for scoreRound

    s.hands[0].clear();
    s.hands[0].add(TIGRESS);
    s.hands[0].add(makeColored(Suit::Yellow, 1));
    s.hands[1].clear();
    s.hands[1].add(makeColored(Suit::Yellow, 5));
    s.hands[1].add(makeColored(Suit::Yellow, 6));
    s.hands[2].clear();
    s.hands[2].add(makeColored(Suit::Purple, 8));
    s.hands[2].add(makeColored(Suit::Purple, 9));
    s.hands[3].clear();
    s.hands[3].add(makeColored(Suit::Green, 14));
    s.hands[3].add(makeColored(Suit::Green, 13));

    applyAction(s, Action::makePlay(TIGRESS));
    REQUIRE(s.pendingTigress);
    REQUIRE(s.currentPlayer == 0);  // still on player 0, awaiting mode

    applyAction(s, Action::makeTigressMode(true));  // as pirate
    REQUIRE(!s.pendingTigress);
    REQUIRE(s.freeTrick);
    REQUIRE(s.currentPlayer == 1);

    applyAction(s, Action::makePlay(makeColored(Suit::Yellow, 5)));
    applyAction(s, Action::makePlay(makeColored(Suit::Purple, 8)));
    applyAction(s, Action::makePlay(makeColored(Suit::Green, 14)));

    // Tigress-as-pirate beats all colors → player 0 wins this trick.
    REQUIRE(s.tricksWon[0] == 1);
    REQUIRE(s.trickLeader == 0);
    REQUIRE(s.tricksPlayed == 1);
}

TEST_CASE("Tigress played as lead-escape lets next colored set lead suit", "[game]") {
    GameState s = initialState(0);
    s.phase = Phase::Playing;
    s.currentPlayer = 0;
    s.trickLeader = 0;
    s.roundNumber = 2;
    s.bidsSubmitted = 4;
    for (int p = 0; p < N_PLAYERS; ++p) s.bids[p] = 0;

    // P0 leads Tigress as Escape, P1 plays Purple 5 (sets lead = Purple),
    // P2 has no Purple → off-suit Yellow 14, P3 plays Purple 9.
    // Highest Purple is 9 → P3 wins.
    s.hands[0].clear(); s.hands[0].add(TIGRESS); s.hands[0].add(makeColored(Suit::Yellow, 1));
    s.hands[1].clear(); s.hands[1].add(makeColored(Suit::Purple, 5)); s.hands[1].add(makeColored(Suit::Purple, 1));
    s.hands[2].clear(); s.hands[2].add(makeColored(Suit::Yellow, 14)); s.hands[2].add(makeColored(Suit::Yellow, 2));
    s.hands[3].clear(); s.hands[3].add(makeColored(Suit::Purple, 9)); s.hands[3].add(makeColored(Suit::Purple, 2));

    applyAction(s, Action::makePlay(TIGRESS));
    applyAction(s, Action::makeTigressMode(false));  // as escape
    REQUIRE(!s.freeTrick);

    applyAction(s, Action::makePlay(makeColored(Suit::Purple, 5)));
    REQUIRE(s.leadSuit == Suit::Purple);

    applyAction(s, Action::makePlay(makeColored(Suit::Yellow, 14)));
    applyAction(s, Action::makePlay(makeColored(Suit::Purple, 9)));

    REQUIRE(s.tricksWon[3] == 1);
    REQUIRE(s.trickLeader == 3);
}
