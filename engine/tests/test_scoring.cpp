#include <catch2/catch_test_macros.hpp>
#include "sk/scoring.hpp"
#include "sk/cards.hpp"

using namespace sk;

TEST_CASE("bidPoints: bid 3, won 3, round 3 → +60", "[scoring]") {
    REQUIRE(bidPoints(3, 3, 3) == 60);
}

TEST_CASE("bidPoints: bid 2, won 4 → -20", "[scoring]") {
    REQUIRE(bidPoints(2, 4, 7) == -20);
}

TEST_CASE("bidPoints: bid 0 hit → +10 * round", "[scoring]") {
    REQUIRE(bidPoints(0, 0, 7) == 70);
    REQUIRE(bidPoints(0, 0, 10) == 100);
}

TEST_CASE("bidPoints: bid 0 missed by 1 → -10 * round", "[scoring]") {
    REQUIRE(bidPoints(0, 1, 9) == -90);
}

TEST_CASE("bonusPoints: 14s", "[scoring]") {
    CardSet cs{};
    cs.add(makeColored(Suit::Yellow, 14));   // +10
    cs.add(makeColored(Suit::Green,  14));   // +10
    cs.add(makeColored(Suit::Purple, 14));   // +10
    cs.add(makeColored(Suit::Black,  14));   // +20
    REQUIRE(bonusPoints(cs, 0) == 50);
}

TEST_CASE("bonusPoints: mermaids in captured pile", "[scoring]") {
    CardSet cs{};
    cs.add(MERMAID_OFFSET);
    cs.add(MERMAID_OFFSET + 1);
    REQUIRE(bonusPoints(cs, 0) == 40);
}

TEST_CASE("bonusPoints: pending bonus carried", "[scoring]") {
    CardSet cs{};
    REQUIRE(bonusPoints(cs, 30) == 30);   // pirate-by-SK
    REQUIRE(bonusPoints(cs, 40) == 40);   // SK-by-mermaid
}

TEST_CASE("scoreRound: example from rules (Bonny bid 3, makes 3)", "[scoring]") {
    GameState s{};
    s.roundNumber = 3;
    s.bids[0] = 3; s.tricksWon[0] = 3;
    s.bids[1] = 0; s.tricksWon[1] = 0;
    s.bids[2] = 2; s.tricksWon[2] = 4;   // off by 2 → -20
    s.bids[3] = 0; s.tricksWon[3] = 1;   // off → -10 * round = -30

    auto rs = scoreRound(s);
    REQUIRE(rs[0].bidPoints == 60);
    REQUIRE(rs[1].bidPoints == 30);
    REQUIRE(rs[2].bidPoints == -20);
    REQUIRE(rs[3].bidPoints == -30);

    // Cumulative scores updated.
    REQUIRE(s.scores[0] == 60);
    REQUIRE(s.scores[1] == 30);
    REQUIRE(s.scores[2] == -20);
    REQUIRE(s.scores[3] == -30);
}

TEST_CASE("scoreRound: bonus only when bid is hit", "[scoring]") {
    GameState s{};
    s.roundNumber = 5;
    s.bids[0] = 2; s.tricksWon[0] = 3;       // missed
    s.captured[0].add(makeColored(Suit::Yellow, 14));  // would be +10 if hit
    s.pendingBonus[0] = 30;                  // would be +30 if hit

    s.bids[1] = 1; s.tricksWon[1] = 1;       // hit
    s.captured[1].add(makeColored(Suit::Black, 14));   // +20
    s.pendingBonus[1] = 40;                  // SK-by-mermaid bonus

    auto rs = scoreRound(s);
    REQUIRE(rs[0].bonusPoints == 0);
    REQUIRE(rs[1].bonusPoints == 60);  // 20 (black 14) + 40 (pending)
    REQUIRE(rs[1].total == 20 + 60);
}
