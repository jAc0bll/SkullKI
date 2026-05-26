#include <catch2/catch_test_macros.hpp>
#include "sk/rules.hpp"
#include "sk/cards.hpp"

using namespace sk;

namespace {

// Helper: resolveTrick with vector inputs in play order.
struct Played {
    Card card;
    std::int8_t player;
};

TrickResult resolve(std::initializer_list<Played> trick, bool tigressAsPirate = false) {
    Card cards[N_PLAYERS];
    std::int8_t players[N_PLAYERS];
    int i = 0;
    for (auto& p : trick) { cards[i] = p.card; players[i] = p.player; ++i; }
    return resolveTrick(cards, players, i, tigressAsPirate);
}

} // namespace

TEST_CASE("Trick: highest of lead suit wins", "[trick]") {
    // Yellow 7 led, Yellow 12, Yellow 8 — Yellow 12 wins.
    auto r = resolve({
        {makeColored(Suit::Yellow,  7), 0},
        {makeColored(Suit::Yellow, 12), 1},
        {makeColored(Suit::Yellow,  8), 2},
    });
    REQUIRE(r.winner == 1);
    REQUIRE(r.bonusForWinner == 0);
}

TEST_CASE("Trick: off-suit non-trump cannot win", "[trick]") {
    // Yellow 8 led, Yellow 5, Purple 10 (off-suit) — Yellow 8 wins.
    auto r = resolve({
        {makeColored(Suit::Yellow,  8), 0},
        {makeColored(Suit::Yellow,  5), 1},
        {makeColored(Suit::Purple, 10), 2},
    });
    REQUIRE(r.winner == 0);
}

TEST_CASE("Trick: trump beats higher non-trump", "[trick]") {
    // Yellow 12 led, Black 5, Black 7 — Black 7 wins (highest trump).
    auto r = resolve({
        {makeColored(Suit::Yellow, 12), 0},
        {makeColored(Suit::Black,   5), 1},
        {makeColored(Suit::Black,   7), 2},
    });
    REQUIRE(r.winner == 2);
}

TEST_CASE("Trick: green 14 vs black 1", "[trick]") {
    // Black is trump; black 1 beats green 14.
    auto r = resolve({
        {makeColored(Suit::Green, 14), 0},
        {makeColored(Suit::Black,  1), 1},
    });
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: pirate beats colors and trump", "[trick]") {
    auto r = resolve({
        {makeColored(Suit::Black, 14), 0},
        {PIRATE_OFFSET,                 1},
    });
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: first-played pirate wins ties", "[trick]") {
    auto r = resolve({
        {makeColored(Suit::Yellow, 5), 0},
        {static_cast<Card>(PIRATE_OFFSET + 0), 1},
        {static_cast<Card>(PIRATE_OFFSET + 1), 2},
    });
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: Skull King beats pirates", "[trick]") {
    auto r = resolve({
        {PIRATE_OFFSET,        0},
        {SKULL_KING,           1},
        {PIRATE_OFFSET + 1,    2},
    });
    REQUIRE(r.winner == 1);
    // Bonus: 30 per pirate captured by SK.
    REQUIRE(r.bonusForWinner == 60);
}

TEST_CASE("Trick: mermaid catches Skull King", "[trick]") {
    auto r = resolve({
        {MERMAID_OFFSET,  0},
        {SKULL_KING,      1},
    });
    REQUIRE(r.winner == 0);
    REQUIRE(r.bonusForWinner == 40);
}

TEST_CASE("Trick: mermaid beats colors but loses to pirate (no SK)", "[trick]") {
    // No SK in trick → mermaid loses to pirate.
    auto r = resolve({
        {makeColored(Suit::Yellow, 14), 0},
        {MERMAID_OFFSET,                 1},
        {PIRATE_OFFSET,                  2},
    });
    REQUIRE(r.winner == 2);
}

TEST_CASE("Trick: Pirate + SK + Mermaid → Mermaid wins, +40 only", "[trick]") {
    // Per rules: when pirate, SK, and ≥1 mermaid in same trick, mermaid (first-played) wins
    // and only she gets the bonus. SK's pirate-bonus is NOT awarded.
    auto r = resolve({
        {makeColored(Suit::Yellow, 14), 0},
        {PIRATE_OFFSET,                  1},
        {SKULL_KING,                     2},
        {MERMAID_OFFSET,                 3},
    });
    REQUIRE(r.winner == 3);
    REQUIRE(r.bonusForWinner == 40);
}

TEST_CASE("Trick: escapes lose to everything", "[trick]") {
    auto r = resolve({
        {ESCAPE_OFFSET,                 0},
        {makeColored(Suit::Yellow, 1),  1},
    });
    // After escape lead, yellow 1 sets lead suit and wins.
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: all escapes — first played wins", "[trick]") {
    auto r = resolve({
        {static_cast<Card>(ESCAPE_OFFSET + 0), 0},
        {static_cast<Card>(ESCAPE_OFFSET + 1), 1},
        {static_cast<Card>(ESCAPE_OFFSET + 2), 2},
    });
    REQUIRE(r.winner == 0);
}

TEST_CASE("Trick: Tigress as pirate beats colors", "[trick]") {
    auto r = resolve({
        {makeColored(Suit::Black, 14), 0},
        {TIGRESS,                      1},
    }, /*tigressAsPirate=*/true);
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: Tigress as escape loses to colors", "[trick]") {
    auto r = resolve({
        {TIGRESS,                       0},   // Lead — escape mode
        {makeColored(Suit::Yellow, 1),  1},
    }, /*tigressAsPirate=*/false);
    REQUIRE(r.winner == 1);
}

TEST_CASE("Trick: Tigress as pirate, SK present — SK wins, Tigress counted as pirate for bonus", "[trick]") {
    auto r = resolve({
        {TIGRESS,         0},
        {SKULL_KING,      1},
        {PIRATE_OFFSET,   2},
    }, /*tigressAsPirate=*/true);
    REQUIRE(r.winner == 1);
    REQUIRE(r.bonusForWinner == 60);  // 2 pirates (incl. Tigress) × 30
}

// ----- legalActions tests -----

#include "sk/game.hpp"

TEST_CASE("legalActions: bidding 0..roundNumber", "[legal]") {
    auto s = initialState();
    REQUIRE(s.phase == Phase::Bidding);
    auto la = legalActions(s);
    REQUIRE(la.size() == 2);  // round 1 → bid 0 or 1
    REQUIRE(la[0].bid == 0);
    REQUIRE(la[1].bid == 1);
}

TEST_CASE("legalActions: must follow lead suit if possible", "[legal]") {
    GameState s = initialState();
    s.phase = Phase::Playing;
    s.currentPlayer = 1;
    s.trickSize = 1;
    s.trickCards[0]   = makeColored(Suit::Yellow, 5);
    s.trickPlayers[0] = 0;
    s.leadSuit = Suit::Yellow;
    s.freeTrick = false;

    s.hands[1].clear();
    s.hands[1].add(makeColored(Suit::Yellow, 3));
    s.hands[1].add(makeColored(Suit::Yellow, 10));
    s.hands[1].add(makeColored(Suit::Purple, 14));
    s.hands[1].add(makeColored(Suit::Black, 2));
    s.hands[1].add(PIRATE_OFFSET);

    auto la = legalActions(s);
    // Yellow 3, Yellow 10, and the Pirate (special) — Purple 14 and Black 2 are illegal.
    REQUIRE(la.size() == 3);
    for (const auto& a : la) {
        Card c = a.card;
        bool ok = (isSpecial(c)) || (suitOf(c) == Suit::Yellow);
        REQUIRE(ok);
    }
}

TEST_CASE("legalActions: no lead suit in hand → free choice", "[legal]") {
    GameState s = initialState();
    s.phase = Phase::Playing;
    s.currentPlayer = 1;
    s.trickSize = 1;
    s.trickCards[0]   = makeColored(Suit::Yellow, 5);
    s.trickPlayers[0] = 0;
    s.leadSuit = Suit::Yellow;
    s.freeTrick = false;

    s.hands[1].clear();
    s.hands[1].add(makeColored(Suit::Purple, 14));
    s.hands[1].add(makeColored(Suit::Black, 2));

    auto la = legalActions(s);
    REQUIRE(la.size() == 2);
}

TEST_CASE("legalActions: freeTrick (character led) → free choice", "[legal]") {
    GameState s = initialState();
    s.phase = Phase::Playing;
    s.currentPlayer = 1;
    s.trickSize = 1;
    s.trickCards[0]   = PIRATE_OFFSET;
    s.trickPlayers[0] = 0;
    s.leadSuit = Suit::None;
    s.freeTrick = true;

    s.hands[1].clear();
    s.hands[1].add(makeColored(Suit::Yellow, 3));
    s.hands[1].add(makeColored(Suit::Purple, 14));

    auto la = legalActions(s);
    REQUIRE(la.size() == 2);
}

TEST_CASE("legalActions: pendingTigress → only TigressMode actions", "[legal]") {
    GameState s = initialState();
    s.phase = Phase::Playing;
    s.pendingTigress = true;
    auto la = legalActions(s);
    REQUIRE(la.size() == 2);
    REQUIRE(la[0].type == ActionType::TigressMode);
    REQUIRE(la[1].type == ActionType::TigressMode);
}
