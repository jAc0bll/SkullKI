#include <catch2/catch_test_macros.hpp>
#include "sk/cards.hpp"
#include "sk/card_set.hpp"

using namespace sk;

TEST_CASE("Card ID layout", "[cards]") {
    REQUIRE(N_CARDS == 70);
    REQUIRE(isColored(makeColored(Suit::Yellow, 1)));
    REQUIRE(valueOf(makeColored(Suit::Yellow, 14)) == 14);
    REQUIRE(suitOf(makeColored(Suit::Black, 7)) == Suit::Black);

    REQUIRE(isEscape(ESCAPE_OFFSET));
    REQUIRE(isEscape(ESCAPE_OFFSET + 4));
    REQUIRE(!isEscape(MERMAID_OFFSET));

    REQUIRE(isMermaid(MERMAID_OFFSET));
    REQUIRE(isMermaid(MERMAID_OFFSET + 1));
    REQUIRE(!isMermaid(PIRATE_OFFSET));

    REQUIRE(isPirate(PIRATE_OFFSET));
    REQUIRE(isPirate(PIRATE_OFFSET + 4));
    REQUIRE(!isPirate(TIGRESS));

    REQUIRE(isTigress(TIGRESS));
    REQUIRE(isSkullKing(SKULL_KING));

    REQUIRE(isTrump(makeColored(Suit::Black, 1)));
    REQUIRE(!isTrump(makeColored(Suit::Yellow, 14)));

    REQUIRE(is14(makeColored(Suit::Green, 14)));
    REQUIRE(!is14(makeColored(Suit::Green, 13)));
    REQUIRE(!is14(SKULL_KING));
}

TEST_CASE("CardSet basic ops", "[card_set]") {
    CardSet s{};
    REQUIRE(s.empty());
    REQUIRE(s.count() == 0);

    s.add(0);
    s.add(SKULL_KING);
    s.add(TIGRESS);
    REQUIRE(s.count() == 3);
    REQUIRE(s.has(0));
    REQUIRE(s.has(TIGRESS));
    REQUIRE(s.has(SKULL_KING));
    REQUIRE(!s.has(1));

    s.remove(0);
    REQUIRE(!s.has(0));
    REQUIRE(s.count() == 2);

    int seen = 0;
    s.forEach([&](Card){ ++seen; });
    REQUIRE(seen == 2);
}
