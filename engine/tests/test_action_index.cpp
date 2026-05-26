#include <catch2/catch_test_macros.hpp>
#include "sk/action_index.hpp"

using namespace sk;

TEST_CASE("action_index round-trip for all action types", "[action_index]") {
    REQUIRE(ACTION_DIM == 83);

    // Bids 0..10
    for (int b = 0; b <= 10; ++b) {
        const Action a = Action::makeBid(b);
        const int idx = actionToIndex(a);
        REQUIRE(idx == b);
        const Action back = indexToAction(idx);
        REQUIRE(back == a);
    }

    // Plays 0..69
    for (int c = 0; c < N_CARDS; ++c) {
        const Action a = Action::makePlay(static_cast<Card>(c));
        const int idx = actionToIndex(a);
        REQUIRE(idx == 11 + c);
        const Action back = indexToAction(idx);
        REQUIRE(back == a);
    }

    // TigressMode
    REQUIRE(actionToIndex(Action::makeTigressMode(true))  == 81);
    REQUIRE(actionToIndex(Action::makeTigressMode(false)) == 82);
    REQUIRE(indexToAction(81) == Action::makeTigressMode(true));
    REQUIRE(indexToAction(82) == Action::makeTigressMode(false));
}
