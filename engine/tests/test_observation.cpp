#include <catch2/catch_test_macros.hpp>
#include "sk/observation.hpp"
#include "sk/game.hpp"
#include "sk/encoder.hpp"
#include <random>

using namespace sk;

TEST_CASE("observe: opponents' hands cleared, own hand kept, sizes preserved", "[observation]") {
    std::mt19937_64 rng(42);
    GameState s = initialState(0);
    dealRound(s, rng);  // round 1 → 1 card per player.

    const CardSet myHand = s.hands[2];

    Observation o = observe(s, 2);
    REQUIRE(o.perspective == 2);
    REQUIRE(o.s.hands[2] == myHand);

    for (int p = 0; p < N_PLAYERS; ++p) {
        REQUIRE(o.handSizes[p] == s.hands[p].count());
        if (p != 2) REQUIRE(o.s.hands[p].empty());
    }

    // Public arrays should be untouched.
    REQUIRE(o.s.roundNumber == s.roundNumber);
    REQUIRE(o.s.phase       == s.phase);
}

TEST_CASE("encode: output has exact ENC_DIM length, deterministic, finite", "[encoder]") {
    std::mt19937_64 rng(7);
    GameState s = initialState(0);
    dealRound(s, rng);

    // Run a few applyActions to mix the state up a bit.
    for (int i = 0; i < 4; ++i) {
        const Action a = legalActions(s).front();
        applyAction(s, a);
    }

    for (int p = 0; p < N_PLAYERS; ++p) {
        Observation o = observe(s, p);
        std::vector<float> v1 = encode(o);
        std::vector<float> v2 = encode(o);
        REQUIRE(v1.size() == static_cast<std::size_t>(ENC_DIM));
        REQUIRE(v1 == v2);  // deterministic
        for (float x : v1) REQUIRE(std::isfinite(x));
    }
}

TEST_CASE("encode: changing only opponent hands does NOT change the encoding", "[encoder][privacy]") {
    // Hidden-info invariant: from the perspective of one player, swapping which
    // cards opponents hold (consistent with public history) must not change
    // their observation. We verify the encoder respects this by mutating
    // opponent hands in the original GameState and checking the encoded
    // observation stays identical.
    std::mt19937_64 rng(123);
    GameState s = initialState(0);
    s.roundNumber = 6;
    dealRound(s, rng);

    Observation oA = observe(s, 1);
    const auto encA = encode(oA);

    // Manually swap two opponents' hands at the engine level.
    std::swap(s.hands[0], s.hands[2]);

    Observation oB = observe(s, 1);
    const auto encB = encode(oB);

    REQUIRE(encA == encB);
}
