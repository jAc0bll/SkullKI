#include <catch2/catch_test_macros.hpp>
#include "sk/game.hpp"
#include "sk/ismcts.hpp"
#include "sk/random_agent.hpp"
#include "sk/heuristic_agent.hpp"
#include <random>

using namespace sk;

namespace {

std::array<std::int32_t, N_PLAYERS>
playGame(Agent* agents[N_PLAYERS], std::mt19937_64& rng) {
    GameState s = initialState(0);
    dealRound(s, rng);
    while (!isTerminal(s)) {
        const Action a = agents[s.currentPlayer]->selectAction(s, rng);
        step(s, a, rng);
    }
    std::array<std::int32_t, N_PLAYERS> finals{};
    for (int p = 0; p < N_PLAYERS; ++p) finals[p] = s.scores[p];
    return finals;
}

} // namespace

TEST_CASE("ISMCTS picks legal actions in a full game", "[ismcts]") {
    ISMCTSAgent mcts({.numSimulations = 64});
    std::mt19937_64 rng(1);
    GameState s = initialState(0);
    dealRound(s, rng);
    while (!isTerminal(s)) {
        const Action a = mcts.selectAction(s, rng);
        const auto la = legalActions(s);
        bool found = false;
        for (const auto& x : la) if (x == a) { found = true; break; }
        REQUIRE(found);
        step(s, a, rng);
    }
}

TEST_CASE("ISMCTS (low sims) at least matches Random over many games", "[ismcts][baseline]") {
    // With only 32 simulations ISMCTS shouldn't be dominant, but it should at
    // least average non-negative score delta against three random opponents.
    constexpr int kGames = 16;

    ISMCTSAgent mcts({.numSimulations = 32});
    RandomAgent r1, r2, r3;
    Agent* roster[N_PLAYERS] = {&mcts, &r1, &r2, &r3};

    std::mt19937_64 rng(2025);
    double sumMcts = 0;
    double sumRand = 0;
    for (int g = 0; g < kGames; ++g) {
        const auto finals = playGame(roster, rng);
        sumMcts += finals[0];
        for (int p = 1; p < N_PLAYERS; ++p) sumRand += finals[p];
    }
    const double avgMcts = sumMcts / kGames;
    const double avgRand = sumRand / (3.0 * kGames);
    INFO("ISMCTS(32) avg=" << avgMcts << "  Random avg=" << avgRand);
    REQUIRE(avgMcts > avgRand);
}
