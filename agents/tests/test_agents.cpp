#include <catch2/catch_test_macros.hpp>
#include "sk/game.hpp"
#include "sk/random_agent.hpp"
#include "sk/heuristic_agent.hpp"
#include <random>

using namespace sk;

namespace {

std::array<std::int32_t, N_PLAYERS>
playGameWithAgents(Agent& a0, Agent& a1, Agent& a2, Agent& a3, std::mt19937_64& rng) {
    Agent* agents[N_PLAYERS] = {&a0, &a1, &a2, &a3};
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

TEST_CASE("RandomAgent picks only legal actions across 200 games", "[agents]") {
    RandomAgent ra;
    std::mt19937_64 rng(42);
    for (int g = 0; g < 200; ++g) {
        GameState s = initialState(g % N_PLAYERS);
        dealRound(s, rng);
        while (!isTerminal(s)) {
            Action a = ra.selectAction(s, rng);
            // Verify the action is in legalActions.
            const auto la = legalActions(s);
            bool found = false;
            for (const auto& x : la) if (x == a) { found = true; break; }
            REQUIRE(found);
            step(s, a, rng);
        }
    }
}

TEST_CASE("HeuristicAgent never selects illegal actions", "[agents]") {
    HeuristicAgent ha;
    std::mt19937_64 rng(7);
    for (int g = 0; g < 200; ++g) {
        GameState s = initialState(g % N_PLAYERS);
        dealRound(s, rng);
        while (!isTerminal(s)) {
            const Action a = ha.selectAction(s, rng);
            const auto la = legalActions(s);
            bool found = false;
            for (const auto& x : la) if (x == a) { found = true; break; }
            REQUIRE(found);
            step(s, a, rng);
        }
    }
}

TEST_CASE("Heuristic vs Random — heuristic should win significantly more on average", "[agents][baseline]") {
    // Heuristic plays seats 0 and 2; Random plays 1 and 3. Then we rotate.
    // After many games heuristic agents' average final score must exceed random's
    // by a clear margin (this is the sanity check that the heuristic isn't a no-op).
    constexpr int kGames = 60;

    HeuristicAgent h1, h2;
    RandomAgent    r1, r2;
    std::mt19937_64 rng(2024);

    double sumH = 0, sumR = 0;
    int gH = 0, gR = 0;

    // Two seat arrangements: HRHR and RHRH.
    for (int rot = 0; rot < 2; ++rot) {
        for (int g = 0; g < kGames; ++g) {
            std::array<std::int32_t, N_PLAYERS> finals;
            if (rot == 0) finals = playGameWithAgents(h1, r1, h2, r2, rng);
            else          finals = playGameWithAgents(r1, h1, r2, h2, rng);

            for (int p = 0; p < N_PLAYERS; ++p) {
                bool isH = (rot == 0) ? (p == 0 || p == 2) : (p == 1 || p == 3);
                if (isH) { sumH += finals[p]; ++gH; }
                else     { sumR += finals[p]; ++gR; }
            }
        }
    }

    const double avgH = sumH / gH;
    const double avgR = sumR / gR;
    INFO("HeuristicAgent avg=" << avgH << "  RandomAgent avg=" << avgR);
    REQUIRE(avgH > avgR + 50.0);  // expect a clear gap; random is awful at bid-prediction
}
