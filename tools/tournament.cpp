// Tournament runner: pits 4 agents against each other across many games,
// rotating seat assignments so seat-position bias is averaged out.
//
// Usage:
//   sk_tournament --games N --seed S --agents A,B,C,D
//
//   Available agents: random, heuristic, ismcts
//   Examples:
//     sk_tournament --games 200 --agents random,random,heuristic,heuristic
//     sk_tournament --games 50  --agents heuristic,heuristic,heuristic,ismcts
//
// Output: per-agent average score, win rate, std-dev across rotations.

#include "sk/game.hpp"
#include "sk/random_agent.hpp"
#include "sk/heuristic_agent.hpp"
#include "sk/ismcts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace sk;

namespace {

std::unique_ptr<Agent> makeAgent(std::string_view name) {
    if (name == "random")    return std::make_unique<RandomAgent>();
    if (name == "heuristic") return std::make_unique<HeuristicAgent>();
    if (name == "ismcts")    return std::make_unique<ISMCTSAgent>();
    std::fprintf(stderr, "Unknown agent: %.*s\n", static_cast<int>(name.size()), name.data());
    std::exit(1);
}

std::vector<std::string> splitCSV(std::string_view s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string_view::npos) j = s.size();
        out.emplace_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

// Play one full Skull King game. `seatToAgentName` indexes which agent NAME plays each seat (0..3).
// `agentsByName` maps name → agent instance (we share instances across games for ISMCTS reuse).
// Returns per-seat final scores.
std::array<std::int32_t, N_PLAYERS>
playOneGame(const std::array<std::string, N_PLAYERS>& seatToAgentName,
            const std::vector<std::pair<std::string, std::unique_ptr<Agent>>>& agentsByName,
            std::mt19937_64& rng)
{
    GameState s = initialState(0);
    dealRound(s, rng);

    while (!isTerminal(s)) {
        const std::string& aname = seatToAgentName[s.currentPlayer];
        Agent* agent = nullptr;
        for (const auto& [n, a] : agentsByName) if (n == aname) { agent = a.get(); break; }
        const Action act = agent->selectAction(s, rng);
        step(s, act, rng);
    }

    std::array<std::int32_t, N_PLAYERS> finals{};
    for (int p = 0; p < N_PLAYERS; ++p) finals[p] = s.scores[p];
    return finals;
}

} // namespace

int main(int argc, char** argv) {
    int numGames = 200;
    std::uint64_t seed = 42;
    std::string agentsCSV = "random,random,heuristic,heuristic";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--games")  numGames  = std::atoi(next("--games"));
        else if (a == "--seed")   seed      = std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--agents") agentsCSV = next("--agents");
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            std::exit(1);
        }
    }

    auto agentList = splitCSV(agentsCSV);
    if (agentList.size() != N_PLAYERS) {
        std::fprintf(stderr, "Need exactly %d agents (got %zu).\n", N_PLAYERS, agentList.size());
        return 1;
    }

    // Build unique agent instances by name (one each, reused across games).
    std::vector<std::pair<std::string, std::unique_ptr<Agent>>> agentsByName;
    for (const auto& nm : agentList) {
        bool exists = false;
        for (const auto& [n, _] : agentsByName) if (n == nm) { exists = true; break; }
        if (!exists) agentsByName.emplace_back(nm, makeAgent(nm));
    }

    // For each "rotation" we cyclically rotate the agent list around the table so
    // each agent plays each seat equally often.
    const int rotations = N_PLAYERS;
    const int gamesPerRotation = std::max(1, numGames / rotations);

    // Stats: total points and wins per (agent name).
    struct Stats {
        std::string name;
        double sumScore = 0;
        double sumSqScore = 0;
        int wins = 0;
        int games = 0;
    };
    std::vector<Stats> stats;
    for (const auto& nm : agentList) {
        bool exists = false;
        for (const auto& s : stats) if (s.name == nm) { exists = true; break; }
        if (!exists) stats.push_back({nm, 0, 0, 0, 0});
    }
    auto bumpStats = [&](const std::string& nm, std::int32_t score, bool win) {
        for (auto& s : stats) if (s.name == nm) {
            s.sumScore += score;
            s.sumSqScore += static_cast<double>(score) * score;
            if (win) ++s.wins;
            ++s.games;
            return;
        }
    };

    std::mt19937_64 rng(seed);

    for (int rot = 0; rot < rotations; ++rot) {
        std::array<std::string, N_PLAYERS> seating{};
        for (int p = 0; p < N_PLAYERS; ++p) {
            seating[p] = agentList[(p + rot) % N_PLAYERS];
        }
        for (int g = 0; g < gamesPerRotation; ++g) {
            const auto finals = playOneGame(seating, agentsByName, rng);
            const int winner = static_cast<int>(
                std::max_element(finals.begin(), finals.end()) - finals.begin());
            for (int p = 0; p < N_PLAYERS; ++p) {
                bumpStats(seating[p], finals[p], p == winner);
            }
        }
    }

    std::printf("\n=== Tournament: %d games per rotation x %d rotations = %d games ===\n",
                gamesPerRotation, rotations, gamesPerRotation * rotations);
    std::printf("Agent line-up: %s\n\n", agentsCSV.c_str());
    std::printf("%-12s %8s %10s %10s %8s %10s\n",
                "agent", "games", "avgScore", "stdScore", "wins", "winRate%");
    std::printf("---------------------------------------------------------------\n");
    for (const auto& s : stats) {
        const double mean = s.sumScore / s.games;
        const double var  = std::max(0.0, s.sumSqScore / s.games - mean * mean);
        const double sd   = std::sqrt(var);
        const double wr   = 100.0 * s.wins / s.games;
        std::printf("%-12s %8d %10.2f %10.2f %8d %9.1f%%\n",
                    s.name.c_str(), s.games, mean, sd, s.wins, wr);
    }
    return 0;
}
