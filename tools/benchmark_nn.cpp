// Benchmark + validation harness for the C++ TorchNNAgent.
//
// - Confirms the C++-side NN inference produces sensible actions (plays
//   N games against three HeuristicAgents and reports score).
// - Times raw NN evaluation throughput (single + batched).
//
// Run from the project root after building:
//     build\tools\benchmark_nn.exe --model train\checkpoints\bc_400sims.scripted.pt --games 80

#include "sk/game.hpp"
#include "sk/heuristic_agent.hpp"
#include "sk/ismcts.hpp"
#include "sk/torch_nn_agent.hpp"
#include "sk/neural_mcts.hpp"
#include "sk/belief_model.hpp"
#include "sk/observation.hpp"

#include <memory>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace sk;

namespace {

double seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// One full game with the given agent line-up. Returns final scores.
std::array<std::int32_t, N_PLAYERS>
playGame(Agent* agents[N_PLAYERS], std::mt19937_64& rng) {
    GameState s = initialState(0);
    dealRound(s, rng);
    while (!isTerminal(s)) {
        const Action a = agents[s.currentPlayer]->selectAction(s, rng);
        step(s, a, rng);
    }
    std::array<std::int32_t, N_PLAYERS> out{};
    for (int p = 0; p < N_PLAYERS; ++p) out[p] = s.scores[p];
    return out;
}

void singleEvalThroughput(TorchModelEvaluator& eval, int iters) {
    std::mt19937_64 rng(0);
    GameState s = initialState(0);
    dealRound(s, rng);
    Observation obs = observe(s, 0);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)eval.evaluate(obs);
    }
    const double dt = seconds(t0, std::chrono::steady_clock::now());
    std::printf("single-eval throughput: %d in %.3fs = %.0f eval/s (%.3f ms each)\n",
                iters, dt, iters / dt, 1000.0 * dt / iters);
}

void batchedEvalThroughput(TorchModelEvaluator& eval, int batch, int iters) {
    std::mt19937_64 rng(1);
    GameState s = initialState(0);
    dealRound(s, rng);
    std::vector<Observation> batchObs(batch, observe(s, 0));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)eval.evaluateBatch(batchObs);
    }
    const double dt = seconds(t0, std::chrono::steady_clock::now());
    const long long total = static_cast<long long>(iters) * batch;
    std::printf("batched(%d) throughput: %d iters in %.3fs = %.0f obs/s (%.3f ms / batch)\n",
                batch, iters, dt, total / dt, 1000.0 * dt / iters);
}

} // namespace

int main(int argc, char** argv) {
    std::string modelPath  = "train/checkpoints/bc_v3_mc.scripted.pt";
    std::string beliefPath = "";   // empty -> uniform determinizer
    std::string device     = "cpu";
    int games              = 40;
    int singleIters        = 2000;
    int batchSize          = 64;
    int batchIters         = 200;
    int mctsSims           = 50;
    int ismctsSims         = 400;
    std::string mode       = "all";  // "nn", "mcts", "vs-ismcts", "belief", "all"

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] (const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--model")        modelPath = next("--model");
        else if (a == "--belief")       beliefPath = next("--belief");
        else if (a == "--device")       device    = next("--device");
        else if (a == "--games")        games     = std::atoi(next("--games"));
        else if (a == "--single-iters") singleIters = std::atoi(next("--single-iters"));
        else if (a == "--batch-size")   batchSize = std::atoi(next("--batch-size"));
        else if (a == "--batch-iters")  batchIters = std::atoi(next("--batch-iters"));
        else if (a == "--mcts-sims")    mctsSims = std::atoi(next("--mcts-sims"));
        else if (a == "--ismcts-sims")  ismctsSims = std::atoi(next("--ismcts-sims"));
        else if (a == "--mode")         mode = next("--mode");
        else { std::fprintf(stderr, "Unknown arg %s\n", a.c_str()); return 1; }
    }

    std::printf("Loading TorchScript module: %s  (device=%s)\n",
                modelPath.c_str(), device.c_str());
    TorchNNAgent nn(modelPath, device);

    auto runTournament = [&](const char* label, Agent& candidate, Agent& opponent1,
                             Agent& opponent2, Agent& opponent3,
                             const char* candLabel, const char* oppLabel) {
        std::printf("\n--- Tournament: %s vs 3xOpponent (seat-rotated) ---\n", label);

        std::mt19937_64 rng(42);
        long long candScore = 0, oppScoreSum = 0;
        int candWins = 0, oppSeats = 0;
        const int rotations    = N_PLAYERS;
        const int gamesPerRot  = std::max(1, games / rotations);
        const int totalGames   = gamesPerRot * rotations;

        auto t0 = std::chrono::steady_clock::now();
        for (int rot = 0; rot < rotations; ++rot) {
            const int candSeat = rot;
            Agent* lineup[N_PLAYERS] = {&opponent1, &opponent2, &opponent3, &opponent1};
            lineup[candSeat] = &candidate;
            for (int g = 0; g < gamesPerRot; ++g) {
                const auto sc = playGame(lineup, rng);
                const int winner = static_cast<int>(
                    std::max_element(sc.begin(), sc.end()) - sc.begin());
                if (winner == candSeat) ++candWins;
                candScore += sc[candSeat];
                for (int p = 0; p < N_PLAYERS; ++p) {
                    if (p != candSeat) { oppScoreSum += sc[p]; ++oppSeats; }
                }
            }
        }
        const double dt = seconds(t0, std::chrono::steady_clock::now());
        std::printf("games          : %d (%.1f games/sec)\n", totalGames, totalGames / dt);
        std::printf("%-14s avg : %.1f   wins %d/%d  (%.1f%%)\n",
                    candLabel,
                    static_cast<double>(candScore) / totalGames,
                    candWins, totalGames, 100.0 * candWins / totalGames);
        std::printf("%-14s avg : %.1f (per seat over %d games)\n",
                    oppLabel,
                    static_cast<double>(oppScoreSum) / oppSeats, oppSeats);
    };

    HeuristicAgent h1, h2, h3, h4;
    ISMCTSConfig ismctsCfg{};
    ismctsCfg.numSimulations = ismctsSims;
    ISMCTSAgent ismcts1(ismctsCfg), ismcts2(ismctsCfg), ismcts3(ismctsCfg);

    NeuralMCTSConfig nmctsCfg{};
    nmctsCfg.mcts.numSimulations = mctsSims;
    nmctsCfg.useNNValue = true;
    NeuralMCTSAgent nmctsValue(&nn.evaluator(), nmctsCfg);

    NeuralMCTSConfig nmctsHybridCfg{};
    nmctsHybridCfg.mcts.numSimulations = mctsSims;
    nmctsHybridCfg.useNNValue = false;
    NeuralMCTSAgent nmctsHybrid(&nn.evaluator(), nmctsHybridCfg);

    std::unique_ptr<BeliefEvaluator> belief;
    std::unique_ptr<NeuralMCTSAgent> nmctsBelief;
    if (!beliefPath.empty()) {
        std::printf("Loading BeliefNet: %s\n", beliefPath.c_str());
        belief = std::make_unique<BeliefEvaluator>(beliefPath, device);
        NeuralMCTSConfig cfg{};
        cfg.mcts.numSimulations = mctsSims;
        cfg.useNNValue          = false;
        cfg.belief              = belief.get();
        nmctsBelief = std::make_unique<NeuralMCTSAgent>(&nn.evaluator(), cfg);
    }

    if (mode == "nn" || mode == "all") {
        std::printf("\n--- Raw NN throughput ---\n");
        singleEvalThroughput(nn.evaluator(), singleIters);
        batchedEvalThroughput(nn.evaluator(), batchSize, batchIters);
        runTournament("TorchNN-only vs 3xHeuristic", nn, h1, h2, h3, "TorchNN", "Heuristic");
    }
    if (mode == "mcts" || mode == "all") {
        std::printf("\n=== NeuralMCTS-NNValue(%d sims) vs 3xHeuristic ===\n", mctsSims);
        runTournament("NeuralMCTS(NN-V) vs 3xHeuristic",
                      nmctsValue, h1, h2, h3, "NMCTS-NN-V", "Heuristic");

        std::printf("\n=== NeuralMCTS-Hybrid(%d sims, MC-rollout) vs 3xHeuristic ===\n", mctsSims);
        runTournament("NeuralMCTS(MC) vs 3xHeuristic",
                      nmctsHybrid, h1, h2, h3, "NMCTS-MC", "Heuristic");
    }
    if (mode == "vs-ismcts" || mode == "all") {
        std::printf("\n=== NeuralMCTS-NNValue(%d) vs ISMCTS(%d) ===\n", mctsSims, ismctsSims);
        runTournament("NeuralMCTS(NN-V) vs 3xISMCTS",
                      nmctsValue, ismcts1, ismcts2, ismcts3, "NMCTS-NN-V", "ISMCTS");

        std::printf("\n=== NeuralMCTS-Hybrid(%d) vs ISMCTS(%d) ===\n", mctsSims, ismctsSims);
        runTournament("NeuralMCTS(MC) vs 3xISMCTS",
                      nmctsHybrid, ismcts1, ismcts2, ismcts3, "NMCTS-MC", "ISMCTS");
    }
    if ((mode == "belief" || mode == "all") && nmctsBelief) {
        std::printf("\n=== NeuralMCTS+Belief(%d) vs 3xHeuristic ===\n", mctsSims);
        runTournament("NMCTS+Belief vs 3xHeuristic",
                      *nmctsBelief, h1, h2, h3, "NMCTS+Belief", "Heuristic");

        std::printf("\n=== NeuralMCTS+Belief(%d) vs ISMCTS(%d) ===\n", mctsSims, ismctsSims);
        runTournament("NMCTS+Belief vs 3xISMCTS",
                      *nmctsBelief, ismcts1, ismcts2, ismcts3, "NMCTS+Belief", "ISMCTS");
    }
    return 0;
}
