// Multi-threaded batched-GPU selfplay.
//
// One process owns:
//   - 1× TorchModelEvaluator on the chosen device (CPU or cuda:i)
//   - 1× AsyncBatchEvaluator wrapping it (handles batching across threads)
//   - T MCTS threads, each with its own NeuralMCTSAgent
//
// Each thread plays G/T games. Per game, on each move the agent calls
// evaluator->evaluate(obs); concurrent calls from other threads are folded
// into one batched forward by the AsyncBatchEvaluator scheduler thread.
//
// On terminal we fill per-move value targets from the actual round score
// deltas (unbiased MC return), matching train/selfplay.py's schema.
//
// Output: a simple binary "SKAZ" file (header + per-column blocks). Load
// from Python with train/load_skaz.py.
//
// Build target: selfplay_az (only when SK_BUILD_TORCH=ON).

#include "sk/torch_model.hpp"
#include "sk/async_batch_evaluator.hpp"
#include "sk/neural_mcts.hpp"
#include "sk/game.hpp"
#include "sk/rules.hpp"
#include "sk/encoder.hpp"
#include "sk/observation.hpp"
#include "sk/action_index.hpp"
#include "sk/scoring.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace sk;

namespace {

// Per-move training sample. Packed for direct dump.
struct Sample {
    std::array<float, ENC_DIM>     features;
    std::array<float, ACTION_DIM>  policy;
    std::array<std::uint8_t, ACTION_DIM> legal;
    std::array<float, N_PLAYERS>   value;
    std::uint8_t                   hands[N_PLAYERS][N_CARDS];  // 0/1
    std::uint8_t                   perspective;
    std::uint8_t                   round;
};

void normalisePolicy(std::array<float, ACTION_DIM>& policy,
                     std::array<std::uint8_t, ACTION_DIM>& legal,
                     const ISMCTSResult& r)
{
    policy.fill(0.0f);
    legal.fill(0);
    int totalVisits = 0;
    for (int v : r.visits) totalVisits += v;
    if (totalVisits > 0) {
        for (std::size_t i = 0; i < r.rootActions.size(); ++i) {
            int idx = actionToIndex(r.rootActions[i]);
            policy[idx] = static_cast<float>(r.visits[i]) / totalVisits;
            legal[idx]  = 1;
        }
    } else {
        // Trivial single-legal case.
        for (const auto& a : r.rootActions) {
            int idx = actionToIndex(a);
            policy[idx] = 1.0f / r.rootActions.size();
            legal[idx]  = 1;
        }
    }
}

// One thread's job: play `numGames` self-play games using a fresh NeuralMCTSAgent
// that shares the global async evaluator. Returns its trajectories.
std::vector<Sample>
runThread(int threadId, int numGames, int sims, std::uint64_t baseSeed,
          BaseEvaluator* evaluator)
{
    std::vector<Sample> out;
    out.reserve(static_cast<std::size_t>(numGames) * 260u);   // ~260 actions/game

    NeuralMCTSConfig cfg;
    cfg.mcts.numSimulations = sims;
    cfg.useNNValue          = false;   // hybrid (MC rollout for value)
    NeuralMCTSAgent agent(evaluator, cfg);

    for (int g = 0; g < numGames; ++g) {
        std::mt19937_64 rng(baseSeed + static_cast<std::uint64_t>(threadId) * 100003u
                                     + static_cast<std::uint64_t>(g));
        GameState s = initialState(0);
        dealRound(s, rng);

        std::unordered_map<int, std::array<std::int32_t, N_PLAYERS>> roundStart;
        std::array<std::int32_t, N_PLAYERS> startScores{};
        for (int p = 0; p < N_PLAYERS; ++p) startScores[p] = s.scores[p];
        roundStart[static_cast<int>(s.roundNumber)] = startScores;

        std::vector<std::size_t> gameSampleIdx;     // indices into `out` belonging to this game
        std::vector<int>         gameSampleRound;   // parallel: round number per sample

        while (!isTerminal(s)) {
            const int roundNow = static_cast<int>(s.roundNumber);
            if (!roundStart.count(roundNow)) {
                std::array<std::int32_t, N_PLAYERS> sc{};
                for (int p = 0; p < N_PLAYERS; ++p) sc[p] = s.scores[p];
                roundStart[roundNow] = sc;
            }

            const auto result = agent.selectActionWithTargets(s, rng);

            Sample sm{};
            const Observation obs = observe(s, s.currentPlayer);
            const auto feats = encode(obs);
            std::memcpy(sm.features.data(), feats.data(), ENC_DIM * sizeof(float));
            normalisePolicy(sm.policy, sm.legal, result);
            // Value left as zeros — filled after game terminal.

            sm.perspective = static_cast<std::uint8_t>(s.currentPlayer);
            sm.round       = static_cast<std::uint8_t>(roundNow);
            // True hands (ground truth for belief net).
            for (int p = 0; p < N_PLAYERS; ++p) {
                std::memset(sm.hands[p], 0, N_CARDS);
                s.hands[p].forEach([&](Card c) { sm.hands[p][c] = 1; });
            }

            gameSampleIdx.push_back(out.size());
            gameSampleRound.push_back(roundNow);
            out.push_back(sm);

            step(s, result.action, rng);
        }

        // Fill MC value targets: each sample's value = score delta of its round.
        std::array<std::int32_t, N_PLAYERS> finalScores{};
        for (int p = 0; p < N_PLAYERS; ++p) finalScores[p] = s.scores[p];

        for (std::size_t i = 0; i < gameSampleIdx.size(); ++i) {
            int r = gameSampleRound[i];
            std::array<std::int32_t, N_PLAYERS> endScores =
                (roundStart.count(r + 1) ? roundStart[r + 1] : finalScores);
            for (int p = 0; p < N_PLAYERS; ++p) {
                out[gameSampleIdx[i]].value[p] =
                    (endScores[p] - roundStart[r][p]) / 200.0f;
            }
        }
    }
    return out;
}

void writeSkaz(const std::string& path, const std::vector<Sample>& all)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s for writing\n", path.c_str());
        std::exit(1);
    }
    // Header
    const char magic[4] = {'S', 'K', 'A', 'Z'};
    f.write(magic, 4);
    const std::uint32_t version = 1;
    const std::uint64_t N = static_cast<std::uint64_t>(all.size());
    const std::uint32_t ed = ENC_DIM, ad = ACTION_DIM, np_ = N_PLAYERS, nc = N_CARDS;
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&N), 8);
    f.write(reinterpret_cast<const char*>(&ed), 4);
    f.write(reinterpret_cast<const char*>(&ad), 4);
    f.write(reinterpret_cast<const char*>(&np_), 4);
    f.write(reinterpret_cast<const char*>(&nc), 4);

    // Column blocks
    for (const auto& s : all) f.write(reinterpret_cast<const char*>(s.features.data()), ENC_DIM * sizeof(float));
    for (const auto& s : all) f.write(reinterpret_cast<const char*>(s.policy.data()),   ACTION_DIM * sizeof(float));
    for (const auto& s : all) f.write(reinterpret_cast<const char*>(s.legal.data()),    ACTION_DIM);
    for (const auto& s : all) f.write(reinterpret_cast<const char*>(s.value.data()),    N_PLAYERS * sizeof(float));
    for (const auto& s : all) f.write(reinterpret_cast<const char*>(s.hands),           N_PLAYERS * N_CARDS);
    for (const auto& s : all) f.put(static_cast<char>(s.perspective));
    for (const auto& s : all) f.put(static_cast<char>(s.round));
}

} // namespace

int main(int argc, char** argv)
{
    std::string modelPath = "train/checkpoints/bc_v3_mc.scripted.pt";
    std::string device    = "cpu";
    std::string outPath   = "selfplay_az.bin";
    int    totalGames     = 200;
    int    threads        = 8;
    int    sims           = 100;
    int    maxBatch       = 32;
    long   maxWaitUs      = 1000;
    std::uint64_t baseSeed = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--model")        modelPath  = next("--model");
        else if (a == "--device")       device     = next("--device");
        else if (a == "--out")          outPath    = next("--out");
        else if (a == "--games")        totalGames = std::atoi(next("--games"));
        else if (a == "--threads")      threads    = std::atoi(next("--threads"));
        else if (a == "--sims")         sims       = std::atoi(next("--sims"));
        else if (a == "--max-batch")    maxBatch   = std::atoi(next("--max-batch"));
        else if (a == "--max-wait-us")  maxWaitUs  = std::atol(next("--max-wait-us"));
        else if (a == "--seed")         baseSeed   = std::strtoull(next("--seed"), nullptr, 10);
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    std::printf("selfplay_az\n"
                "  model:      %s\n"
                "  device:     %s\n"
                "  games:      %d (across %d threads)\n"
                "  sims:       %d\n"
                "  max-batch:  %d  max-wait-us: %ld\n"
                "  out:        %s\n",
                modelPath.c_str(), device.c_str(), totalGames, threads,
                sims, maxBatch, maxWaitUs, outPath.c_str());

    // Build shared evaluator stack.
    auto torchEval = std::make_unique<TorchModelEvaluator>(modelPath, device);
    AsyncBatchConfig acfg;
    acfg.maxBatch = maxBatch;
    acfg.maxWait  = std::chrono::microseconds(maxWaitUs);
    AsyncBatchEvaluator async(std::move(torchEval), acfg);

    // Split games across threads (round-robin remainder).
    std::vector<int> gamesPerThread(threads, totalGames / threads);
    for (int i = 0; i < totalGames % threads; ++i) gamesPerThread[i]++;

    // Launch.
    std::vector<std::thread> ths;
    std::vector<std::vector<Sample>> perThread(threads);
    auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < threads; ++t) {
        ths.emplace_back([&, t]{
            perThread[t] = runThread(t, gamesPerThread[t], sims, baseSeed, &async);
        });
    }
    for (auto& th : ths) th.join();
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Merge.
    std::size_t totalSamples = 0;
    for (auto& v : perThread) totalSamples += v.size();
    std::vector<Sample> all;
    all.reserve(totalSamples);
    for (auto& v : perThread) {
        all.insert(all.end(), std::make_move_iterator(v.begin()),
                              std::make_move_iterator(v.end()));
    }

    std::printf("\nGenerated %zu samples from %d games in %.1fs "
                "(%.2f games/sec, %.0f samples/sec)\n"
                "Batched inference: %zu batches, avg batch size %.1f\n",
                all.size(), totalGames, dt,
                totalGames / dt, all.size() / dt,
                async.totalBatches(),
                async.totalBatches() ? double(async.totalRequests()) / double(async.totalBatches()) : 0.0);

    writeSkaz(outPath, all);
    std::printf("Wrote %s\n", outPath.c_str());
    return 0;
}
