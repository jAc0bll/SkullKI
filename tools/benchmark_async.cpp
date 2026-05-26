// Benchmark for AsyncBatchEvaluator.
//
// Compares two paths for N total NN-evaluations:
//   1) Serial: one thread, call underlying TorchModelEvaluator->evaluate N times.
//   2) Async-batched: T worker threads, each calling AsyncBatchEvaluator->evaluate
//      N/T times; the scheduler folds concurrent requests into batched forward
//      passes of up to MAX_BATCH.
//
// Run from project root after building:
//   build/tools/benchmark_async --model train/checkpoints/bc_v3_mc.scripted.pt \
//       --threads 8 --calls-per-thread 200 --max-batch 32 --max-wait-us 1000
//
// Expected behaviour:
//   - On CPU: async should be marginally faster (one model, fewer cBLAS calls).
//   - On GPU: async should be substantially faster (batched forward saturates
//     the device; single-call forward has high launch overhead).

#include "sk/torch_model.hpp"
#include "sk/async_batch_evaluator.hpp"
#include "sk/observation.hpp"
#include "sk/encoder.hpp"
#include "sk/game.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using sk::BaseEvaluator;
using sk::Observation;
using sk::PolicyValue;

// Mock evaluator that simulates fixed per-batch latency independent of batch
// size — i.e. the regime a GPU operates in (launch overhead dominates, batch
// dim is "free"). Lets us show the batching speedup without an actual GPU.
class MockEvaluator final : public BaseEvaluator {
public:
    explicit MockEvaluator(std::chrono::microseconds latency_per_batch)
        : latency_(latency_per_batch) {}
    PolicyValue evaluate(const Observation&) override {
        std::this_thread::sleep_for(latency_);
        PolicyValue pv{};
        pv.values[0] = 0.123f;
        return pv;
    }
    std::vector<PolicyValue> evaluateBatch(const std::vector<Observation>& batch) override {
        // Per-batch latency is constant — the heart of the batching win.
        std::this_thread::sleep_for(latency_);
        std::vector<PolicyValue> out(batch.size());
        for (auto& pv : out) pv.values[0] = 0.123f;
        return out;
    }
private:
    std::chrono::microseconds latency_;
};

namespace {

double seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

Observation makeSampleObservation(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    sk::GameState s = sk::initialState(0);
    sk::dealRound(s, rng);
    return sk::observe(s, 0);
}

} // namespace

int main(int argc, char** argv) {
    std::string modelPath   = "train/checkpoints/bc_v3_mc.scripted.pt";
    std::string device      = "cpu";
    int  threads            = 8;
    int  callsPerThread     = 200;
    int  maxBatch           = 32;
    long maxWaitUs          = 1000;
    bool useMock            = false;
    long mockLatencyUs      = 1000;  // simulate GPU launch overhead

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--model")             modelPath       = next("--model");
        else if (a == "--device")            device          = next("--device");
        else if (a == "--threads")           threads         = std::atoi(next("--threads"));
        else if (a == "--calls-per-thread")  callsPerThread  = std::atoi(next("--calls-per-thread"));
        else if (a == "--max-batch")         maxBatch        = std::atoi(next("--max-batch"));
        else if (a == "--max-wait-us")       maxWaitUs       = std::atol(next("--max-wait-us"));
        else if (a == "--mock")              useMock         = true;
        else if (a == "--mock-latency-us")   mockLatencyUs   = std::atol(next("--mock-latency-us"));
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    auto makeBase = [&]() -> std::unique_ptr<BaseEvaluator> {
        if (useMock) {
            return std::make_unique<MockEvaluator>(
                std::chrono::microseconds(mockLatencyUs));
        }
        return std::make_unique<sk::TorchModelEvaluator>(modelPath, device);
    };

    const long long total = static_cast<long long>(threads) * callsPerThread;
    std::printf("Model: %s  device=%s  threads=%d  calls/thread=%d  total=%lld\n",
                modelPath.c_str(), device.c_str(), threads, callsPerThread, total);

    // Pre-build per-thread observations (slight variation by seed so the
    // batcher never trivially deduplicates).
    std::vector<Observation> obsList;
    obsList.reserve(threads);
    for (int t = 0; t < threads; ++t) obsList.push_back(makeSampleObservation(0x1234ULL + t));

    // ---- 1. Serial baseline ----
    {
        auto base = makeBase();
        auto t0 = std::chrono::steady_clock::now();
        long long checksum = 0;
        for (long long i = 0; i < total; ++i) {
            const Observation& o = obsList[i % threads];
            auto pv = base->evaluate(o);
            checksum += static_cast<long long>(pv.values[0] * 1000.0f);
        }
        auto dt = seconds(t0, std::chrono::steady_clock::now());
        std::printf("\n[serial]            %lld calls in %.3fs  =>  %.1f calls/s  (checksum %lld)\n",
                    total, dt, total / dt, checksum);
    }

    // ---- 2. Async-batched ----
    {
        auto base = makeBase();
        sk::AsyncBatchConfig cfg;
        cfg.maxBatch = maxBatch;
        cfg.maxWait  = std::chrono::microseconds(maxWaitUs);
        sk::AsyncBatchEvaluator async(std::move(base), cfg);

        std::vector<std::thread> ths;
        std::vector<long long>   sums(threads, 0);
        std::atomic<int>         go{0};

        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; ++t) {
            ths.emplace_back([&, t]{
                while (go.load() == 0) std::this_thread::yield();
                for (int i = 0; i < callsPerThread; ++i) {
                    auto pv = async.evaluate(obsList[t]);
                    sums[t] += static_cast<long long>(pv.values[0] * 1000.0f);
                }
            });
        }
        go.store(1);
        for (auto& th : ths) th.join();
        auto dt = seconds(t0, std::chrono::steady_clock::now());

        long long checksum = 0;
        for (auto s : sums) checksum += s;
        std::printf("[async maxBatch=%d] %lld calls in %.3fs  =>  %.1f calls/s  "
                    "(%zu batches, avg size %.1f)  (checksum %lld)\n",
                    maxBatch, total, dt, total / dt,
                    async.totalBatches(),
                    async.totalBatches()
                        ? double(async.totalRequests()) / double(async.totalBatches())
                        : 0.0,
                    checksum);
    }
    return 0;
}
