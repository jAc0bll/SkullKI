#pragma once

#include "sk/evaluator.hpp"

#include <chrono>
#include <cstddef>
#include <memory>

namespace sk {

struct AsyncBatchConfig {
    // Maximum number of requests to fold into one underlying forward pass.
    int                       maxBatch = 64;
    // After the first pending request arrives, wait up to this long for
    // additional requests before flushing. Trade-off: longer = bigger
    // batches but higher per-call latency. 0 means flush immediately.
    std::chrono::microseconds maxWait{1000};
};

// Multi-producer, single-consumer batched-inference evaluator.
//
// Many MCTS worker threads call `evaluate(obs)` concurrently; the call
// blocks the worker thread on a std::future. A dedicated scheduler thread
// owned by this object:
//   1. Waits for at least one pending request.
//   2. Briefly waits for more requests (up to cfg.maxWait or cfg.maxBatch).
//   3. Drains pending requests, calls underlying->evaluateBatch on the
//      whole batch.
//   4. Fulfils each worker's promise so they unblock.
//
// This lets one TorchModelEvaluator (CPU or GPU) saturate a GPU with
// real batched forward passes even though MCTS is fundamentally serial
// per-game: parallelism comes from running many MCTS games as threads
// inside one process, all sharing one AsyncBatchEvaluator.
//
// The wrapped `underlying` is owned by AsyncBatchEvaluator; on destruction
// the scheduler thread is joined and any in-flight requests have their
// promises rejected.
//
// `evaluateBatch(batch)` bypasses the scheduler and forwards straight to
// the underlying — caller already knows what they want batched.
class AsyncBatchEvaluator final : public BaseEvaluator {
public:
    AsyncBatchEvaluator(std::unique_ptr<BaseEvaluator> underlying,
                        AsyncBatchConfig cfg = {});
    ~AsyncBatchEvaluator() override;

    AsyncBatchEvaluator(const AsyncBatchEvaluator&)            = delete;
    AsyncBatchEvaluator& operator=(const AsyncBatchEvaluator&) = delete;

    PolicyValue evaluate(const Observation& obs) override;
    std::vector<PolicyValue>
    evaluateBatch(const std::vector<Observation>& batch) override;

    // Diagnostics: how many requests are currently sitting in the queue.
    std::size_t pendingCount() const;

    // Diagnostics: cumulative batch counts since construction.
    std::size_t totalBatches()  const;
    std::size_t totalRequests() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sk
