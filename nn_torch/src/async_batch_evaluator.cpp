#include "sk/async_batch_evaluator.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sk {

namespace {

struct Request {
    Observation               obs;
    std::promise<PolicyValue> result;
};

} // namespace

struct AsyncBatchEvaluator::Impl {
    std::unique_ptr<BaseEvaluator>     underlying;
    AsyncBatchConfig                   cfg;
    mutable std::mutex                 mtx;
    std::condition_variable            cv;
    std::deque<Request>                queue;
    std::atomic<bool>                  running{true};
    std::atomic<std::size_t>           totalBatches{0};
    std::atomic<std::size_t>           totalRequests{0};
    std::thread                        scheduler;

    Impl(std::unique_ptr<BaseEvaluator> u, AsyncBatchConfig c)
        : underlying(std::move(u)), cfg(c) {}

    void start() { scheduler = std::thread([this]{ run(); }); }

    void stop() {
        running.store(false);
        cv.notify_all();
        if (scheduler.joinable()) scheduler.join();

        // Reject any leftover requests so callers don't hang.
        std::lock_guard<std::mutex> lock(mtx);
        while (!queue.empty()) {
            try {
                queue.front().result.set_exception(std::make_exception_ptr(
                    std::runtime_error("AsyncBatchEvaluator destroyed with pending requests")));
            } catch (...) { /* future already retrieved exception, ignore */ }
            queue.pop_front();
        }
    }

    void run() {
        while (running.load()) {
            std::vector<Request> batch;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]{ return !queue.empty() || !running.load(); });
                if (!running.load() && queue.empty()) break;

                // Brief wait to let more requests arrive (batching window).
                if (static_cast<int>(queue.size()) < cfg.maxBatch && cfg.maxWait.count() > 0) {
                    cv.wait_for(lock, cfg.maxWait, [this]{
                        return static_cast<int>(queue.size()) >= cfg.maxBatch
                               || !running.load();
                    });
                }

                const int take = std::min(static_cast<int>(queue.size()), cfg.maxBatch);
                batch.reserve(take);
                for (int i = 0; i < take; ++i) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }

            if (batch.empty()) continue;

            // Forward pass (outside lock — other threads can enqueue meanwhile).
            std::vector<Observation> obsList;
            obsList.reserve(batch.size());
            for (auto& r : batch) obsList.push_back(r.obs);

            std::vector<PolicyValue> results;
            try {
                results = underlying->evaluateBatch(obsList);
            } catch (...) {
                // Propagate failure to each waiting caller.
                auto eptr = std::current_exception();
                for (auto& r : batch) {
                    try { r.result.set_exception(eptr); } catch (...) {}
                }
                continue;
            }

            for (std::size_t i = 0; i < batch.size(); ++i) {
                try {
                    batch[i].result.set_value(std::move(results[i]));
                } catch (...) {
                    // Future on the caller side gone (shouldn't happen).
                }
            }

            totalBatches.fetch_add(1);
            totalRequests.fetch_add(batch.size());
        }
    }
};

AsyncBatchEvaluator::AsyncBatchEvaluator(std::unique_ptr<BaseEvaluator> underlying,
                                         AsyncBatchConfig cfg)
    : impl_(std::make_unique<Impl>(std::move(underlying), cfg))
{
    impl_->start();
}

AsyncBatchEvaluator::~AsyncBatchEvaluator() {
    impl_->stop();
}

PolicyValue AsyncBatchEvaluator::evaluate(const Observation& obs) {
    std::promise<PolicyValue> prom;
    auto fut = prom.get_future();
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        impl_->queue.push_back({obs, std::move(prom)});
    }
    impl_->cv.notify_one();
    return fut.get();
}

std::vector<PolicyValue>
AsyncBatchEvaluator::evaluateBatch(const std::vector<Observation>& batch)
{
    // Caller already chose the batch; forward straight to the underlying
    // model — going through the scheduler would only add latency.
    return impl_->underlying->evaluateBatch(batch);
}

std::size_t AsyncBatchEvaluator::pendingCount() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->queue.size();
}

std::size_t AsyncBatchEvaluator::totalBatches()  const { return impl_->totalBatches.load(); }
std::size_t AsyncBatchEvaluator::totalRequests() const { return impl_->totalRequests.load(); }

} // namespace sk
