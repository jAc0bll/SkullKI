#pragma once

#include "sk/solver/deals.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace sk::solver {

inline int resolveThreads(int requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? static_cast<int>(hw) : 1;
}

// Prefix size giving enough independent tasks to balance `threads` workers.
inline int dealPrefixFor(int round) {
    (void)round;
    return std::min(2, N_PLAYERS - 1);
}

// Run `perDeal(worker, deal)` over every deal of the round on `nWorkers`
// threads. Each worker index owns its state; deals are distributed in
// task chunks pulled from a shared counter.
template <class Worker, class PerDeal>
void parallelOverDeals(int round, std::vector<Worker>& workers, PerDeal&& perDeal) {
    const std::vector<DealTask> tasks = makeDealTasks(round, dealPrefixFor(round));
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers.size());
    for (std::size_t w = 0; w < workers.size(); ++w) {
        threads.emplace_back([&, w] {
            Worker& wk = workers[w];
            for (;;) {
                const std::size_t t = next.fetch_add(1);
                if (t >= tasks.size()) break;
                forEachDealInTask(tasks[t], round, [&](const KindDeal& d) { perDeal(wk, d); });
            }
        });
    }
    for (auto& th : threads) th.join();
}

} // namespace sk::solver
