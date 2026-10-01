#pragma once

#include "sk/solver/kinds.hpp"
#include "sk/card_set.hpp"
#include "sk/state.hpp"

#include <array>
#include <functional>
#include <vector>

namespace sk::solver {

// A deal at the level of kinds: each player's hand as a sorted kind list,
// plus the probability of being dealt exactly these kind-multisets.
struct KindDeal {
    std::array<std::array<Kind, MAX_ROUND>, N_PLAYERS> hand{};
    double prob = 0.0;
};

// Concrete hands (distinct card ids) realising a kind deal. Interchangeable
// cards get arbitrary distinct ids — the rules never tell them apart.
std::array<CardSet, N_PLAYERS> realizeDeal(const KindDeal& d, int round);

// A unit of parallel work: the hands of the first `prefixPlayers` players.
// The remaining players' hands are enumerated by forEachDealInTask.
struct DealTask {
    KindDeal partial;   // prob = weight of the prefix (unnormalised)
    int      prefixPlayers = 0;
};

// Split the full deal space of a round into independent tasks.
std::vector<DealTask> makeDealTasks(int round, int prefixPlayers);

// Enumerate every completion of a task; `prob` passed to `f` is normalised
// so that summing over all tasks gives 1.
void forEachDealInTask(const DealTask& task, int round,
                       const std::function<void(const KindDeal&)>& f);

// Total number of kind-deals for a round (for progress / sanity checks).
std::size_t countDeals(int round);

} // namespace sk::solver
