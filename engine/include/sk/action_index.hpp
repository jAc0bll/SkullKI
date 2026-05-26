#pragma once

#include "action.hpp"
#include "cards.hpp"

namespace sk {

// Unified action index used by the neural network heads.
//   [0..10]   : Bid 0..10                             (11 actions)
//   [11..80]  : Play card 0..69                       (70 actions)
//   [81]      : TigressMode pirate (asPirate = true)
//   [82]      : TigressMode escape (asPirate = false)
constexpr int ACTION_DIM = 11 + N_CARDS + 2;
static_assert(ACTION_DIM == 83, "ACTION_DIM layout invariant");

constexpr int actionToIndex(const Action& a) noexcept {
    switch (a.type) {
        case ActionType::Bid:         return static_cast<int>(a.bid);
        case ActionType::Play:        return 11 + static_cast<int>(a.card);
        case ActionType::TigressMode: return a.asPirate ? 81 : 82;
    }
    return -1;
}

constexpr Action indexToAction(int idx) noexcept {
    if (idx < 11)         return Action::makeBid(idx);
    if (idx < 11 + N_CARDS) return Action::makePlay(static_cast<Card>(idx - 11));
    if (idx == 81)        return Action::makeTigressMode(true);
    if (idx == 82)        return Action::makeTigressMode(false);
    return Action::makeBid(0);
}

} // namespace sk
