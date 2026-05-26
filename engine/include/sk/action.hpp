#pragma once

#include "cards.hpp"
#include <cstdint>

namespace sk {

enum class ActionType : std::uint8_t {
    Bid,          // payload: bid value 0..roundNumber
    Play,         // payload: Card from hand
    TigressMode,  // payload: tigressAsPirate (true=Pirate, false=Escape)
};

struct Action {
    ActionType type;
    union {
        std::uint8_t bid;
        Card card;
        bool asPirate;
    };

    static constexpr Action makeBid(int v) {
        Action a{};
        a.type = ActionType::Bid;
        a.bid = static_cast<std::uint8_t>(v);
        return a;
    }
    static constexpr Action makePlay(Card c) {
        Action a{};
        a.type = ActionType::Play;
        a.card = c;
        return a;
    }
    static constexpr Action makeTigressMode(bool pirate) {
        Action a{};
        a.type = ActionType::TigressMode;
        a.asPirate = pirate;
        return a;
    }
};

constexpr bool operator==(const Action& a, const Action& b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ActionType::Bid:         return a.bid == b.bid;
        case ActionType::Play:        return a.card == b.card;
        case ActionType::TigressMode: return a.asPirate == b.asPirate;
    }
    return false;
}

constexpr bool operator!=(const Action& a, const Action& b) { return !(a == b); }

} // namespace sk
