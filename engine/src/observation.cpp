#include "sk/observation.hpp"

namespace sk {

Observation observe(const GameState& src, int perspective) {
    Observation o;
    o.s = src;
    o.perspective = static_cast<std::int8_t>(perspective);
    const bool bidding = (src.phase == Phase::Bidding);
    for (int p = 0; p < N_PLAYERS; ++p) {
        o.handSizes[p] = static_cast<std::int8_t>(src.hands[p].count());
        if (p != perspective) {
            o.s.hands[p].clear();
            // Bids are simultaneous: nobody sees another bid until all are in.
            if (bidding) o.s.bids[p] = -1;
        }
    }
    return o;
}

} // namespace sk
