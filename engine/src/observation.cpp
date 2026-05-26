#include "sk/observation.hpp"

namespace sk {

Observation observe(const GameState& src, int perspective) {
    Observation o;
    o.s = src;
    o.perspective = static_cast<std::int8_t>(perspective);
    for (int p = 0; p < N_PLAYERS; ++p) {
        o.handSizes[p] = static_cast<std::int8_t>(src.hands[p].count());
        if (p != perspective) {
            o.s.hands[p].clear();
        }
    }
    return o;
}

} // namespace sk
