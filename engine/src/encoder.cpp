#include "sk/encoder.hpp"
#include "sk/cards.hpp"

#include <array>
#include <cstddef>

namespace sk {

namespace {

inline void appendOneHot(std::vector<float>& v, int value, int size) {
    const std::size_t start = v.size();
    v.resize(start + size, 0.0f);
    if (value >= 0 && value < size) v[start + value] = 1.0f;
}

inline void appendCardSet(std::vector<float>& v, const CardSet& cs) {
    const std::size_t start = v.size();
    v.resize(start + N_CARDS, 0.0f);
    cs.forEach([&](Card c) { v[start + c] = 1.0f; });
}

} // namespace

void encodeInto(const Observation& o, std::vector<float>& out) {
    out.clear();
    out.reserve(ENC_DIM);

    // Own hand (70).
    appendCardSet(out, o.s.hands[o.perspective]);

    // Hand sizes / 10 (4).
    for (int p = 0; p < N_PLAYERS; ++p) {
        out.push_back(static_cast<float>(o.handSizes[p]) / 10.0f);
    }

    // Bids and validity (4 + 4).
    for (int p = 0; p < N_PLAYERS; ++p) {
        out.push_back(o.s.bids[p] >= 0 ? static_cast<float>(o.s.bids[p]) / 10.0f : -0.1f);
    }
    for (int p = 0; p < N_PLAYERS; ++p) {
        out.push_back(o.s.bids[p] >= 0 ? 1.0f : 0.0f);
    }

    // tricksWon / 10 (4).
    for (int p = 0; p < N_PLAYERS; ++p) {
        out.push_back(static_cast<float>(o.s.tricksWon[p]) / 10.0f);
    }

    // scores / 200 (4).
    for (int p = 0; p < N_PLAYERS; ++p) {
        out.push_back(static_cast<float>(o.s.scores[p]) / 200.0f);
    }

    // Round number one-hot (10).
    appendOneHot(out, o.s.roundNumber - 1, 10);

    // Phase one-hot (3).
    appendOneHot(out, static_cast<int>(o.s.phase), 3);

    // currentPlayer / startPlayer / trickLeader (3 × 4).
    appendOneHot(out, o.s.currentPlayer, N_PLAYERS);
    appendOneHot(out, o.s.startPlayer,   N_PLAYERS);
    appendOneHot(out, o.s.trickLeader,   N_PLAYERS);

    // leadSuit one-hot (5).
    appendOneHot(out, static_cast<int>(o.s.leadSuit), 5);

    // perspective one-hot (4).
    appendOneHot(out, o.perspective, N_PLAYERS);

    // Flags (4).
    out.push_back(o.s.freeTrick        ? 1.0f : 0.0f);
    out.push_back(o.s.pendingTigress   ? 1.0f : 0.0f);
    out.push_back(o.s.tigressAsPirate  ? 1.0f : 0.0f);
    out.push_back(static_cast<float>(o.s.bidsSubmitted) / 4.0f);

    // Trick progress (2).
    out.push_back(static_cast<float>(o.s.trickSize)    / 4.0f);
    out.push_back(static_cast<float>(o.s.tricksPlayed) / 10.0f);

    // Per-player current-trick card (4 × 70 = 280).
    std::array<Card, N_PLAYERS>   inTrickCard{};
    std::array<bool, N_PLAYERS>   inTrickFlag{};
    for (int i = 0; i < o.s.trickSize; ++i) {
        const int pl = o.s.trickPlayers[i];
        inTrickCard[pl] = o.s.trickCards[i];
        inTrickFlag[pl] = true;
    }
    for (int p = 0; p < N_PLAYERS; ++p) {
        const std::size_t start = out.size();
        out.resize(start + N_CARDS, 0.0f);
        if (inTrickFlag[p]) out[start + inTrickCard[p]] = 1.0f;
    }

    // Captured pile per player (4 × 70 = 280).
    for (int p = 0; p < N_PLAYERS; ++p) {
        appendCardSet(out, o.s.captured[p]);
    }

    // Union of all cards played this round (70).
    const std::size_t playedStart = out.size();
    out.resize(playedStart + N_CARDS, 0.0f);
    for (int p = 0; p < N_PLAYERS; ++p) {
        o.s.captured[p].forEach([&](Card c) { out[playedStart + c] = 1.0f; });
    }
    for (int i = 0; i < o.s.trickSize; ++i) {
        out[playedStart + o.s.trickCards[i]] = 1.0f;
    }
}

std::vector<float> encode(const Observation& o) {
    std::vector<float> v;
    encodeInto(o, v);
    return v;
}

} // namespace sk
