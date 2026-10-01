#include "sk/solver/encoding.hpp"

#include <cstring>

namespace sk::solver {

int actionIndex(const Action& a) {
    switch (a.type) {
        case ActionType::Bid:         return a.bid;
        case ActionType::Play:        return 11 + kindOf(a.card);
        case ActionType::TigressMode: return a.asPirate ? 72 : 73;
    }
    return -1;
}

void legalMask(const ActionList& legal, std::uint8_t* out) {
    std::memset(out, 0, ACT_DIM);
    for (int i = 0; i < legal.n; ++i) out[actionIndex(legal[i])] = 1;
}

namespace {

void addKinds(const CardSet& cs, std::uint8_t* out) {
    cs.forEach([&](Card c) { ++out[kindOf(c)]; });
}

int rel(int p, int me) { return (p - me + N_PLAYERS) % N_PLAYERS; }

} // namespace

void encodeInfoset(const RoundState& rs, int me, std::uint8_t* out) {
    const GameState& s = rs.s;
    std::memset(out, 0, INFO_DIM);
    std::uint8_t* o = out;

    o[rs.round - 1] = 1;                                    o += 10;
    const bool bidding = s.phase == Phase::Bidding;
    o[bidding ? 0 : s.pendingTigress ? 2 : 1] = 1;          o += 3;
    o[rel(me, s.startPlayer)] = 1;                          o += 4;   // seat from start
    const int myPos = bidding ? 0 : rel(me, s.trickLeader);
    o[myPos] = 1;                                           o += 4;
    o[0] = s.tricksPlayed;
    o[1] = static_cast<std::uint8_t>(s.roundNumber - s.tricksPlayed);
                                                            o += 2;
    addKinds(s.hands[me], o);                               o += 61;

    CardSet completed;
    for (int q = 0; q < N_PLAYERS; ++q) s.captured[q].forEach([&](Card c) { completed.add(c); });
    addKinds(completed, o);                                 o += 61;

    if (!bidding) {
        for (int i = 0; i < s.trickSize; ++i) {
            const int pos = rel(s.trickPlayers[i], s.trickLeader);
            o[pos * 62 + kindOf(s.trickCards[i])] = 1;
            o[pos * 62 + 61] = 1;
        }
    }
    o += 4 * 62;
    bool tigressInTrick = false;
    for (int i = 0; i < s.trickSize; ++i) tigressInTrick |= isTigress(s.trickCards[i]);
    if (tigressInTrick && !s.pendingTigress) o[s.tigressAsPirate ? 0 : 1] = 1;
    o += 2;

    // Per player, relative to me. Other players' bids are hidden while bidding.
    for (int r = 0; r < N_PLAYERS; ++r) {
        const int q = (me + r) % N_PLAYERS;
        const bool known = s.bids[q] >= 0 && (!bidding || q == me);
        o[r * 12 + (known ? s.bids[q] : 11)] = 1;
    }
    o += 4 * 12;
    for (int r = 0; r < N_PLAYERS; ++r) o[r] = s.tricksWon[(me + r) % N_PLAYERS];
    o += 4;
    for (int r = 0; r < N_PLAYERS; ++r) {
        const int q = (me + r) % N_PLAYERS;
        const bool known = s.bids[q] >= 0 && (!bidding || q == me);
        o[r] = known ? static_cast<std::uint8_t>(10 + s.bids[q] - s.tricksWon[q]) : 0;
    }
    o += 4;
    for (int r = 0; r < N_PLAYERS; ++r)
        for (int su = 0; su < 4; ++su)
            o[r * 4 + su] = (s.voidSuits[(me + r) % N_PLAYERS] >> su) & 1u;
    o += 16;
    if (!bidding) o[rel(s.trickLeader, me)] = 1;
    o += 4;

    // Unseen = everything not in my hand, not played, not on the table.
    std::uint8_t* unseen = o;
    for (int k = 0; k < N_KINDS; ++k) unseen[k] = static_cast<std::uint8_t>(kindMultiplicity(static_cast<Kind>(k)));
    s.hands[me].forEach([&](Card c) { --unseen[kindOf(c)]; });
    completed.forEach([&](Card c) { --unseen[kindOf(c)]; });
    for (int i = 0; i < s.trickSize; ++i) --unseen[kindOf(s.trickCards[i])];
    o += 61;

    // Would each card in my hand win the trick if played now?
    if (!bidding && !s.pendingTigress && s.trickSize > 0) {
        Card cards[N_PLAYERS];
        std::int8_t players[N_PLAYERS];
        for (int i = 0; i < s.trickSize; ++i) { cards[i] = s.trickCards[i]; players[i] = s.trickPlayers[i]; }
        s.hands[me].forEach([&](Card c) {
            cards[s.trickSize] = c;
            players[s.trickSize] = static_cast<std::int8_t>(me);
            const bool tig = isTigress(c) ? true : s.tigressAsPirate;
            if (resolveTrick(cards, players, s.trickSize + 1, tig).winner == me) o[kindOf(c)] = 1;
        });
    }
    o += 61;
}

void encodeHistory(const RoundState& rs, int perspective, std::uint8_t* out) {
    encodeInfoset(rs, perspective, out);
    std::uint8_t* o = out + INFO_DIM;
    std::memset(o, 0, 3 * 61);
    for (int r = 1; r < N_PLAYERS; ++r)
        addKinds(rs.s.hands[(perspective + r) % N_PLAYERS], o + (r - 1) * 61);
}

} // namespace sk::solver
