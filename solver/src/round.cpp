#include "sk/solver/round.hpp"

#include <cassert>

namespace sk::solver {

RoundState makeRoundState(int round, const std::array<CardSet, N_PLAYERS>& hands) {
    RoundState rs;
    rs.s = initialState(0);
    rs.s.roundNumber = static_cast<std::uint8_t>(round);
    for (int p = 0; p < N_PLAYERS; ++p) rs.s.hands[p] = hands[p];
    rs.round = static_cast<std::uint8_t>(round);
    return rs;
}

void applyRound(RoundState& rs, Action a) {
    switch (a.type) {
        case ActionType::Bid:
            break;
        case ActionType::Play:
            assert(rs.pubLen < rs.pub.size());
            rs.pub[rs.pubLen++] = kindOf(a.card);
            break;
        case ActionType::TigressMode:
            assert(rs.pubLen < rs.pub.size());
            rs.pub[rs.pubLen++] = a.asPirate ? PUB_TIGRESS_PIRATE : PUB_TIGRESS_ESCAPE;
            break;
    }
    applyAction(rs.s, a);
}

void legalKindActions(const GameState& s, ActionList& out) {
    ActionList all;
    legalActionsInto(s, all);
    out.n = 0;
    // Cards come in ascending id order and ids of a kind are contiguous,
    // so duplicates of a kind are always adjacent.
    int lastKind = -1;
    for (int i = 0; i < all.n; ++i) {
        const Action& a = all[i];
        if (a.type == ActionType::Play) {
            const int k = kindOf(a.card);
            if (k == lastKind) continue;
            lastKind = k;
        }
        out.push(a);
    }
}

std::array<double, N_PLAYERS> roundUtilities(const RoundState& rs, Utility u) {
    // Scores start at 0 in makeRoundState, so s.scores are the round points.
    std::array<double, N_PLAYERS> out{};
    double total = 0.0;
    for (int p = 0; p < N_PLAYERS; ++p) total += rs.s.scores[p];
    for (int p = 0; p < N_PLAYERS; ++p) {
        const double own = rs.s.scores[p];
        out[p] = (u == Utility::Absolute)
            ? own
            : own - (total - own) / (N_PLAYERS - 1);
    }
    return out;
}

namespace {

void appendHand(const CardSet& hand, InfoKey& out) {
    const std::size_t lenPos = out.size();
    out.push_back(0);
    std::uint8_t n = 0;
    hand.forEach([&](Card c) {
        out.push_back(static_cast<char>(kindOf(c)));  // ascending: kinds are monotone in ids
        ++n;
    });
    out[lenPos] = static_cast<char>(n);
}

} // namespace

void biddingKey(int round, int seat, const CardSet& hand, InfoKey& out) {
    out.clear();
    out.push_back(static_cast<char>(round));
    out.push_back(static_cast<char>(seat));
    out.push_back(0);  // phase: bidding
    appendHand(hand, out);
}

void infosetKey(const RoundState& rs, int player, InfoKey& out) {
    if (rs.s.phase == Phase::Bidding) {
        biddingKey(rs.round, player, rs.s.hands[player], out);
        return;
    }
    out.clear();
    out.push_back(static_cast<char>(rs.round));
    out.push_back(static_cast<char>(player));
    out.push_back(1);  // phase: playing
    appendHand(rs.s.hands[player], out);
    for (int p = 0; p < N_PLAYERS; ++p) out.push_back(static_cast<char>(rs.s.bids[p]));
    out.append(reinterpret_cast<const char*>(rs.pub.data()), rs.pubLen);
}

std::string describeAction(const Action& a) {
    switch (a.type) {
        case ActionType::Bid:         return "bid " + std::to_string(a.bid);
        case ActionType::Play:        return kindName(kindOf(a.card));
        case ActionType::TigressMode: return a.asPirate ? "Tigress=Pirate" : "Tigress=Escape";
    }
    return "?";
}

std::string describeKey(const InfoKey& key) {
    auto at = [&](std::size_t i) { return static_cast<std::uint8_t>(key[i]); };
    std::string s = "R" + std::to_string(at(0)) + " seat" + std::to_string(at(1));
    const bool playing = at(2) != 0;
    const int n = at(3);
    s += " hand[";
    for (int i = 0; i < n; ++i) {
        if (i) s += ' ';
        s += kindName(at(4 + i));
    }
    s += "]";
    if (!playing) return s + " (bidding)";

    std::size_t i = 4 + n;
    s += " bids[";
    for (int p = 0; p < N_PLAYERS; ++p, ++i) {
        if (p) s += ' ';
        s += std::to_string(static_cast<int>(static_cast<std::int8_t>(key[i])));
    }
    s += "] play:";
    for (; i < key.size(); ++i) {
        const std::uint8_t b = at(i);
        s += ' ';
        if (b == PUB_TIGRESS_PIRATE)      s += "(as Pirate)";
        else if (b == PUB_TIGRESS_ESCAPE) s += "(as Escape)";
        else                              s += kindName(b);
    }
    return s;
}

} // namespace sk::solver
