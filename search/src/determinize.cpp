#include "sk/determinize.hpp"
#include "sk/cards.hpp"

#include <algorithm>
#include <vector>

namespace sk {

namespace {

bool voidIn(const GameState& s, int p, Card c) {
    return isColored(c) && (s.voidSuits[p] >> static_cast<int>(suitOf(c))) & 1u;
}

// One attempt at a random deal that respects public void constraints.
// Cards are taken in random order; each goes to a random eligible opponent,
// weighted by that opponent's remaining capacity. Returns false on a dead end.
bool tryConstrainedDeal(GameState& d, const GameState& src, int perspective,
                        const std::vector<Card>& unseen, std::mt19937_64& rng)
{
    std::array<int, N_PLAYERS> need{};
    for (int p = 0; p < N_PLAYERS; ++p) {
        need[p] = (p == perspective) ? 0 : src.hands[p].count();
        if (p != perspective) d.hands[p].clear();
    }
    int remaining = 0;
    for (int p = 0; p < N_PLAYERS; ++p) remaining += need[p];

    for (Card c : unseen) {
        if (remaining == 0) break;
        int total = 0;
        for (int p = 0; p < N_PLAYERS; ++p)
            if (need[p] > 0 && !voidIn(src, p, c)) total += need[p];
        if (total == 0) continue;  // card stays out (unseen pool > hands)

        int pick = std::uniform_int_distribution<int>(0, total - 1)(rng);
        for (int p = 0; p < N_PLAYERS; ++p) {
            if (need[p] <= 0 || voidIn(src, p, c)) continue;
            if (pick < need[p]) {
                d.hands[p].add(c);
                --need[p];
                --remaining;
                break;
            }
            pick -= need[p];
        }
    }
    return remaining == 0;
}

} // namespace

GameState determinize(const GameState& src, int perspective, std::mt19937_64& rng) {
    GameState d = src;

    // 0. Bids are simultaneous: bids of others are hidden during Bidding.
    //    "Un-submit" them so the search re-samples them as decisions.
    if (src.phase == Phase::Bidding) {
        for (int p = 0; p < N_PLAYERS; ++p) {
            if (p != perspective && d.bids[p] >= 0) {
                d.bids[p] = -1;
                --d.bidsSubmitted;
            }
        }
    }

    // 1. Compute the set of cards already accounted for ("known"):
    //    - cards in perspective's hand
    //    - cards in any captured pile (won tricks of this round)
    //    - cards currently on the table in the in-progress trick
    CardSet known;
    src.hands[perspective].forEach([&](Card c) { known.add(c); });
    for (int p = 0; p < N_PLAYERS; ++p) {
        src.captured[p].forEach([&](Card c) { known.add(c); });
    }
    for (int i = 0; i < src.trickSize; ++i) {
        known.add(src.trickCards[i]);
    }

    // 2. Build the deck of unseen cards.
    std::vector<Card> unseen;
    unseen.reserve(N_CARDS);
    for (Card c = 0; c < N_CARDS; ++c) {
        if (!known.has(c)) unseen.push_back(c);
    }

    // 3. Re-deal opponents with their publicly known hand sizes, respecting
    //    publicly revealed voids. Not exactly uniform over consistent worlds
    //    (capacity-weighted sequential assignment). If every attempt dead-ends
    //    we fall back to ignoring voids — never to the true hands, which
    //    would leak hidden information.
    for (int attempt = 0; attempt < 64; ++attempt) {
        std::shuffle(unseen.begin(), unseen.end(), rng);
        if (tryConstrainedDeal(d, src, perspective, unseen, rng)) return d;
    }
    std::size_t idx = 0;
    for (int p = 0; p < N_PLAYERS; ++p) {
        if (p == perspective) continue;
        const int sz = src.hands[p].count();
        d.hands[p].clear();
        for (int i = 0; i < sz; ++i) d.hands[p].add(unseen[idx++]);
    }
    return d;
}

} // namespace sk
