#include "sk/determinize.hpp"
#include "sk/cards.hpp"

#include <algorithm>
#include <vector>

namespace sk {

GameState determinize(const GameState& src, int perspective, std::mt19937_64& rng) {
    GameState d = src;

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
    std::shuffle(unseen.begin(), unseen.end(), rng);

    // 3. Re-deal opponents. Each opponent's hand size is publicly known
    //    (equal to roundNumber minus number of cards they've already played).
    //    We use src.hands[p].count() which is the true count — and the count
    //    is itself a public quantity, so this leaks nothing.
    std::size_t idx = 0;
    for (int p = 0; p < N_PLAYERS; ++p) {
        if (p == perspective) continue;
        const int sz = src.hands[p].count();
        d.hands[p].clear();
        for (int i = 0; i < sz; ++i) {
            d.hands[p].add(unseen[idx++]);
        }
    }
    return d;
}

} // namespace sk
