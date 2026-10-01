#include "sk/solver/abstraction.hpp"

#include <algorithm>

namespace sk::solver {

// ---------------------------------------------------------------------------
// Lossy feature abstraction (level 1).
//
// Bidding: seat + the hand described card by card with rank buckets (finer at
// the top, where a single step changes who wins) and plain-suit length.
//
// Playing: the key holds what drives the decision, not card identities:
//   - trick position, tricks left
//   - every player's bid status (needs k / made / over / zero / hopeless),
//     in trick order from the leader
//   - the trick so far: lead type, who is winning with what class of card,
//     bonus points riding on it (14s, pirates for a Skull King, mermaid, SK)
//   - unseen specials and trumps
//   - public voids of the players still to act
//   - own hand as card descriptors: suit relation (led / trump / other),
//     unseen higher cards of that suit, suit length, 14 or not, and whether
//     the card would currently win the trick.
// Cards with identical descriptors are merged into one action slot.
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint8_t KEY_BIDDING = 0;
constexpr std::uint8_t KEY_PLAYING = 1;
constexpr std::uint8_t KEY_TIGRESS = 2;

int clampi(int v, int hi) { return v < hi ? v : hi; }

// Rounds up to this size describe the whole hand card by card; bigger rounds
// use count summaries (otherwise almost every situation is unique and
// nothing is pooled).
constexpr int FULL_HAND_MAX_ROUND = 3;

CardSet playedCards(const GameState& s) {
    CardSet played;
    for (int q = 0; q < N_PLAYERS; ++q) s.captured[q].forEach([&](Card c) { played.add(c); });
    for (int i = 0; i < s.trickSize; ++i) played.add(s.trickCards[i]);
    return played;
}

int unseenHigher(Card c, const CardSet& unseen) {
    const Suit su = suitOf(c);
    int n = 0;
    for (int v = valueOf(c) + 1; v <= CARDS_PER_SUIT; ++v)
        n += unseen.has(makeColored(su, v));
    return n;
}

int countInSuit(const CardSet& hand, Suit su) {
    int n = 0;
    hand.forEach([&](Card c) { n += isColored(c) && suitOf(c) == su; });
    return n;
}

std::uint8_t specialType(Card c) {
    if (isEscape(c))  return 0;
    if (isMermaid(c)) return 1;
    if (isPirate(c))  return 2;
    if (isTigress(c)) return 3;
    return 4;   // Skull King
}

// Bid status of player q: 1..3 = still needs that many, 10 zero-bid intact,
// 11 zero-bid busted, 12 made exactly, 13 over, 14 cannot make it any more.
std::uint8_t bidStatus(const GameState& s, int q, int tricksLeft) {
    const int bid = s.bids[q], won = s.tricksWon[q];
    if (bid == 0) return won == 0 ? 10 : 11;
    const int need = bid - won;
    if (need < 0)  return 13;
    if (need == 0) return 12;
    if (need > tricksLeft) return 14;
    return static_cast<std::uint8_t>(clampi(need, 3));
}

// ---- bidding -------------------------------------------------------------
std::uint8_t trumpBucket(int v) {
    if (v >= 10) return static_cast<std::uint8_t>(v - 7);   // 10..14 -> 3..7
    if (v >= 8)  return 2;
    if (v >= 5)  return 1;
    return 0;
}
std::uint8_t plainBucket(int v) {
    if (v >= 12) return static_cast<std::uint8_t>(v - 9);   // 12..14 -> 3..5
    if (v >= 10) return 2;
    if (v >= 7)  return 1;
    return 0;
}

// Hand summary for bidding in big rounds: counts per strength class.
void appendBiddingSummary(const CardSet& hand, InfoKey& key) {
    int sk = 0, pir = 0, mer = 0, esc = 0, tHigh = 0, tMid = 0, tLow = 0, pTop = 0, pMid = 0;
    std::array<int, 3> plainLen{};
    hand.forEach([&](Card c) {
        if (isSkullKing(c)) ++sk;
        else if (isPirate(c) || isTigress(c)) ++pir;
        else if (isMermaid(c)) ++mer;
        else if (isEscape(c)) ++esc;
        else if (isTrump(c)) {
            const int v = valueOf(c);
            if (v >= 12) ++tHigh; else if (v >= 8) ++tMid; else ++tLow;
        } else {
            const int v = valueOf(c);
            if (v >= 13) ++pTop; else if (v >= 10) ++pMid;
            ++plainLen[static_cast<int>(suitOf(c))];
        }
    });
    std::sort(plainLen.begin(), plainLen.end());
    int voids = 0;
    for (int l : plainLen) voids += l == 0;
    const int longest = plainLen[2];
    const int vals[] = {sk, clampi(pir, 3), mer, clampi(esc, 3), tHigh, clampi(tMid, 3),
                        clampi(tLow, 3), clampi(pTop, 3), clampi(pMid, 3),
                        longest <= 2 ? 0 : clampi(longest - 2, 3), voids};
    for (int v : vals) key.push_back(static_cast<char>(v));
}

void biddingView(const RoundState& rs, int p, const ActionList& legal, InfosetView& out) {
    const CardSet& hand = rs.s.hands[p];
    if (rs.round > FULL_HAND_MAX_ROUND) {
        out.key.clear();
        out.key.push_back(static_cast<char>(rs.round));
        out.key.push_back(static_cast<char>(KEY_BIDDING));
        out.key.push_back(static_cast<char>(p));
        appendBiddingSummary(hand, out.key);
        out.nSlots = legal.n;
        for (int a = 0; a < legal.n; ++a) out.slot[a] = static_cast<std::uint8_t>(a);
        out.symmetric = false;
        return;
    }
    std::array<std::uint8_t, MAX_ROUND> codes{};
    int n = 0;
    hand.forEach([&](Card c) {
        std::uint8_t code;
        if (isSpecial(c))      code = static_cast<std::uint8_t>(200 + specialType(c));
        else if (isTrump(c))   code = static_cast<std::uint8_t>(100 + trumpBucket(valueOf(c)));
        else                   code = static_cast<std::uint8_t>(
                                   plainBucket(valueOf(c)) * 4 +
                                   clampi(countInSuit(hand, suitOf(c)), 4) - 1);
        codes[n++] = code;
    });
    std::sort(codes.begin(), codes.begin() + n);
    out.key.clear();
    out.key.push_back(static_cast<char>(rs.round));
    out.key.push_back(static_cast<char>(KEY_BIDDING));
    out.key.push_back(static_cast<char>(p));   // seat; seat 0 leads
    out.key.append(reinterpret_cast<const char*>(codes.data()), n);
    out.nSlots = legal.n;
    for (int a = 0; a < legal.n; ++a) out.slot[a] = static_cast<std::uint8_t>(a);
    out.symmetric = false;
}

// ---- playing -------------------------------------------------------------
struct TrickView {
    Card         cards[N_PLAYERS];
    std::int8_t  players[N_PLAYERS];
    int          n = 0;
    bool         tigressAsPirate = false;
};

bool wouldWin(const TrickView& t, Card c, int p) {
    if (t.n == 0) return false;   // leading: "beats the trick" is meaningless
    Card cards[N_PLAYERS];
    std::int8_t players[N_PLAYERS];
    for (int i = 0; i < t.n; ++i) { cards[i] = t.cards[i]; players[i] = t.players[i]; }
    cards[t.n] = c;
    players[t.n] = static_cast<std::int8_t>(p);
    const bool tig = isTigress(c) ? true : t.tigressAsPirate;
    return resolveTrick(cards, players, t.n + 1, tig).winner == p;
}

std::uint8_t cardClass(Card c, bool tigressAsPirate, const CardSet& unseen) {
    if (isEscape(c))    return 1;
    if (isTigress(c))   return tigressAsPirate ? 3 : 1;
    if (isPirate(c))    return 3;
    if (isMermaid(c))   return 4;
    if (isSkullKing(c)) return 5;
    const int hi = clampi(unseenHigher(c, unseen), 3);
    return static_cast<std::uint8_t>((isTrump(c) ? 20 : 30) + hi);
}

std::uint8_t playCode(Card c, const CardSet& hand, const GameState& s, const TrickView& t,
                      const CardSet& unseen, int p)
{
    const int beats = wouldWin(t, c, p) ? 1 : 0;
    if (isSpecial(c)) return static_cast<std::uint8_t>(200 + specialType(c) * 2 + beats);
    const Suit su = suitOf(c);
    const int suitClass = (s.leadSuit != Suit::None && su == s.leadSuit) ? 0 : isTrump(c) ? 1 : 2;
    const int hi  = clampi(unseenHigher(c, unseen), 2);
    const int len = clampi(countInSuit(hand, su), 3) - 1;
    return static_cast<std::uint8_t>(suitClass * 36 + hi * 12 + len * 4 + is14(c) * 2 + beats);
}

void playingView(const RoundState& rs, int p, const ActionList& legal, InfosetView& out) {
    const GameState& s = rs.s;
    const bool tigressPending = s.pendingTigress;

    // The trick as the decision sees it (a pending Tigress is p's own card).
    TrickView t;
    t.n = s.trickSize - (tigressPending ? 1 : 0);
    for (int i = 0; i < t.n; ++i) { t.cards[i] = s.trickCards[i]; t.players[i] = s.trickPlayers[i]; }
    t.tigressAsPirate = s.tigressAsPirate;

    const CardSet& hand = s.hands[p];
    CardSet unseen;
    {
        const CardSet played = playedCards(s);
        for (Card c = 0; c < N_CARDS; ++c)
            if (!played.has(c) && !hand.has(c)) unseen.add(c);
    }
    const int tricksLeft = s.roundNumber - s.tricksPlayed;
    const int leader = s.trickLeader;

    out.key.clear();
    auto put = [&](int v) { out.key.push_back(static_cast<char>(v)); };
    put(rs.round);
    put(tigressPending ? KEY_TIGRESS : KEY_PLAYING);
    put(t.n);            // my position in the trick
    put(tricksLeft);
    for (int i = 0; i < N_PLAYERS; ++i) put(bidStatus(s, (leader + i) % N_PLAYERS, tricksLeft));

    // Trick so far.
    int leadClass = 0;   // 0: I lead
    if (t.n > 0) {
        if (s.freeTrick)                     leadClass = 1;   // character led
        else if (s.leadSuit == Suit::None)   leadClass = 2;   // escapes only so far
        else if (s.leadSuit == Suit::Black)  leadClass = 4;
        else                                 leadClass = 3;
    }
    put(leadClass);
    int winPos = 9, winClass = 0, bonus14 = 0, pirates = 0, mermaid = 0, sk = 0;
    if (t.n > 0) {
        const TrickResult r = resolveTrick(t.cards, t.players, t.n, t.tigressAsPirate);
        for (int i = 0; i < t.n; ++i) {
            const Card c = t.cards[i];
            if (t.players[i] == r.winner) {
                winPos = i;
                winClass = cardClass(c, t.tigressAsPirate, unseen);
            }
            if (is14(c)) bonus14 += isTrump(c) ? 2 : 1;
            if (isPirate(c) || (isTigress(c) && t.tigressAsPirate)) ++pirates;
            mermaid |= isMermaid(c);
            sk |= isSkullKing(c);
        }
    }
    put(winPos); put(winClass); put(clampi(bonus14, 5)); put(clampi(pirates, 3)); put(mermaid); put(sk);

    // Unseen specials and trumps.
    int uPir = 0, uSK = 0, uMer = 0, uEsc = 0, uTrump = 0;
    unseen.forEach([&](Card c) {
        uPir += isPirate(c) || isTigress(c);
        uSK  += isSkullKing(c);
        uMer += isMermaid(c);
        uEsc += isEscape(c);
        uTrump += isColored(c) && isTrump(c);
    });
    put(uPir == 0 ? 0 : uPir <= 2 ? 1 : 2); put(uSK); put(clampi(uMer, 1));
    put(uEsc == 0 ? 0 : 1);
    put(uTrump == 0 ? 0 : uTrump <= 3 ? 1 : 2);

    // Public voids (led suit, trump) of the players still to act this trick.
    int voids = 0;
    for (int i = t.n + 1; i < N_PLAYERS; ++i) {
        const int q = (leader + i) % N_PLAYERS;
        const int ledVoid = (s.leadSuit != Suit::None) &&
                            ((s.voidSuits[q] >> static_cast<int>(s.leadSuit)) & 1u);
        const int trumpVoid = (s.voidSuits[q] >> static_cast<int>(Suit::Black)) & 1u;
        voids = voids * 4 + ledVoid * 2 + trumpVoid;
    }
    put(voids);

    // Own hand: small rounds card by card; big rounds a strength summary
    // plus descriptors of the legal cards only.
    put(255);   // separator
    std::array<std::uint8_t, ActionList::CAPACITY> ac{};
    if (!tigressPending)
        for (int a = 0; a < legal.n; ++a) ac[a] = playCode(legal[a].card, hand, s, t, unseen, p);
    if (rs.round <= FULL_HAND_MAX_ROUND) {
        std::array<std::uint8_t, MAX_ROUND> codes{};
        int nh = 0;
        hand.forEach([&](Card c) { codes[nh++] = playCode(c, hand, s, t, unseen, p); });
        std::sort(codes.begin(), codes.begin() + nh);
        out.key.append(reinterpret_cast<const char*>(codes.data()), nh);
    } else {
        int winSp = 0, mer = 0, esc = 0, tTop = 0, tOther = 0, pHigh = 0, pLow = 0;
        hand.forEach([&](Card c) {
            if (isSkullKing(c) || isPirate(c) || isTigress(c)) ++winSp;
            else if (isMermaid(c)) ++mer;
            else if (isEscape(c)) ++esc;
            else if (isTrump(c)) (unseenHigher(c, unseen) == 0 ? tTop : tOther)++;
            else (unseenHigher(c, unseen) <= 1 ? pHigh : pLow)++;
        });
        put(clampi(winSp, 3)); put(mer); put(clampi(esc, 2));
        put(clampi(tTop, 2)); put(clampi(tOther, 2)); put(clampi(pHigh, 2)); put(clampi(pLow, 3));
        if (!tigressPending) {
            std::array<std::uint8_t, ActionList::CAPACITY> d = ac;
            std::sort(d.begin(), d.begin() + legal.n);
            const int nd = static_cast<int>(std::unique(d.begin(), d.begin() + legal.n) - d.begin());
            put(254);
            out.key.append(reinterpret_cast<const char*>(d.data()), nd);
        }
    }
    out.symmetric = false;

    if (tigressPending) {
        out.nSlots = legal.n;
        for (int a = 0; a < legal.n; ++a) out.slot[a] = static_cast<std::uint8_t>(a);
        return;
    }
    // Actions: one slot per distinct card descriptor, in descriptor order.
    std::array<std::uint8_t, ActionList::CAPACITY> distinct = ac;
    std::sort(distinct.begin(), distinct.begin() + legal.n);
    const int nd = static_cast<int>(std::unique(distinct.begin(), distinct.begin() + legal.n) -
                                    distinct.begin());
    out.nSlots = nd;
    for (int a = 0; a < legal.n; ++a)
        out.slot[a] = static_cast<std::uint8_t>(
            std::lower_bound(distinct.begin(), distinct.begin() + nd, ac[a]) - distinct.begin());
}

} // namespace

void FeatureAbstraction::view(const RoundState& rs, int player, const ActionList& legal,
                              InfosetView& out) const
{
    if (rs.s.phase == Phase::Bidding) biddingView(rs, player, legal, out);
    else                              playingView(rs, player, legal, out);
}

} // namespace sk::solver
