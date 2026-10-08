#include "sk/solver/spot.hpp"

#include "sk/solver/encoding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace sk::solver {

namespace {

// User-facing messages are German: the web app shows them as they are.
const char* SUIT_DE[4] = {"Gelb", "Grün", "Lila", "Schwarz"};

std::string seatName(int p) { return "Sitz " + std::to_string(p + 1); }

std::string kindDe(int k) {
    if (k < N_COLORED) return std::string(SUIT_DE[k / CARDS_PER_SUIT]) + " " + std::to_string(k % CARDS_PER_SUIT + 1);
    switch (k) {
        case KIND_ESCAPE:  return "Flucht";
        case KIND_MERMAID: return "Meerjungfrau";
        case KIND_PIRATE:  return "Pirat";
        case KIND_TIGRESS: return "Tigress";
        default:           return "Skull King";
    }
}

bool isMarker(int x) { return x == PUB_TIGRESS_PIRATE || x == PUB_TIGRESS_ESCAPE; }

// Next unused physical card of a kind, or -1.
int allocate(int kind, std::array<bool, N_CARDS>& used) {
    const Card first = firstCardOfKind(static_cast<Kind>(kind));
    for (int i = 0; i < kindMultiplicity(static_cast<Kind>(kind)); ++i)
        if (!used[first + i]) {
            used[first + i] = true;
            return first + i;
        }
    return -1;
}

// Lowest card of `kind` in `hand`, or -1.
int cardOfKind(const CardSet& hand, int kind) {
    const Card first = firstCardOfKind(static_cast<Kind>(kind));
    for (int i = 0; i < kindMultiplicity(static_cast<Kind>(kind)); ++i)
        if (hand.has(static_cast<Card>(first + i))) return first + i;
    return -1;
}

bool legalKind(const GameState& s, int kind) {
    ActionList legal;
    legalKindActions(s, legal);
    for (int i = 0; i < legal.n; ++i)
        if (legal[i].type == ActionType::Play && kindOf(legal[i].card) == kind) return true;
    return false;
}

// Softmax over the legal actions' logits (NetPolicy, NetMode::Softmax).
void policy(const MLP& net, const RoundState& rs, int me, const ActionList& legal, double* out) {
    std::uint8_t x[INFO_DIM];
    float o[ACT_DIM];
    encodeInfoset(rs, me, x);
    net.forward(x, o);
    float mx = -1e30f;
    for (int a = 0; a < legal.n; ++a) mx = std::max(mx, o[actionIndex(legal[a])]);
    double sum = 0.0;
    for (int a = 0; a < legal.n; ++a) {
        out[a] = std::exp(static_cast<double>(o[actionIndex(legal[a])] - mx));
        sum += out[a];
    }
    for (int a = 0; a < legal.n; ++a) out[a] /= sum;
}

struct Trick {
    int leader = 0;
    std::vector<std::pair<int, Card>> cards;   // (seat, physical card)
    int tigress = -1;                          // -1 none/pending, 0 escape, 1 pirate
    int winner = -1;
};

std::string error(const std::string& msg) {
    std::string s = "{\"ok\":false,\"error\":\"";
    for (char c : msg) {
        if (c == '"' || c == '\\') s += '\\';
        s += c;
    }
    return s + "\"}";
}

void appendInts(std::ostringstream& o, const int* v, int n) {
    o << '[';
    for (int i = 0; i < n; ++i) o << (i ? "," : "") << v[i];
    o << ']';
}

std::string prob(double p) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4f", p);
    return buf;
}

} // namespace

bool parseSpot(const std::string& text, SpotInput& in, std::string& err) {
    std::istringstream words(text);
    std::string w;
    auto ints = [&](const std::string& v, std::vector<int>& out) {
        out.clear();
        std::istringstream items(v);
        std::string item;
        while (std::getline(items, item, ',')) {
            if (item.empty()) continue;
            try { out.push_back(std::stoi(item)); } catch (...) { return false; }
        }
        return true;
    };
    while (words >> w) {
        const auto eq = w.find('=');
        if (eq == std::string::npos) { err = "bad token: " + w; return false; }
        const std::string key = w.substr(0, eq), val = w.substr(eq + 1);
        std::vector<int> v;
        if (!ints(val, v)) { err = "bad number in: " + w; return false; }
        if (key == "round" && v.size() == 1)      in.round = v[0];
        else if (key == "me" && v.size() == 1)    in.me = v[0];
        else if (key == "hand")                   in.hand = v;
        else if (key == "play")                   in.play = v;
        else if (key == "bids" && v.size() == N_PLAYERS)
            for (int p = 0; p < N_PLAYERS; ++p) in.bids[p] = v[p];
        else { err = "bad token: " + w; return false; }
    }
    return true;
}

// JSON fields:
//   ok, round, me, phase ("bidding" | "playing" | "done"), toAct, pendingTigress,
//   bids[4], won[4], hand[kinds], unseen[61 counts], voids[4 suit bitmasks],
//   tricks[{leader, cards:[[seat,kind]], tigress: null|"pirate"|"escape", winner}]
//     (the last one is the current trick while it is incomplete, winner -1),
//   bidAdvice[{bid, p}]              my bidding strategy (always, if a net is given)
//   options[{type, value, p?, n?}]   the choices of the player to act:
//     me:        type "bid"/"card"/"tigress" with my strategy's probability p
//     opponent:  type "card" with n = copies still unseen, or "tigress"
//   points[4]                        round points, once the round is over
std::string spotQuery(const SpotInput& in, const MLP* net) {
    if (in.round < 1 || in.round > MAX_ROUND) return error("Runde muss zwischen 1 und 10 liegen");
    if (in.me < 0 || in.me >= N_PLAYERS) return error("Sitz muss zwischen 1 und 4 liegen");
    if (static_cast<int>(in.hand.size()) != in.round)
        return error("Deine Hand braucht genau " + std::to_string(in.round) + " Karten (gerade " +
                     std::to_string(in.hand.size()) + ")");

    std::array<bool, N_CARDS> used{};
    std::array<CardSet, N_PLAYERS> hands{};
    for (int k : in.hand) {
        if (k < 0 || k >= N_KINDS) return error("Unbekannte Karte in deiner Hand");
        const int c = allocate(k, used);
        if (c < 0) return error("Zu oft auf der Hand: " + kindDe(k));
        hands[in.me].add(static_cast<Card>(c));
    }
    bool allBids = true;
    for (int p = 0; p < N_PLAYERS; ++p) {
        if (in.bids[p] < -1 || in.bids[p] > in.round)
            return error("Ansage von " + seatName(p) + " muss zwischen 0 und " + std::to_string(in.round) + " liegen");
        allBids &= in.bids[p] >= 0;
    }
    if (!allBids && !in.play.empty()) return error("Erst alle vier Ansagen eintragen, dann Karten");

    const RoundState start = makeRoundState(in.round, hands);
    std::ostringstream o;
    o << "{\"ok\":true,\"round\":" << in.round << ",\"me\":" << in.me;

    // My bidding strategy (it depends on my hand and seat only).
    ActionList bidLegal;
    {
        RoundState bidding = start;
        bidding.s.currentPlayer = static_cast<std::int8_t>(in.me);
        legalKindActions(bidding.s, bidLegal);
        if (net) {
            double p[ActionList::CAPACITY];
            policy(*net, bidding, in.me, bidLegal, p);
            o << ",\"bidAdvice\":[";
            for (int a = 0; a < bidLegal.n; ++a)
                o << (a ? "," : "") << "{\"bid\":" << int(bidLegal[a].bid) << ",\"p\":" << prob(p[a]) << '}';
            o << ']';
        }
    }

    if (!allBids) {
        o << ",\"phase\":\"bidding\",\"toAct\":" << in.me << ",\"pendingTigress\":false,\"bids\":";
        appendInts(o, in.bids, N_PLAYERS);
        o << ",\"options\":[";
        for (int a = 0; a < bidLegal.n; ++a)
            o << (a ? "," : "") << "{\"type\":\"bid\",\"value\":" << int(bidLegal[a].bid) << '}';
        o << "]}";
        return o.str();
    }

    // Pass 1: who plays each logged card (follows from turn order and trick
    // winners). Opponents get each card just before they play it.
    std::vector<int> by(in.play.size(), -1);
    std::vector<Card> phys(in.play.size(), 0);
    {
        RoundState rs = start;
        for (int p = 0; p < N_PLAYERS; ++p) applyRound(rs, Action::makeBid(in.bids[p]));
        for (std::size_t i = 0; i < in.play.size(); ++i) {
            const int x = in.play[i];
            if (rs.terminal()) return error("Mehr Karten eingetragen, als die Runde hat");
            const int p = rs.s.currentPlayer;
            by[i] = p;
            if (rs.s.pendingTigress) {
                if (!isMarker(x)) return error("Nach der Tigress fehlt: als Pirat oder als Flucht?");
                applyRound(rs, Action::makeTigressMode(x == PUB_TIGRESS_PIRATE));
                continue;
            }
            if (isMarker(x) || x < 0 || x >= N_KINDS) return error("Ungültiger Eintrag im Spielverlauf");
            int c;
            if (p == in.me) {
                c = cardOfKind(rs.s.hands[p], x);
                if (c < 0) return error("Du hast " + kindDe(x) + " nicht (mehr) auf der Hand");
                if (!legalKind(rs.s, x)) return error(kindDe(x) + " ist nicht erlaubt: du musst Farbe bedienen");
            } else {
                if (x < N_COLORED && ((rs.s.voidSuits[p] >> (x / CARDS_PER_SUIT)) & 1u))
                    return error(seatName(p) + " hat " + SUIT_DE[x / CARDS_PER_SUIT] +
                                 " schon nicht bedient, kann also kein " + kindDe(x) + " haben");
                c = allocate(x, used);
                if (c < 0) return error("Es gibt kein " + kindDe(x) + " mehr (schon alle gesehen)");
                rs.s.hands[p].add(static_cast<Card>(c));
            }
            phys[i] = static_cast<Card>(c);
            applyRound(rs, Action::makePlay(static_cast<Card>(c)));
        }
    }

    // Pass 2: opponents hold every card they are seen to play from the start,
    // so the engine checks their follow-suit obligations too.
    std::array<CardSet, N_PLAYERS> full = hands;
    for (std::size_t i = 0; i < in.play.size(); ++i)
        if (!isMarker(in.play[i]) && by[i] != in.me) full[by[i]].add(phys[i]);
    RoundState rs = makeRoundState(in.round, full);
    for (int p = 0; p < N_PLAYERS; ++p) applyRound(rs, Action::makeBid(in.bids[p]));
    std::vector<Trick> tricks;
    int won[N_PLAYERS] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < in.play.size(); ++i) {
        if (isMarker(in.play[i])) {
            tricks.back().tigress = in.play[i] == PUB_TIGRESS_PIRATE ? 1 : 0;
            applyRound(rs, Action::makeTigressMode(in.play[i] == PUB_TIGRESS_PIRATE));
        } else {
            const int p = by[i];
            if (!legalKind(rs.s, in.play[i])) {
                std::string later;
                const Suit lead = rs.s.leadSuit;
                rs.s.hands[p].forEach([&](Card c) {
                    if (later.empty() && c != phys[i] && suitOf(c) == lead) later = kindDe(kindOf(c));
                });
                return error(seatName(p) + " hätte " + (lead != Suit::None ? SUIT_DE[static_cast<int>(lead)] : "Farbe") +
                             " bedienen müssen (spielt später noch " + later + ")");
            }
            if (rs.s.trickSize == 0) tricks.push_back({p, {}, -1, -1});
            tricks.back().cards.push_back({p, phys[i]});
            applyRound(rs, Action::makePlay(phys[i]));
        }
        Trick& t = tricks.back();
        if (static_cast<int>(t.cards.size()) == N_PLAYERS && !rs.s.pendingTigress && t.winner < 0) {
            Card cs[N_PLAYERS];
            std::int8_t ps[N_PLAYERS];
            for (int j = 0; j < N_PLAYERS; ++j) {
                ps[j] = static_cast<std::int8_t>(t.cards[j].first);
                cs[j] = t.cards[j].second;
            }
            t.winner = resolveTrick(cs, ps, N_PLAYERS, t.tigress == 1).winner;
            ++won[t.winner];
        }
    }

    const bool done = rs.terminal();
    const int toAct = rs.s.currentPlayer;
    o << ",\"phase\":\"" << (done ? "done" : "playing") << "\",\"toAct\":" << (done ? -1 : toAct)
      << ",\"pendingTigress\":" << (rs.s.pendingTigress ? "true" : "false") << ",\"bids\":";
    appendInts(o, in.bids, N_PLAYERS);
    o << ",\"won\":";
    appendInts(o, won, N_PLAYERS);

    o << ",\"hand\":[";
    bool first = true;
    if (!done)
        rs.s.hands[in.me].forEach([&](Card c) { o << (first ? "" : ",") << int(kindOf(c)); first = false; });
    o << "],\"unseen\":[";
    int unseen[N_KINDS];
    for (int k = 0; k < N_KINDS; ++k) unseen[k] = kindMultiplicity(static_cast<Kind>(k));
    for (int k : in.hand) --unseen[k];
    for (std::size_t i = 0; i < in.play.size(); ++i)
        if (!isMarker(in.play[i]) && by[i] != in.me) --unseen[in.play[i]];
    for (int k = 0; k < N_KINDS; ++k) o << (k ? "," : "") << unseen[k];
    o << "],\"voids\":[";
    for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << int(rs.s.voidSuits[p]);
    o << "],\"tricks\":[";
    for (std::size_t j = 0; j < tricks.size(); ++j) {
        const Trick& t = tricks[j];
        o << (j ? "," : "") << "{\"leader\":" << t.leader << ",\"cards\":[";
        for (std::size_t c = 0; c < t.cards.size(); ++c)
            o << (c ? "," : "") << '[' << t.cards[c].first << ',' << int(kindOf(t.cards[c].second)) << ']';
        o << "],\"tigress\":" << (t.tigress < 0 ? "null" : t.tigress ? "\"pirate\"" : "\"escape\"")
          << ",\"winner\":" << t.winner << '}';
    }
    o << ']';

    if (done) {
        o << ",\"points\":";
        appendInts(o, rs.s.scores, N_PLAYERS);
        o << ",\"options\":[]}";
        return o.str();
    }

    o << ",\"options\":[";
    if (toAct == in.me) {
        ActionList legal;
        legalKindActions(rs.s, legal);
        double p[ActionList::CAPACITY];
        if (net) policy(*net, rs, in.me, legal, p);
        for (int a = 0; a < legal.n; ++a) {
            const Action& act = legal[a];
            o << (a ? "," : "");
            if (act.type == ActionType::TigressMode)
                o << "{\"type\":\"tigress\",\"value\":\"" << (act.asPirate ? "pirate" : "escape") << '"';
            else
                o << "{\"type\":\"card\",\"value\":" << int(kindOf(act.card));
            if (net) o << ",\"p\":" << prob(p[a]);
            o << '}';
        }
    } else if (rs.s.pendingTigress) {
        o << "{\"type\":\"tigress\",\"value\":\"pirate\"},{\"type\":\"tigress\",\"value\":\"escape\"}";
    } else {
        // Anything still unseen that this opponent can hold.
        bool any = false;
        for (int k = 0; k < N_KINDS; ++k) {
            if (unseen[k] <= 0) continue;
            if (k < N_COLORED && ((rs.s.voidSuits[toAct] >> (k / CARDS_PER_SUIT)) & 1u)) continue;
            o << (any ? "," : "") << "{\"type\":\"card\",\"value\":" << k << ",\"n\":" << unseen[k] << '}';
            any = true;
        }
    }
    o << "]}";
    return o.str();
}

} // namespace sk::solver
