#include "sk/solver/session.hpp"

#include "sk/solver/spot.hpp"

#include <map>
#include <sstream>

namespace sk::solver {

namespace {

std::map<int, GameSession> sessions;
int nextId = 1;

std::string fail(const std::string& msg) { return "{\"ok\":false,\"error\":\"" + msg + "\"}"; }

RoundState asRound(const GameState& s) {
    RoundState rs;
    rs.s = s;
    rs.round = s.roundNumber;
    return rs;
}

bool bidding(const GameSession& g) { return !g.roundOver && g.s.phase == Phase::Bidding; }

// What `seat` may do now, with the state the decision is made in.
void legalFor(const GameSession& g, int seat, GameState& at, ActionList& out) {
    out.n = 0;
    at = g.s;
    if (g.roundOver || g.s.phase == Phase::GameEnd) return;
    if (bidding(g)) {
        if (g.pendingBids[seat] >= 0) return;
        at.currentPlayer = static_cast<std::int8_t>(seat);   // bids are hidden: order is irrelevant
        legalKindActions(at, out);
        return;
    }
    if (g.s.currentPlayer == seat) legalKindActions(at, out);
}

void dealNext(GameSession& g) {
    dealRound(g.s, g.rng);
    RoundLog r;
    r.round = g.s.roundNumber;
    r.start = g.s.startPlayer;
    for (int p = 0; p < N_PLAYERS; ++p) g.s.hands[p].forEach([&](Card c) { r.hands[p].push_back(kindOf(c)); });
    g.log.push_back(std::move(r));
    for (int p = 0; p < N_PLAYERS; ++p) {
        g.pendingBids[p] = -1;
        g.scoresAtRoundStart[p] = g.s.scores[p];
    }
    g.roundOver = false;
    g.current = {};
    g.last = {};
}

void apply(GameSession& g, int seat, const Action& a) {
    if (a.type == ActionType::Bid) {
        g.pendingBids[seat] = a.bid;
        for (int p = 0; p < N_PLAYERS; ++p)
            if (g.pendingBids[p] < 0) return;
        while (g.s.phase == Phase::Bidding)   // reveal: the engine takes them in turn order
            applyAction(g.s, Action::makeBid(g.pendingBids[g.s.currentPlayer]));
        return;
    }
    const GameState prev = g.s;
    if (a.type == ActionType::Play) {
        if (g.s.trickSize == 0) g.current = {g.s.currentPlayer, {}, -1, -1};
        g.current.cards.push_back({seat, a.card});
    } else {
        g.current.tigress = a.asPirate ? 1 : 0;
    }
    applyAction(g.s, a);

    if (static_cast<int>(g.current.cards.size()) == N_PLAYERS && !g.s.pendingTigress) {
        Card cs[N_PLAYERS];
        std::int8_t ps[N_PLAYERS];
        for (int i = 0; i < N_PLAYERS; ++i) {
            ps[i] = static_cast<std::int8_t>(g.current.cards[i].first);
            cs[i] = g.current.cards[i].second;
        }
        g.current.winner = resolveTrick(cs, ps, N_PLAYERS, g.current.tigress == 1).winner;
        g.last = g.current;
        g.current = {};
    }
    if (g.s.phase == Phase::GameEnd || g.s.roundNumber != prev.roundNumber) {
        RoundResult r;
        r.round = prev.roundNumber;
        for (int p = 0; p < N_PLAYERS; ++p) {
            r.bids[p] = prev.bids[p];
            r.won[p] = prev.tricksWon[p] + (g.last.winner == p ? 1 : 0);
            r.points[p] = g.s.scores[p] - g.scoresAtRoundStart[p];
        }
        g.results.push_back(r);
        g.roundOver = true;
    }
}

std::string actionText(const Action& a) {
    switch (a.type) {
        case ActionType::Bid:         return "bid:" + std::to_string(a.bid);
        case ActionType::Play:        return "card:" + std::to_string(kindOf(a.card));
        case ActionType::TigressMode: return std::string("tig:") + (a.asPirate ? "1" : "0");
    }
    return "";
}

int actionValue(const Action& a) {
    switch (a.type) {
        case ActionType::Bid:         return a.bid;
        case ActionType::Play:        return kindOf(a.card);
        case ActionType::TigressMode: return a.asPirate ? 1 : 0;
    }
    return 0;
}

const char* actionType(const Action& a) {
    return a.type == ActionType::Bid ? "bid" : a.type == ActionType::Play ? "card" : "tigress";
}

void trickJson(std::ostringstream& o, const TrickRecord& t) {
    o << "{\"leader\":" << t.leader << ",\"cards\":[";
    for (std::size_t i = 0; i < t.cards.size(); ++i)
        o << (i ? "," : "") << '[' << t.cards[i].first << ',' << int(kindOf(t.cards[i].second)) << ']';
    o << "],\"tigress\":" << (t.tigress < 0 ? "null" : t.tigress ? "\"pirate\"" : "\"escape\"")
      << ",\"winner\":" << t.winner << '}';
}

// {"v":1,"rounds":[{"round":r,"start":p,"hands":[[kinds]x4],
//   "actions":[[seat,"card:13",human(0/1),p,"best"]...]}],"scores":[..]}
std::string logJson(const GameSession& g) {
    std::ostringstream o;
    o << "{\"v\":1,\"rounds\":[";
    for (std::size_t i = 0; i < g.log.size(); ++i) {
        const RoundLog& r = g.log[i];
        o << (i ? "," : "") << "{\"round\":" << r.round << ",\"start\":" << r.start << ",\"hands\":[";
        for (int p = 0; p < N_PLAYERS; ++p) {
            o << (p ? "," : "") << '[';
            for (std::size_t k = 0; k < r.hands[p].size(); ++k) o << (k ? "," : "") << r.hands[p][k];
            o << ']';
        }
        o << "],\"actions\":[";
        for (std::size_t k = 0; k < r.actions.size(); ++k) {
            const LoggedAction& a = r.actions[k];
            o << (k ? "," : "") << '[' << a.seat << ",\"" << a.a << "\"," << (a.human ? 1 : 0);
            if (a.human) o << ',' << a.p << ",\"" << a.best << '"';
            o << ']';
        }
        o << "]}";
    }
    o << "],\"scores\":[";
    for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << g.s.scores[p];
    o << "]}";
    return o.str();
}

std::string view(const GameSession& g, int id, int seat) {
    const GameState& s = g.s;
    std::ostringstream o;
    const char* phase = g.roundOver ? (s.phase == Phase::GameEnd ? "gameOver" : "roundEnd")
                        : s.phase == Phase::Bidding ? "bidding" : "playing";
    const int round = g.roundOver && !g.results.empty() ? g.results.back().round : s.roundNumber;
    o << "{\"ok\":true,\"id\":" << id << ",\"seat\":" << seat << ",\"round\":" << round << ",\"phase\":\"" << phase
      << "\",\"start\":" << int(s.startPlayer) << ",\"pendingTigress\":" << (s.pendingTigress ? "true" : "false");

    o << ",\"toAct\":[";
    bool first = true;
    if (!g.roundOver && s.phase != Phase::GameEnd)
        for (int p = 0; p < N_PLAYERS; ++p)
            if (bidding(g) ? g.pendingBids[p] < 0 : s.currentPlayer == p) {
                o << (first ? "" : ",") << p;
                first = false;
            }
    o << "],\"hand\":[";
    first = true;
    s.hands[seat].forEach([&](Card c) { o << (first ? "" : ",") << int(kindOf(c)); first = false; });
    o << "],\"bids\":[";
    for (int p = 0; p < N_PLAYERS; ++p) {
        int b = s.bids[p];
        if (bidding(g)) b = p == seat ? g.pendingBids[p] : -1;
        if (g.roundOver && !g.results.empty()) b = g.results.back().bids[p];
        o << (p ? "," : "") << b;
    }
    o << "],\"bidIn\":[";
    for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << (bidding(g) ? (g.pendingBids[p] >= 0 ? "true" : "false") : "true");
    o << "],\"won\":[";
    for (int p = 0; p < N_PLAYERS; ++p)
        o << (p ? "," : "") << (g.roundOver && !g.results.empty() ? g.results.back().won[p] : int(s.tricksWon[p]));
    o << "],\"scores\":[";
    for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << s.scores[p];
    o << "],\"trick\":";
    trickJson(o, g.current);
    o << ",\"lastTrick\":";
    if (g.last.cards.empty()) o << "null";
    else trickJson(o, g.last);

    GameState at;
    ActionList legal;
    legalFor(g, seat, at, legal);
    o << ",\"legal\":[";
    for (int a = 0; a < legal.n; ++a) {
        o << (a ? "," : "") << "{\"type\":\"" << actionType(legal[a]) << "\",\"value\":" << actionValue(legal[a])
          << ",\"a\":\"" << actionText(legal[a]) << "\"}";
    }
    o << "],\"results\":[";
    for (std::size_t i = 0; i < g.results.size(); ++i) {
        const RoundResult& r = g.results[i];
        o << (i ? "," : "") << "{\"round\":" << r.round << ",\"bids\":[";
        for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << r.bids[p];
        o << "],\"won\":[";
        for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << r.won[p];
        o << "],\"points\":[";
        for (int p = 0; p < N_PLAYERS; ++p) o << (p ? "," : "") << r.points[p];
        o << "]}";
    }
    o << "],\"review\":[";
    first = true;
    for (const Review& r : g.reviews) {
        if (r.seat != seat) continue;
        o << (first ? "" : ",") << "{\"round\":" << r.round << ",\"trick\":" << r.trick << ",\"what\":\"" << r.what
          << "\",\"chosen\":" << r.chosen << ",\"p\":" << r.pChosen << ",\"best\":" << r.best << ",\"pBest\":" << r.pBest
          << '}';
        first = false;
    }
    o << "]}";
    return o.str();
}

// Parses "a=card:13" style actions against the legal list.
bool parseAction(const std::string& a, const ActionList& legal, Action& out) {
    for (int i = 0; i < legal.n; ++i)
        if (actionText(legal[i]) == a) {
            out = legal[i];
            return true;
        }
    return false;
}

} // namespace

std::string gameCommand(const std::string& text, const std::array<const MLP*, 11>& nets) {
    std::istringstream words(text);
    std::string game, cmd, w;
    words >> game >> cmd;
    std::map<std::string, std::string> kv;
    while (words >> w) {
        const auto eq = w.find('=');
        if (eq != std::string::npos) kv[w.substr(0, eq)] = w.substr(eq + 1);
    }
    auto num = [&](const char* k, long long def) {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        try { return std::stoll(it->second); } catch (...) { return def; }
    };

    if (cmd == "new") {
        const int id = nextId++;
        GameSession& g = sessions[id];
        g.rng.seed(static_cast<std::uint64_t>(num("seed", 1)));
        g.s = initialState(static_cast<int>(num("start", 0)) & 3);
        dealNext(g);
        return "{\"ok\":true,\"id\":" + std::to_string(id) + "}";
    }
    const int id = static_cast<int>(num("id", -1));
    auto it = sessions.find(id);
    if (it == sessions.end()) return fail("Spiel nicht gefunden");
    GameSession& g = it->second;
    if (cmd == "drop") {
        sessions.erase(it);
        return "{\"ok\":true}";
    }
    if (cmd == "log") return logJson(g);
    if (cmd == "next") {
        if (!g.roundOver || g.s.phase == Phase::GameEnd) return fail("Runde läuft noch");
        dealNext(g);
        return "{\"ok\":true}";
    }
    const int seat = static_cast<int>(num("seat", -1));
    if (seat < 0 || seat >= N_PLAYERS) return fail("Sitz fehlt");
    if (cmd == "view") return view(g, id, seat);

    GameState at;
    ActionList legal;
    legalFor(g, seat, at, legal);
    if (legal.n == 0) return fail("Du bist nicht dran");
    const MLP* net = nets[at.roundNumber];
    double p[ActionList::CAPACITY] = {};
    if (net) strategy(*net, asRound(at), seat, legal, p);

    if (cmd == "hint") {   // the strategy's probabilities for seat's options, without acting
        if (!net) return fail("net");
        std::ostringstream o;
        o << "{\"ok\":true,\"options\":[";
        for (int i = 0; i < legal.n; ++i)
            o << (i ? "," : "") << "{\"a\":\"" << actionText(legal[i]) << "\",\"p\":" << p[i] << '}';
        o << "]}";
        return o.str();
    }

    Action a{};
    LoggedAction entry;
    entry.seat = seat;
    if (cmd == "bot") {
        if (!net) return fail("net");
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(g.rng), acc = 0.0;
        a = legal[legal.n - 1];
        for (int i = 0; i < legal.n; ++i) {
            acc += p[i];
            if (r < acc) {
                a = legal[i];
                break;
            }
        }
    } else if (cmd == "act") {
        if (!parseAction(kv["a"], legal, a)) return fail("Zug nicht erlaubt");
        if (net) {
            int best = 0, chosen = 0;
            for (int i = 0; i < legal.n; ++i) {
                if (p[i] > p[best]) best = i;
                if (actionText(legal[i]) == actionText(a)) chosen = i;
            }
            g.reviews.push_back({seat, at.roundNumber, at.tricksPlayed + 1, actionType(a), actionValue(a),
                                 actionValue(legal[best]), p[chosen], p[best]});
            entry.p = p[chosen];
            entry.best = actionText(legal[best]);
        }
        entry.human = true;
    } else {
        return fail("unbekannter Befehl");
    }
    entry.a = actionText(a);
    if (!g.log.empty()) g.log.back().actions.push_back(entry);
    apply(g, seat, a);
    return "{\"ok\":true,\"a\":\"" + actionText(a) + "\"}";
}

} // namespace sk::solver
