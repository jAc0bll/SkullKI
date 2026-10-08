#pragma once

#include "sk/solver/mlp.hpp"
#include "sk/solver/round.hpp"

#include <array>
#include <random>
#include <string>
#include <vector>

namespace sk::solver {

// ---------------------------------------------------------------------------
// A full 10-round game for the app (bot mode on the phone, multiplayer on
// the server): deals, checks moves, lets bots play the trained strategy and
// records how the humans' choices compare to it. Bidding is simultaneous:
// bids are collected in any order and revealed together.
//
// Text commands (gameCommand), answers are JSON:
//   game new seed=S [start=P]          -> {"ok":true,"id":N}
//   game view id=N seat=P              -> what seat P sees (see session.cpp)
//   game act id=N seat=P a=bid:2|card:13|tig:1|tig:0
//   game bot id=N seat=P               -> seat P plays the strategy (sampled)
//   game hint id=N seat=P              -> {"options":[{"a":..,"p":..}]} without acting
//   game next id=N                     -> deal the next round after a round end
//   game log id=N                      -> the whole game, replayable (see session.cpp)
//   game drop id=N
// Nets: nets[round] is the round's average-strategy net (may be null while
// loading: "bot" then answers {"ok":false,"error":"net"}).
// ---------------------------------------------------------------------------

struct TrickRecord {
    int leader = 0;
    std::vector<std::pair<int, Card>> cards;
    int tigress = -1;   // -1 none, 0 escape, 1 pirate
    int winner = -1;
};

struct RoundResult {
    int round = 0;
    int bids[N_PLAYERS] = {};
    int won[N_PLAYERS] = {};
    int points[N_PLAYERS] = {};
};

struct Review {
    int seat = 0, round = 0, trick = 0;
    std::string what;          // "bid" | "card" | "tigress"
    int chosen = 0, best = 0;  // bid value / card kind / tigress 1=pirate 0=escape
    double pChosen = 0, pBest = 0;
};

// Everything needed to replay a game exactly, for analysing human play.
struct LoggedAction {
    int seat = 0;
    std::string a;          // "bid:2" | "card:13" | "tig:1"
    bool human = false;
    double p = -1;          // strategy probability of the chosen action (humans; -1 unknown)
    std::string best;       // the strategy's most likely action (humans)
};
struct RoundLog {
    int round = 0, start = 0;
    std::vector<int> hands[N_PLAYERS];   // kinds dealt
    std::vector<LoggedAction> actions;   // in the order they happened
};

struct GameSession {
    GameState s;
    std::mt19937_64 rng;
    int pendingBids[N_PLAYERS] = {-1, -1, -1, -1};
    bool roundOver = false;          // waiting for "next"
    int scoresAtRoundStart[N_PLAYERS] = {};
    TrickRecord current, last;
    std::vector<RoundResult> results;
    std::vector<Review> reviews;
    std::vector<RoundLog> log;
};

std::string gameCommand(const std::string& text, const std::array<const MLP*, 11>& nets);

} // namespace sk::solver
