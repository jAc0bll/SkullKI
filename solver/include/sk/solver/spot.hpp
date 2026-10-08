#pragma once

#include "sk/solver/mlp.hpp"
#include "sk/solver/round.hpp"

#include <string>
#include <vector>

namespace sk::solver {

// ---------------------------------------------------------------------------
// Spot queries: "what does the strategy do here?" for a situation entered by
// hand (web app, CLI). Only what one player knows goes in:
//
//   round   1..10
//   me      my seat relative to the round's start player (0 = bids/leads first)
//   hand    my hand at the start of the round, as kinds (duplicates allowed)
//   bids    4 bids by seat, -1 = not entered yet
//   play    public play log in order: kinds, and after a Tigress its mode
//           (PUB_TIGRESS_PIRATE / PUB_TIGRESS_ESCAPE). Who played each card
//           follows from the rules (turn order, trick winners).
//
// The other hands are unknown. They do not matter: the strategy only sees my
// infoset. To replay the log through the engine, every opponent is given
// exactly the cards they are seen to play, which also catches revokes (an
// opponent who later plays the led suit cannot have discarded on it before).
// ---------------------------------------------------------------------------
struct SpotInput {
    int round = 1;
    int me = 0;
    std::vector<int> hand;
    int bids[N_PLAYERS] = {-1, -1, -1, -1};
    std::vector<int> play;
};

// Mid-round entry without the history: what the strategy sees is only the
// current situation, so this gives exactly the answer of the full replay.
//   hand    my cards now
//   won     tricks won so far by seat
//   played  every card of the completed tricks (any order)
//   trick   the cards played before me in the current trick, in order (the
//           leader follows: me - trick size); tigress = mode of a Tigress in it
//   voids   suits (bit 0..3 = Gelb, Grün, Lila, Schwarz) a seat is known not
//           to have; voids shown in the current trick are added automatically
struct DirectInput {
    int round = 1;
    int me = 0;
    std::vector<int> hand;
    int bids[N_PLAYERS] = {-1, -1, -1, -1};
    int won[N_PLAYERS] = {0, 0, 0, 0};
    std::vector<int> played;
    std::vector<int> trick;
    int tigress = -1;                      // -1 none, 0 escape, 1 pirate
    int voids[N_PLAYERS] = {0, 0, 0, 0};
};

// Parses "round=5 me=2 hand=3,27,58 bids=1,0,-1,2 play=4,6,59,241".
// Missing keys keep their defaults. Returns false and sets `error` on junk.
bool parseSpot(const std::string& text, SpotInput& in, std::string& error);

// Evaluates the spot and returns a JSON object (see spot.cpp for the fields).
// `net` is the round's average-strategy network (avg.bin); without one the
// options carry no probabilities. Never throws: input errors come back as
// {"ok":false,"error":"..."}.
std::string spotQuery(const SpotInput& in, const MLP* net);

// The strategy's probabilities (softmax over the legal actions' logits) for
// `me` in `rs`, as NetPolicy with NetMode::Softmax.
void strategy(const MLP& net, const RoundState& rs, int me, const ActionList& legal, double* out);

// "mode=direct round=.. me=.. hand=.. bids=.. won=.. played=.. trick=.. tigress=.. voids=.."
bool parseDirect(const std::string& text, DirectInput& in, std::string& error);

// JSON: ok, round, me, phase "playing", toAct (= me), hand, unseen,
// options[{type:"card", value, p}] or {ok:false, error}.
std::string spotDirect(const DirectInput& in, const MLP* net);

} // namespace sk::solver
