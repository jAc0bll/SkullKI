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

// Parses "round=5 me=2 hand=3,27,58 bids=1,0,-1,2 play=4,6,59,241".
// Missing keys keep their defaults. Returns false and sets `error` on junk.
bool parseSpot(const std::string& text, SpotInput& in, std::string& error);

// Evaluates the spot and returns a JSON object (see spot.cpp for the fields).
// `net` is the round's average-strategy network (avg.bin); without one the
// options carry no probabilities. Never throws: input errors come back as
// {"ok":false,"error":"..."}.
std::string spotQuery(const SpotInput& in, const MLP* net);

} // namespace sk::solver
