#pragma once

#include "sk/game.hpp"
#include "sk/solver/kinds.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace sk::solver {

// ---------------------------------------------------------------------------
// One round of Skull King as a self-contained game.
//
// Cards are reshuffled every round, so rounds only interact through the score
// vector. Under a points objective each round is an independent game and can
// be solved on its own; seat 0 is always the round's start player.
// ---------------------------------------------------------------------------

enum class Utility : std::uint8_t {
    // Own round points minus the mean of the other players' round points.
    // Zero-sum; what matters for winning the game.
    Relative,
    // Own round points only (general-sum).
    Absolute,
};

// Public action log of the play phase, in kinds. A Tigress play is followed
// by its declared mode. Bids are read from GameState once all are revealed.
constexpr std::uint8_t PUB_TIGRESS_ESCAPE = 0xF0;
constexpr std::uint8_t PUB_TIGRESS_PIRATE = 0xF1;

struct RoundState {
    GameState    s;
    std::uint8_t round  = 1;
    std::uint8_t pubLen = 0;
    std::array<std::uint8_t, 48> pub{};   // <= 40 plays + 1 Tigress mode

    bool terminal() const {
        return s.phase == Phase::GameEnd || s.roundNumber != round;
    }
};

// Build the start-of-round state (bidding phase, seat 0 starts, scores 0).
RoundState makeRoundState(int round, const std::array<CardSet, N_PLAYERS>& hands);

// Apply an action and record its public trace.
void applyRound(RoundState& rs, Action a);

// Legal actions with interchangeable cards collapsed to one action per kind
// (e.g. two Pirates in hand → one "play Pirate" action). Deterministic order.
void legalKindActions(const GameState& s, ActionList& out);

// Round utilities for all players at a terminal RoundState.
std::array<double, N_PLAYERS> roundUtilities(const RoundState& rs, Utility u);

// ---------------------------------------------------------------------------
// Information-set keys.
//
// A key is a byte string holding exactly what `player` knows:
//   [round][seat][phase][handLen][hand kinds, sorted]          (bidding)
//   ... + [4 bids][public play log]                            (playing)
// While bidding, other players' bids are NOT part of the key — bidding is
// simultaneous. Own past plays are public, so the current hand plus the
// public log determines the player's full private history (perfect recall).
// ---------------------------------------------------------------------------
using InfoKey = std::string;

void infosetKey(const RoundState& rs, int player, InfoKey& out);

// Bidding-phase key from seat + hand alone (no state needed).
void biddingKey(int round, int seat, const CardSet& hand, InfoKey& out);

// Human-readable rendering of a key and of an action.
std::string describeKey(const InfoKey& key);
std::string describeAction(const Action& a);

} // namespace sk::solver
