#pragma once

#include "cards.hpp"
#include "card_set.hpp"
#include <array>
#include <cstdint>

namespace sk {

constexpr int N_PLAYERS = 4;
constexpr int MAX_ROUND = 10;

enum class Phase : std::uint8_t {
    Bidding,
    Playing,
    GameEnd
};

struct GameState {
    // ---- Round / phase ----
    std::uint8_t roundNumber = 1;          // 1..10
    Phase        phase       = Phase::Bidding;
    std::int8_t  startPlayer = 0;          // who leads the first trick of this round
    std::int8_t  currentPlayer = 0;        // whose turn

    // ---- Bidding state ----
    std::int8_t  bids[N_PLAYERS]      = {-1, -1, -1, -1};
    std::int8_t  bidsSubmitted        = 0; // 0..4

    // ---- Trick state ----
    std::int8_t  trickLeader   = 0;        // who led the current trick
    std::int8_t  trickSize     = 0;        // cards in trick so far (0..N_PLAYERS)
    std::int8_t  tricksPlayed  = 0;        // completed tricks this round
    Suit         leadSuit      = Suit::None;
    bool         freeTrick     = false;    // true if first card was a character (no follow required)
    bool         pendingTigress= false;    // last play was Tigress, awaiting TigressMode
    bool         tigressAsPirate=false;    // mode of the Tigress in current trick (if played)

    Card         trickCards[N_PLAYERS]   = {};
    std::int8_t  trickPlayers[N_PLAYERS] = {};

    // ---- Per-player hand / collected info ----
    CardSet      hands[N_PLAYERS];
    std::int8_t  tricksWon[N_PLAYERS]    = {0,0,0,0};
    CardSet      captured[N_PLAYERS];                  // cards collected in won tricks (this round)
    std::int32_t pendingBonus[N_PLAYERS] = {0,0,0,0};  // trick-event bonuses accumulated this round

    // ---- Cumulative scores (carry across rounds) ----
    std::int32_t scores[N_PLAYERS] = {0,0,0,0};
};

static_assert(std::is_trivially_copyable_v<GameState>,
              "GameState must be trivially copyable for cheap MCTS cloning");

} // namespace sk
