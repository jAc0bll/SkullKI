#pragma once

#include "sk/cards.hpp"
#include <cstdint>
#include <string>

namespace sk::solver {

// A card "kind" identifies a card up to rules-equivalence. The 5 Escapes,
// 2 Mermaids and 5 Pirates are interchangeable, so a strategy must never
// distinguish them: infoset keys and actions are expressed in kinds.
//   0..55 : colored cards (same layout as Card ids)
//   56    : Escape (x5)
//   57    : Mermaid (x2)
//   58    : Pirate (x5)
//   59    : Tigress
//   60    : Skull King
using Kind = std::uint8_t;

constexpr int  N_KINDS      = 61;
constexpr Kind KIND_ESCAPE  = 56;
constexpr Kind KIND_MERMAID = 57;
constexpr Kind KIND_PIRATE  = 58;
constexpr Kind KIND_TIGRESS = 59;
constexpr Kind KIND_SK      = 60;

constexpr Kind kindOf(Card c) {
    if (isColored(c))  return c;
    if (isEscape(c))   return KIND_ESCAPE;
    if (isMermaid(c))  return KIND_MERMAID;
    if (isPirate(c))   return KIND_PIRATE;
    if (isTigress(c))  return KIND_TIGRESS;
    return KIND_SK;
}

// How many physical cards of this kind are in the deck.
constexpr int kindMultiplicity(Kind k) {
    switch (k) {
        case KIND_ESCAPE:  return 5;
        case KIND_MERMAID: return 2;
        case KIND_PIRATE:  return 5;
        default:           return 1;
    }
}

// Lowest Card id of this kind; ids of a kind are contiguous.
constexpr Card firstCardOfKind(Kind k) {
    switch (k) {
        case KIND_ESCAPE:  return ESCAPE_OFFSET;
        case KIND_MERMAID: return MERMAID_OFFSET;
        case KIND_PIRATE:  return PIRATE_OFFSET;
        case KIND_TIGRESS: return TIGRESS;
        case KIND_SK:      return SKULL_KING;
        default:           return k;
    }
}

inline std::string kindName(Kind k) {
    return cardName(firstCardOfKind(k));
}

} // namespace sk::solver
