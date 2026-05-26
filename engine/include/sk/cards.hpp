#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace sk {

// Base-game deck: 70 cards.
//   IDs 0..13   : Yellow (Schatztruhe) values 1..14
//   IDs 14..27  : Green  (Papagei)     values 1..14
//   IDs 28..41  : Purple (Schatzkarte) values 1..14
//   IDs 42..55  : Black  (Piratenflagge, trump) values 1..14
//   IDs 56..60  : Escape (Flucht)   x5
//   IDs 61..62  : Mermaid           x2
//   IDs 63..67  : Pirate            x5  (functionally identical in base game)
//   ID  68      : Tigress
//   ID  69      : Skull King

using Card = std::uint8_t;

constexpr int N_CARDS         = 70;
constexpr int N_COLORED       = 56;
constexpr int CARDS_PER_SUIT  = 14;
constexpr Card YELLOW_OFFSET  = 0;
constexpr Card GREEN_OFFSET   = 14;
constexpr Card PURPLE_OFFSET  = 28;
constexpr Card BLACK_OFFSET   = 42;
constexpr Card ESCAPE_OFFSET  = 56;
constexpr Card MERMAID_OFFSET = 61;
constexpr Card PIRATE_OFFSET  = 63;
constexpr Card TIGRESS        = 68;
constexpr Card SKULL_KING     = 69;

enum class Suit : std::uint8_t {
    Yellow = 0,
    Green  = 1,
    Purple = 2,
    Black  = 3,   // Trump
    None   = 4    // For special cards / "no lead suit yet"
};

constexpr bool isColored(Card c)   { return c < N_COLORED; }
constexpr bool isEscape(Card c)    { return c >= ESCAPE_OFFSET && c < MERMAID_OFFSET; }
constexpr bool isMermaid(Card c)   { return c >= MERMAID_OFFSET && c < PIRATE_OFFSET; }
constexpr bool isPirate(Card c)    { return c >= PIRATE_OFFSET && c < TIGRESS; }
constexpr bool isTigress(Card c)   { return c == TIGRESS; }
constexpr bool isSkullKing(Card c) { return c == SKULL_KING; }
constexpr bool isSpecial(Card c)   { return !isColored(c); }

constexpr Suit suitOf(Card c) {
    if (c < GREEN_OFFSET)  return Suit::Yellow;
    if (c < PURPLE_OFFSET) return Suit::Green;
    if (c < BLACK_OFFSET)  return Suit::Purple;
    if (c < ESCAPE_OFFSET) return Suit::Black;
    return Suit::None;
}

// 1..14 for colored cards, 0 for specials.
constexpr int valueOf(Card c) {
    return isColored(c) ? (c % CARDS_PER_SUIT) + 1 : 0;
}

constexpr bool isTrump(Card c) { return suitOf(c) == Suit::Black; }
constexpr bool is14(Card c)    { return isColored(c) && valueOf(c) == CARDS_PER_SUIT; }

constexpr Card makeColored(Suit s, int value) {
    // value 1..14
    Card offset = 0;
    switch (s) {
        case Suit::Yellow: offset = YELLOW_OFFSET; break;
        case Suit::Green:  offset = GREEN_OFFSET;  break;
        case Suit::Purple: offset = PURPLE_OFFSET; break;
        case Suit::Black:  offset = BLACK_OFFSET;  break;
        default: return 0;
    }
    return static_cast<Card>(offset + value - 1);
}

inline std::string cardName(Card c) {
    if (isColored(c)) {
        static constexpr const char* names[4] = {"Y", "G", "P", "B"};
        return std::string(names[static_cast<int>(suitOf(c))]) + std::to_string(valueOf(c));
    }
    if (isEscape(c))    return "Escape";
    if (isMermaid(c))   return "Mermaid";
    if (isPirate(c))    return "Pirate";
    if (isTigress(c))   return "Tigress";
    if (isSkullKing(c)) return "SkullKing";
    return "?";
}

} // namespace sk
