#pragma once

#include "cards.hpp"
#include <bit>
#include <cstdint>

namespace sk {

// Trivially-copyable 70-bit set over Card IDs.
// Lower 64 bits in `lo`, upper 6 bits (cards 64..69) in `hi`.
struct CardSet {
    std::uint64_t lo = 0;
    std::uint8_t  hi = 0;  // only bits 0..5 used

    constexpr void clear() noexcept { lo = 0; hi = 0; }

    constexpr bool has(Card c) const noexcept {
        return c < 64
            ? ((lo >> c) & 1ull) != 0
            : ((hi >> (c - 64)) & 1u) != 0;
    }

    constexpr void add(Card c) noexcept {
        if (c < 64) lo |= (1ull << c);
        else        hi |= static_cast<std::uint8_t>(1u << (c - 64));
    }

    constexpr void remove(Card c) noexcept {
        if (c < 64) lo &= ~(1ull << c);
        else        hi &= static_cast<std::uint8_t>(~(1u << (c - 64)));
    }

    constexpr bool empty() const noexcept { return lo == 0 && hi == 0; }

    int count() const noexcept {
        return std::popcount(lo) + std::popcount(static_cast<std::uint32_t>(hi));
    }

    // Iterate set bits. Callable receives a Card.
    template <class F>
    void forEach(F&& f) const {
        std::uint64_t x = lo;
        while (x) {
            int b = std::countr_zero(x);
            f(static_cast<Card>(b));
            x &= x - 1;
        }
        std::uint32_t y = hi;
        while (y) {
            int b = std::countr_zero(y);
            f(static_cast<Card>(64 + b));
            y &= y - 1;
        }
    }

    constexpr bool operator==(const CardSet& o) const noexcept {
        return lo == o.lo && hi == o.hi;
    }
};

static_assert(sizeof(CardSet) <= 16, "CardSet should stay small");

} // namespace sk
