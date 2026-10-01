#pragma once

#include "sk/solver/round.hpp"

#include <array>
#include <memory>
#include <string>

namespace sk::solver {

// Where a real decision point lives in a solver's table.
//   key    : the table key (infosets mapping to the same key share a strategy)
//   slot[a]: which strategy entry of that key belongs to real legal action a
//   nSlots : number of strategy entries of the key
struct InfosetView {
    InfoKey key;
    std::array<std::uint8_t, ActionList::CAPACITY> slot{};
    int nSlots = 0;
    // True if the situation is symmetric under some relabeling (several
    // relabelings give the same key). Actions swapped by that symmetry are
    // interchangeable, so which of them gets which slot is arbitrary.
    bool symmetric = false;
};

// Maps real information sets to (possibly coarser) table entries.
//
// A LOSSLESS abstraction only merges infosets that are strategically
// identical (e.g. the same situation with Yellow and Green swapped), so the
// solution of the abstract game is a solution of the real game. A LOSSY one
// merges situations that merely look alike; how much that costs is measured
// exactly on round 1 (NashConv of the abstract strategy in the real game).
//
// Contract: the view may only depend on what `player` knows, and the same
// real infoset must always produce the same key and slots. Action
// abstraction is not supported yet: nSlots == legal.n, slot is a bijection.
class Abstraction {
public:
    virtual ~Abstraction() = default;
    virtual void view(const RoundState& rs, int player, const ActionList& legal,
                      InfosetView& out) const = 0;
    virtual std::string name() const = 0;
};

// The real game: key = infosetKey, identity slots.
class ExactAbstraction final : public Abstraction {
public:
    void view(const RoundState& rs, int player, const ActionList& legal,
              InfosetView& out) const override;
    std::string name() const override { return "exact"; }
};

// Lossless: Yellow, Green and Purple are interchangeable (only Black is
// trump), so each infoset is mapped to the lexicographically smallest key
// over the 6 relabelings of those three suits.
class SuitIsomorphism final : public Abstraction {
public:
    void view(const RoundState& rs, int player, const ActionList& legal,
              InfosetView& out) const override;
    std::string name() const override { return "suit-iso"; }
};

const Abstraction& exactAbstraction();
std::shared_ptr<const Abstraction> makeAbstraction(const std::string& name);

} // namespace sk::solver
