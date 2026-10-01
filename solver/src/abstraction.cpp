#include "sk/solver/abstraction.hpp"

#include <algorithm>
#include <stdexcept>

namespace sk::solver {

namespace {

constexpr int N_PERMS = 6;
// Permutations of the three plain suits (Yellow=0, Green=1, Purple=2).
constexpr int PERMS[N_PERMS][3] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
};

// Relabel a kind under a suit permutation; Black and specials are fixed.
std::uint8_t permuteKind(std::uint8_t k, const int* perm) {
    if (k >= BLACK_OFFSET) return k;   // Black, specials, Tigress-mode codes
    const int suit  = k / CARDS_PER_SUIT;
    const int value = k % CARDS_PER_SUIT;
    return static_cast<std::uint8_t>(perm[suit] * CARDS_PER_SUIT + value);
}

void identitySlots(int n, InfosetView& out) {
    out.nSlots = n;
    for (int a = 0; a < n; ++a) out.slot[a] = static_cast<std::uint8_t>(a);
}

} // namespace

void ExactAbstraction::view(const RoundState& rs, int player, const ActionList& legal,
                            InfosetView& out) const
{
    infosetKey(rs, player, out.key);
    identitySlots(legal.n, out);
    out.symmetric = false;
}

void SuitIsomorphism::view(const RoundState& rs, int player, const ActionList& legal,
                           InfosetView& out) const
{
    // Key layout (see round.hpp): [round][seat][phase][handLen][hand...]
    //                              ([bids x N][public log...] when playing).
    InfoKey real;
    infosetKey(rs, player, real);
    const int handLen = static_cast<std::uint8_t>(real[3]);
    const std::size_t handBegin = 4;
    const std::size_t handEnd = handBegin + handLen;
    const std::size_t logBegin =
        real[2] == 0 ? real.size() : handEnd + N_PLAYERS;   // bids are not cards

    InfoKey best;
    int bestPerm = 0;
    bool tie = false;
    InfoKey cand;
    for (int pi = 0; pi < N_PERMS; ++pi) {
        cand = real;
        for (std::size_t i = handBegin; i < handEnd; ++i)
            cand[i] = static_cast<char>(permuteKind(static_cast<std::uint8_t>(cand[i]), PERMS[pi]));
        std::sort(cand.begin() + handBegin, cand.begin() + handEnd,
                  [](char a, char b) { return static_cast<std::uint8_t>(a) < static_cast<std::uint8_t>(b); });
        for (std::size_t i = logBegin; i < cand.size(); ++i)
            cand[i] = static_cast<char>(permuteKind(static_cast<std::uint8_t>(cand[i]), PERMS[pi]));
        if (pi == 0 || cand < best) {
            best = cand;
            bestPerm = pi;
            tie = false;
        } else if (cand == best) {
            tie = true;
        }
    }
    out.key = std::move(best);
    out.symmetric = tie;

    // Slots: plays are ordered by their relabeled kind, matching the order
    // in which legalKindActions lists them in the canonical situation.
    const bool plays = legal.n > 0 && legal[0].type == ActionType::Play;
    if (!plays) {
        identitySlots(legal.n, out);
        return;
    }
    std::array<std::uint8_t, ActionList::CAPACITY> canon{};
    for (int a = 0; a < legal.n; ++a)
        canon[a] = permuteKind(kindOf(legal[a].card), PERMS[bestPerm]);
    out.nSlots = legal.n;
    for (int a = 0; a < legal.n; ++a) {
        int rank = 0;
        for (int b = 0; b < legal.n; ++b) rank += canon[b] < canon[a];
        out.slot[a] = static_cast<std::uint8_t>(rank);
    }
}

const Abstraction& exactAbstraction() {
    static const ExactAbstraction instance;
    return instance;
}

std::shared_ptr<const Abstraction> makeAbstraction(const std::string& name) {
    if (name == "exact")    return std::make_shared<ExactAbstraction>();
    if (name == "suit-iso") return std::make_shared<SuitIsomorphism>();
    throw std::invalid_argument("unknown abstraction: " + name);
}

} // namespace sk::solver
