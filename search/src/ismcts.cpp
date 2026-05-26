#include "sk/ismcts.hpp"
#include "sk/determinize.hpp"
#include "sk/rules.hpp"
#include "sk/game.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace sk {

namespace {

struct Node {
    int parent = -1;
    Action incoming{};        // action that led from parent to this node (root: dummy)
    int playerAtNode = 0;     // who acts at this node
    int visits = 0;
    std::array<double, N_PLAYERS> rewardSum{};
    std::vector<int> children;
};

// Returns true if `acts` contains `a`.
bool containsAction(const std::vector<Action>& acts, const Action& a) {
    for (const Action& x : acts) if (x == a) return true;
    return false;
}

// Plays random legal actions until the round identified by `targetRound`
// (= the round we entered the search in) ends — i.e. the state transitions
// out of `targetRound` (phase=Bidding with roundNumber!=targetRound, or
// game terminates). Returns per-player score delta accumulated during the rollout.
//
// `targetRound` must come from obs.roundNumber: comparing against s.roundNumber
// at rollout start is a bug, because expansion may have already transitioned us
// into the next round before the rollout begins (in which case we'd then play
// out the next round's bidding with empty hands and crash on legalActions).
std::array<double, N_PLAYERS>
randomRolloutToRoundEnd(GameState s, std::mt19937_64& rng,
                        const std::array<std::int32_t, N_PLAYERS>& baselineScores,
                        std::uint8_t targetRound)
{
    while (!isTerminal(s)) {
        if (s.phase == Phase::Bidding && s.roundNumber != targetRound) break;

        const auto la = legalActions(s);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
        applyAction(s, la[dist(rng)]);
    }
    std::array<double, N_PLAYERS> rew{};
    for (int p = 0; p < N_PLAYERS; ++p) {
        rew[p] = static_cast<double>(s.scores[p] - baselineScores[p]);
    }
    return rew;
}

// UCB1 score for child `c` from parent's player perspective.
double ucb1(const Node& parent, const Node& child, int parentPlayer,
            double explorationC, double rewardScale)
{
    if (child.visits == 0) return std::numeric_limits<double>::infinity();
    const double avg = (child.rewardSum[parentPlayer] / child.visits) / rewardScale;
    const double exploration = explorationC *
        std::sqrt(std::log(static_cast<double>(parent.visits)) / child.visits);
    return avg + exploration;
}

// Pick best child index among those whose incoming action is in `legals`.
// Untried legal actions get expansion priority via the caller.
int selectChildUCB(const std::vector<Node>& nodes,
                   int curIdx,
                   const std::vector<Action>& legals,
                   double explorationC, double rewardScale)
{
    const Node& cur = nodes[curIdx];
    int best = -1;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int cIdx : cur.children) {
        if (!containsAction(legals, nodes[cIdx].incoming)) continue;
        const double sc = ucb1(cur, nodes[cIdx], cur.playerAtNode, explorationC, rewardScale);
        if (sc > bestScore) { bestScore = sc; best = cIdx; }
    }
    return best;
}

// Untried = legals minus actions already present as children.
std::vector<Action> untriedActions(const std::vector<Node>& nodes,
                                   int curIdx,
                                   const std::vector<Action>& legals)
{
    std::vector<Action> out;
    out.reserve(legals.size());
    const auto& kids = nodes[curIdx].children;
    for (const Action& a : legals) {
        bool found = false;
        for (int k : kids) if (nodes[k].incoming == a) { found = true; break; }
        if (!found) out.push_back(a);
    }
    return out;
}

} // namespace

Action ISMCTSAgent::selectAction(const GameState& s, std::mt19937_64& rng) {
    return selectActionWithTargets(s, rng).action;
}

ISMCTSResult ISMCTSAgent::selectActionWithTargets(const GameState& obs, std::mt19937_64& rng) {
    ISMCTSResult result{};

    // Trivial cases: no real choice.
    {
        const auto la = legalActions(obs);
        if (la.size() == 1) {
            result.action = la.front();
            result.rootActions = la;
            result.visits.assign(1, 1);
            result.values.fill(0.0);
            return result;
        }
    }

    const int mctsPlayer = obs.currentPlayer;

    // Baseline scores so we can compute "score earned this round" as the reward.
    std::array<std::int32_t, N_PLAYERS> baseline{};
    for (int p = 0; p < N_PLAYERS; ++p) baseline[p] = obs.scores[p];

    std::vector<Node> nodes;
    nodes.reserve(static_cast<std::size_t>(cfg_.numSimulations) + 1);
    nodes.emplace_back();
    nodes[0].playerAtNode = mctsPlayer;

    for (int sim = 0; sim < cfg_.numSimulations; ++sim) {
        GameState state = determinize(obs, mctsPlayer, rng);

        std::vector<int> path;
        path.push_back(0);
        int cur = 0;

        // ---- Selection + Expansion ----
        while (!isTerminal(state)) {
            // End-of-round terminal for our reward horizon.
            if (state.phase == Phase::Bidding && state.roundNumber != obs.roundNumber) break;

            const auto la = legalActions(state);
            const auto untried = untriedActions(nodes, cur, la);

            if (!untried.empty()) {
                // Expansion
                std::uniform_int_distribution<int> dist(0, static_cast<int>(untried.size()) - 1);
                const Action a = untried[dist(rng)];
                applyAction(state, a);
                Node child{};
                child.parent       = cur;
                child.incoming     = a;
                child.playerAtNode = state.currentPlayer;
                const int newIdx = static_cast<int>(nodes.size());
                nodes.push_back(child);
                nodes[cur].children.push_back(newIdx);
                path.push_back(newIdx);
                cur = newIdx;
                break;
            }

            // Selection: pick best compatible child by UCB1.
            const int next = selectChildUCB(nodes, cur, la, cfg_.explorationC, cfg_.rewardScale);
            if (next < 0) {
                // Pathological: no compatible child despite all legals being tried (shouldn't happen
                // in practice because untried covers anything missing). Bail to rollout from here.
                break;
            }
            applyAction(state, nodes[next].incoming);
            path.push_back(next);
            cur = next;
        }

        // ---- Simulation ----
        const auto reward = randomRolloutToRoundEnd(state, rng, baseline, obs.roundNumber);

        // ---- Backpropagation ----
        for (int idx : path) {
            ++nodes[idx].visits;
            for (int p = 0; p < N_PLAYERS; ++p) {
                nodes[idx].rewardSum[p] += reward[p];
            }
        }
    }

    // Collect root-child visit statistics for the training targets.
    int bestChild = -1;
    int bestVisits = -1;
    result.rootActions.reserve(nodes[0].children.size());
    result.visits.reserve(nodes[0].children.size());
    for (int cIdx : nodes[0].children) {
        result.rootActions.push_back(nodes[cIdx].incoming);
        result.visits.push_back(nodes[cIdx].visits);
        if (nodes[cIdx].visits > bestVisits) {
            bestVisits = nodes[cIdx].visits;
            bestChild = cIdx;
        }
    }
    if (nodes[0].visits > 0) {
        for (int p = 0; p < N_PLAYERS; ++p) {
            result.values[p] = nodes[0].rewardSum[p] / nodes[0].visits;
        }
    }

    if (bestChild < 0) {
        // Fallback: random legal (shouldn't happen unless numSimulations == 0).
        const auto la = legalActions(obs);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
        result.action = la[dist(rng)];
        return result;
    }
    result.action = nodes[bestChild].incoming;
    return result;
}

} // namespace sk
