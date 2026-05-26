#include "sk/neural_mcts.hpp"
#include "sk/determinize.hpp"
#include "sk/rules.hpp"
#include "sk/game.hpp"
#include "sk/observation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace sk {

namespace {

struct PUCTNode {
    int                              parent       = -1;
    Action                           incoming{};
    std::int8_t                      playerAtNode = 0;
    int                              visits       = 0;
    std::array<double, N_PLAYERS>    rewardSum{};
    std::vector<int>                 children;
    std::array<float, ACTION_DIM>    prior{};         // P(a) — softmax over legal actions
    std::array<float, N_PLAYERS>     nnValues{};      // cached NN per-player values
    bool                             nnEvaluated = false;
};

bool containsAction(const std::vector<Action>& acts, const Action& a) {
    for (const Action& x : acts) if (x == a) return true;
    return false;
}

// Convert raw logits into a probability over legal actions; illegal entries become 0.
void normalizePriorOverLegal(std::array<float, ACTION_DIM>& prior,
                             const std::vector<Action>& legals)
{
    if (legals.empty()) return;

    float maxLogit = -std::numeric_limits<float>::infinity();
    for (const Action& a : legals) {
        const int idx = actionToIndex(a);
        if (prior[idx] > maxLogit) maxLogit = prior[idx];
    }

    std::array<float, ACTION_DIM> exps{};
    float sum = 0.0f;
    for (const Action& a : legals) {
        const int idx = actionToIndex(a);
        const float e = std::exp(prior[idx] - maxLogit);
        exps[idx] = e;
        sum += e;
    }
    for (int i = 0; i < ACTION_DIM; ++i) {
        prior[i] = (sum > 0.0f) ? (exps[i] / sum) : 0.0f;
    }
}

void evaluateNodeWithNN(PUCTNode& node,
                        const GameState& state,
                        TorchModelEvaluator& model)
{
    const Observation obs = observe(state, state.currentPlayer);
    const PolicyValue pv  = model.evaluate(obs);

    for (int a = 0; a < ACTION_DIM; ++a) node.prior[a] = pv.policyLogits[a];
    for (int p = 0; p < N_PLAYERS; ++p)  node.nnValues[p] = pv.values[p];

    const auto la = legalActions(state);
    normalizePriorOverLegal(node.prior, la);
    node.nnEvaluated = true;
}

bool isHorizonEnd(const GameState& state, std::uint8_t rootRound) {
    if (isTerminal(state)) return true;
    if (state.phase == Phase::Bidding && state.roundNumber != rootRound) return true;
    return false;
}

} // anonymous namespace

Action NeuralMCTSAgent::selectAction(const GameState& s, std::mt19937_64& rng) {
    return selectActionWithTargets(s, rng).action;
}

ISMCTSResult NeuralMCTSAgent::selectActionWithTargets(
    const GameState& obs, std::mt19937_64& rng)
{
    ISMCTSResult result{};

    // Trivial: only one legal action — no need to search.
    {
        const auto la = legalActions(obs);
        if (la.size() == 1) {
            result.action       = la.front();
            result.rootActions  = la;
            result.visits.assign(1, 1);
            result.values.fill(0.0);
            return result;
        }
    }

    const std::int8_t mctsPlayer = obs.currentPlayer;

    // Baseline scores for terminal reward deltas.
    std::array<std::int32_t, N_PLAYERS> baseline{};
    for (int p = 0; p < N_PLAYERS; ++p) baseline[p] = obs.scores[p];

    // Allocate tree. Worst case 1 root + numSimulations expansions.
    std::vector<PUCTNode> nodes;
    nodes.reserve(static_cast<std::size_t>(cfg_.mcts.numSimulations) + 1);
    nodes.emplace_back();
    nodes[0].playerAtNode = mctsPlayer;

    // Initial NN evaluation of the root.
    {
        // Build a "dummy" observation from `obs` itself — the caller already gives
        // us their viewpoint, but to be safe we re-observe from mctsPlayer.
        // (`obs` here is named confusingly — it's the true GameState we received.)
        evaluateNodeWithNN(nodes[0], obs, *model_);
    }

    const float cPUCT      = static_cast<float>(cfg_.mcts.explorationC);
    const double rewardSc  = cfg_.mcts.rewardScale;

    for (int sim = 0; sim < cfg_.mcts.numSimulations; ++sim) {
        GameState state = (cfg_.belief != nullptr)
            ? beliefDeterminize(obs, mctsPlayer, *cfg_.belief, rng)
            : determinize(obs, mctsPlayer, rng);

        std::vector<int> path;
        path.reserve(16);
        path.push_back(0);
        int cur = 0;
        std::array<double, N_PLAYERS> reward{};
        bool rewardSet = false;

        while (!isHorizonEnd(state, obs.roundNumber)) {
            const auto la = legalActions(state);
            const int parentPlayer = nodes[cur].playerAtNode;
            const int parentVisits = std::max(1, nodes[cur].visits);
            const float sqrtN      = std::sqrt(static_cast<float>(parentVisits));

            // Best PUCT score across existing children + untried legal actions.
            float bestScore       = -std::numeric_limits<float>::infinity();
            int   bestChildIdx    = -1;     // index into nodes[] for existing-child case
            Action bestNewAction{};         // for expansion case
            bool  pickExpand      = false;

            for (int childIdx : nodes[cur].children) {
                if (!containsAction(la, nodes[childIdx].incoming)) continue;
                const int aIdx = actionToIndex(nodes[childIdx].incoming);
                const float P  = nodes[cur].prior[aIdx];
                const int   n  = nodes[childIdx].visits;
                const float Q  = (n > 0)
                    ? static_cast<float>(nodes[childIdx].rewardSum[parentPlayer] / n / rewardSc)
                    : 0.0f;
                const float U  = cPUCT * P * sqrtN / (1.0f + static_cast<float>(n));
                const float sc = Q + U;
                if (sc > bestScore) { bestScore = sc; bestChildIdx = childIdx; pickExpand = false; }
            }

            for (const Action& a : la) {
                // Skip if already a child.
                bool already = false;
                for (int childIdx : nodes[cur].children) {
                    if (nodes[childIdx].incoming == a) { already = true; break; }
                }
                if (already) continue;
                const int aIdx = actionToIndex(a);
                const float P  = nodes[cur].prior[aIdx];
                // First-Play-Urgency: 0 (commonly used default).
                const float U  = cPUCT * P * sqrtN; // n = 0 -> denominator 1
                const float sc = 0.0f + U;
                if (sc > bestScore) { bestScore = sc; bestNewAction = a; pickExpand = true; }
            }

            if (bestChildIdx < 0 && !pickExpand) {
                // Pathological: should not occur because legals is non-empty here.
                break;
            }

            if (pickExpand) {
                applyAction(state, bestNewAction);

                PUCTNode child{};
                child.parent       = cur;
                child.incoming     = bestNewAction;
                child.playerAtNode = state.currentPlayer;
                const int newIdx   = static_cast<int>(nodes.size());
                nodes.push_back(child);
                nodes[cur].children.push_back(newIdx);
                path.push_back(newIdx);
                cur = newIdx;

                if (isHorizonEnd(state, obs.roundNumber)) {
                    for (int p = 0; p < N_PLAYERS; ++p) {
                        reward[p] = static_cast<double>(state.scores[p] - baseline[p]);
                    }
                } else {
                    // Always evaluate with NN for the prior at this new node.
                    evaluateNodeWithNN(nodes[cur], state, *model_);
                    if (cfg_.useNNValue) {
                        for (int p = 0; p < N_PLAYERS; ++p) {
                            reward[p] = static_cast<double>(nodes[cur].nnValues[p]) * rewardSc;
                        }
                    } else {
                        // Hybrid: random rollout to round end for leaf value.
                        GameState sim = state;
                        while (!isHorizonEnd(sim, obs.roundNumber)) {
                            const auto la = legalActions(sim);
                            std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
                            applyAction(sim, la[dist(rng)]);
                        }
                        for (int p = 0; p < N_PLAYERS; ++p) {
                            reward[p] = static_cast<double>(sim.scores[p] - baseline[p]);
                        }
                    }
                }
                rewardSet = true;
                break;
            }

            // Pure selection — descend into existing child.
            applyAction(state, nodes[bestChildIdx].incoming);
            path.push_back(bestChildIdx);
            cur = bestChildIdx;
        }

        if (!rewardSet) {
            // Loop exited because state is already at horizon end (terminal or new round).
            for (int p = 0; p < N_PLAYERS; ++p) {
                reward[p] = static_cast<double>(state.scores[p] - baseline[p]);
            }
        }

        for (int idx : path) {
            ++nodes[idx].visits;
            for (int p = 0; p < N_PLAYERS; ++p) {
                nodes[idx].rewardSum[p] += reward[p];
            }
        }
    }

    // Collect root statistics for training targets and pick the most-visited action.
    int bestChild  = -1;
    int bestVisits = -1;
    result.rootActions.reserve(nodes[0].children.size());
    result.visits.reserve(nodes[0].children.size());
    for (int childIdx : nodes[0].children) {
        result.rootActions.push_back(nodes[childIdx].incoming);
        result.visits.push_back(nodes[childIdx].visits);
        if (nodes[childIdx].visits > bestVisits) {
            bestVisits = nodes[childIdx].visits;
            bestChild  = childIdx;
        }
    }
    if (nodes[0].visits > 0) {
        for (int p = 0; p < N_PLAYERS; ++p) {
            result.values[p] = nodes[0].rewardSum[p] / nodes[0].visits;
        }
    }

    if (bestChild < 0) {
        // Fallback: legal pick (essentially numSimulations == 0).
        const auto la = legalActions(obs);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(la.size()) - 1);
        result.action = la[dist(rng)];
        return result;
    }
    result.action = nodes[bestChild].incoming;
    return result;
}

} // namespace sk
