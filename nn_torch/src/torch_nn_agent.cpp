#include "sk/torch_nn_agent.hpp"
#include "sk/rules.hpp"

#include <algorithm>
#include <limits>

namespace sk {

Action TorchNNAgent::selectAction(const GameState& s, std::mt19937_64& /*rng*/) {
    const auto la = legalActions(s);
    if (la.size() == 1) return la.front();

    const Observation obs = observe(s, s.currentPlayer);
    const PolicyValue pv  = evaluator_.evaluate(obs);

    // Argmax over legal actions only.
    float bestLogit = -std::numeric_limits<float>::infinity();
    Action bestAction = la.front();
    for (const Action& a : la) {
        const int idx = actionToIndex(a);
        if (idx < 0 || idx >= ACTION_DIM) continue;
        const float logit = pv.policyLogits[idx];
        if (logit > bestLogit) { bestLogit = logit; bestAction = a; }
    }
    return bestAction;
}

} // namespace sk
