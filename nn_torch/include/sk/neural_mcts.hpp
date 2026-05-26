#pragma once

#include "sk/agent.hpp"
#include "sk/ismcts.hpp"           // reuse ISMCTSConfig and ISMCTSResult
#include "sk/evaluator.hpp"        // abstract BaseEvaluator
#include "sk/torch_model.hpp"      // for convenience (concrete impl)
#include "sk/belief_model.hpp"     // optional belief-conditioned determinizer

namespace sk {

// PUCT-style ISMCTS with a TorchScript PolicyValueNet guiding both the prior
// and the leaf evaluation.
//
//   Selection : argmax_a [ Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a)) ]
//   Expansion : add the highest-prior unvisited legal action as a new child.
//   Evaluate  : on new nodes call the NN; cache prior and per-player values;
//               propagate NN.values * rewardScale as the rollout reward.
//   At terminal / end-of-round leaves: use the real per-player score delta.
//
// The agent does NOT own the evaluator — pass a stable pointer that lives at
// least as long as the agent. This makes it cheap to share a single model
// across many parallel selfplay workers.
struct NeuralMCTSConfig {
    ISMCTSConfig mcts{};
    // If true, the leaf value comes from the NN. If false, run a random rollout
    // to round end and use the actual per-player score delta (AlphaGo-style
    // hybrid: NN-prior + MC rollout).
    bool useNNValue = true;
    // Optional belief evaluator: when non-null each rollout's determinization
    // samples opponent hands according to the predicted belief distribution
    // instead of uniformly over unseen cards. Non-owning pointer — caller
    // must outlive the agent.
    BeliefEvaluator* belief = nullptr;
};

class NeuralMCTSAgent final : public Agent {
public:
    NeuralMCTSAgent(BaseEvaluator* evaluator, NeuralMCTSConfig cfg = {})
        : model_(evaluator), cfg_(cfg) {}

    Action selectAction(const GameState& s, std::mt19937_64& rng) override;
    const char* name() const override { return "neural_mcts"; }

    ISMCTSResult selectActionWithTargets(const GameState& s, std::mt19937_64& rng);

    void setConfig(NeuralMCTSConfig cfg) { cfg_ = cfg; }
    const NeuralMCTSConfig& config() const { return cfg_; }

private:
    BaseEvaluator*       model_;
    NeuralMCTSConfig     cfg_;
};

} // namespace sk
