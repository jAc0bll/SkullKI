#pragma once

#include "sk/agent.hpp"
#include "sk/torch_model.hpp"

namespace sk {

// Agent that evaluates a TorchScript PolicyValueNet to choose actions.
// Deterministic argmax over legal actions; for stochastic play, lower-level
// callers can sample from the masked policy themselves.
class TorchNNAgent final : public Agent {
public:
    TorchNNAgent(const std::string& model_path, const std::string& device = "cpu")
        : evaluator_(model_path, device) {}

    Action selectAction(const GameState& s, std::mt19937_64& rng) override;
    const char* name() const override { return "torch_nn"; }

    TorchModelEvaluator& evaluator() noexcept { return evaluator_; }

private:
    TorchModelEvaluator evaluator_;
};

} // namespace sk
