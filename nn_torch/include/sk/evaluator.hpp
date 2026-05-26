#pragma once

#include "sk/observation.hpp"
#include "sk/encoder.hpp"
#include "sk/action_index.hpp"

#include <array>
#include <vector>

namespace sk {

// One position's policy/value output. (Used by both single-eval and batched
// codepaths; the array sizes are the bindings' single source of truth.)
struct PolicyValue {
    std::array<float, ACTION_DIM> policyLogits{};
    std::array<float, N_PLAYERS>  values{};
};

// Polymorphic evaluator interface. Lets MCTS work with:
//   - TorchModelEvaluator        (synchronous per-call)
//   - AsyncBatchEvaluator        (worker-pool, batched on GPU, comes in Phase 5C.3)
//   - PythonCallbackEvaluator    (debug / testing)
//   - any future variant
//
// Owners hand pointers to NeuralMCTSAgent; the agent must not outlive the
// evaluator.
class BaseEvaluator {
public:
    virtual ~BaseEvaluator() = default;

    // Single-position inference. Implementations may internally batch with
    // pending requests from other threads — the caller blocks until its
    // own result is ready.
    virtual PolicyValue evaluate(const Observation& obs) = 0;

    // Best-effort batched inference. Default implementation loops; subclasses
    // that actually batch on the GPU should override.
    virtual std::vector<PolicyValue>
    evaluateBatch(const std::vector<Observation>& batch) {
        std::vector<PolicyValue> out;
        out.reserve(batch.size());
        for (const auto& obs : batch) out.push_back(evaluate(obs));
        return out;
    }
};

} // namespace sk
