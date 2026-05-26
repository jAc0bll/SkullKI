#pragma once

#include "sk/observation.hpp"
#include "sk/encoder.hpp"
#include "sk/action_index.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace sk {

// One position's NN output.
//   policyLogits : raw logits over the unified action space
//   values       : per-player expected reward at this state
//                  (`values[state.currentPlayer]` is the active actor's expected reward)
struct PolicyValue {
    std::array<float, ACTION_DIM> policyLogits{};
    std::array<float, N_PLAYERS>  values{};
};

// Loads a TorchScript-exported PolicyValueNet and runs single or batched
// forward passes. Construct once, reuse across many evaluations.
class TorchModelEvaluator {
public:
    // path  : .pt file produced by train/export.py
    // device: "cpu" or "cuda" (or "cuda:0" etc.)
    TorchModelEvaluator(const std::string& path, const std::string& device = "cpu");
    ~TorchModelEvaluator();

    TorchModelEvaluator(const TorchModelEvaluator&) = delete;
    TorchModelEvaluator& operator=(const TorchModelEvaluator&) = delete;

    PolicyValue evaluate(const Observation& obs);
    std::vector<PolicyValue> evaluateBatch(const std::vector<Observation>& batch);

    const std::string& deviceName() const noexcept { return deviceName_; }

private:
    // Pimpl: torch_model.cpp pulls in <torch/script.h>.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string           deviceName_;
};

} // namespace sk
