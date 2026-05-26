#pragma once

#include "sk/evaluator.hpp"

#include <memory>
#include <string>

namespace sk {

// TorchScript-backed evaluator: synchronous per-call forward on CPU or CUDA.
// Use this for single-process simple inference (e.g. existing selfplay
// workers). For the multi-threaded batched setup, see AsyncBatchEvaluator.
class TorchModelEvaluator final : public BaseEvaluator {
public:
    // path  : .pt file produced by train/export.py
    // device: "cpu" or "cuda" (or "cuda:0" etc.)
    TorchModelEvaluator(const std::string& path, const std::string& device = "cpu");
    ~TorchModelEvaluator() override;

    TorchModelEvaluator(const TorchModelEvaluator&) = delete;
    TorchModelEvaluator& operator=(const TorchModelEvaluator&) = delete;

    PolicyValue evaluate(const Observation& obs) override;
    std::vector<PolicyValue> evaluateBatch(const std::vector<Observation>& batch) override;

    const std::string& deviceName() const noexcept { return deviceName_; }

private:
    // Pimpl: torch_model.cpp pulls in <torch/script.h>.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string           deviceName_;
};

} // namespace sk
