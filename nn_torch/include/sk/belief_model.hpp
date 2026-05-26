#pragma once

#include "sk/observation.hpp"
#include "sk/cards.hpp"
#include "sk/state.hpp"

#include <array>
#include <memory>
#include <random>
#include <string>

namespace sk {

struct BeliefPrediction {
    // probs[p][c] = predicted probability that player p holds card c.
    std::array<std::array<float, N_CARDS>, N_PLAYERS> probs{};
};

// Loads a TorchScript-exported BeliefNet and runs forward passes.
class BeliefEvaluator {
public:
    BeliefEvaluator(const std::string& path, const std::string& device = "cpu");
    ~BeliefEvaluator();
    BeliefEvaluator(const BeliefEvaluator&)            = delete;
    BeliefEvaluator& operator=(const BeliefEvaluator&) = delete;

    BeliefPrediction evaluate(const Observation& obs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string           deviceName_;
};

// Sample a determinized GameState using `belief` to bias the assignment of
// unseen cards to opponents. Uses each opponent's publicly-known hand size as
// a hard constraint.
GameState beliefDeterminize(const GameState& src,
                            int perspective,
                            BeliefEvaluator& belief,
                            std::mt19937_64& rng);

} // namespace sk
