#include "sk/belief_model.hpp"
#include "sk/encoder.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sk {

struct BeliefEvaluator::Impl {
    torch::jit::script::Module module;
    torch::Device              device;

    explicit Impl(torch::Device dev) : device(dev) {}
};

BeliefEvaluator::BeliefEvaluator(const std::string& path, const std::string& device) {
    torch::Device dev = torch::kCPU;
    if (device == "cuda" || device.rfind("cuda:", 0) == 0) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("BeliefEvaluator: cuda requested but not available");
        }
        dev = torch::Device(device);
    } else if (device != "cpu") {
        throw std::runtime_error("BeliefEvaluator: unknown device '" + device + "'");
    }
    impl_ = std::make_unique<Impl>(dev);
    impl_->module = torch::jit::load(path, dev);
    impl_->module.eval();
    deviceName_ = device;
}

BeliefEvaluator::~BeliefEvaluator() = default;

BeliefPrediction BeliefEvaluator::evaluate(const Observation& obs) {
    torch::NoGradGuard nograd;

    std::vector<float> feats = encode(obs);
    auto input = torch::from_blob(
        feats.data(),
        {1, ENC_DIM},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(impl_->device);

    auto logits = impl_->module.forward({input}).toTensor()
                       .to(torch::kCPU).contiguous();   // (1, N_PLAYERS, N_CARDS)
    auto probs  = torch::sigmoid(logits);
    auto acc    = probs.accessor<float, 3>();

    BeliefPrediction out;
    for (int p = 0; p < N_PLAYERS; ++p) {
        for (int c = 0; c < N_CARDS; ++c) {
            out.probs[p][c] = acc[0][p][c];
        }
    }
    return out;
}

namespace {

// Iterative weighted assignment of unseen cards to opponents.
// `perspective` keeps the perspective player's own hand untouched.
void distributeWithBelief(GameState& d,
                          const GameState& src,
                          int perspective,
                          const BeliefPrediction& belief,
                          std::mt19937_64& rng)
{
    // Identify unseen cards.
    CardSet known;
    src.hands[perspective].forEach([&](Card c) { known.add(c); });
    for (int p = 0; p < N_PLAYERS; ++p) {
        src.captured[p].forEach([&](Card c) { known.add(c); });
    }
    for (int i = 0; i < src.trickSize; ++i) known.add(src.trickCards[i]);

    std::vector<Card> unseen;
    unseen.reserve(N_CARDS);
    for (Card c = 0; c < N_CARDS; ++c) if (!known.has(c)) unseen.push_back(c);

    // Hand-size budget per opponent.
    std::array<int, N_PLAYERS> budget{};
    for (int p = 0; p < N_PLAYERS; ++p) budget[p] = src.hands[p].count();
    budget[perspective] = 0;

    // Clear opponents' hands in the determinized state.
    for (int p = 0; p < N_PLAYERS; ++p) {
        if (p != perspective) d.hands[p].clear();
    }

    // Shuffle cards so we don't always assign in card-id order
    // (gives more diverse determinizations across rollouts).
    std::shuffle(unseen.begin(), unseen.end(), rng);

    for (const Card c : unseen) {
        std::array<float, N_PLAYERS> w{};
        float sum = 0.0f;
        for (int p = 0; p < N_PLAYERS; ++p) {
            if (p == perspective)  continue;
            if (budget[p] == 0)    continue;
            // Small floor so a player who lost all probability mass still has a chance.
            w[p] = std::max(belief.probs[p][c], 1e-4f);
            sum += w[p];
        }

        if (sum <= 0.0f) {
            // No candidate has budget. Should be unreachable given the budgets sum to
            // |unseen|, but guard anyway.
            break;
        }

        std::uniform_real_distribution<float> dist(0.0f, sum);
        float r = dist(rng);
        for (int p = 0; p < N_PLAYERS; ++p) {
            if (w[p] <= 0.0f) continue;
            if (r < w[p]) {
                d.hands[p].add(c);
                --budget[p];
                break;
            }
            r -= w[p];
        }
    }
}

} // namespace

GameState beliefDeterminize(const GameState& src,
                            int perspective,
                            BeliefEvaluator& belief,
                            std::mt19937_64& rng)
{
    GameState d = src;
    const Observation obs = observe(src, perspective);
    const BeliefPrediction pred = belief.evaluate(obs);
    distributeWithBelief(d, src, perspective, pred, rng);
    return d;
}

} // namespace sk
