#include "sk/torch_model.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <cstring>
#include <stdexcept>

namespace sk {

struct TorchModelEvaluator::Impl {
    torch::jit::script::Module module;
    torch::Device              device;

    explicit Impl(torch::Device dev) : device(dev) {}
};

TorchModelEvaluator::TorchModelEvaluator(const std::string& path,
                                         const std::string& device)
{
    torch::Device dev = torch::kCPU;
    if (device == "cuda" || device.rfind("cuda:", 0) == 0) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("TorchModelEvaluator: cuda requested but not available");
        }
        dev = torch::Device(device);
    } else if (device != "cpu") {
        throw std::runtime_error("TorchModelEvaluator: unknown device '" + device + "'");
    }

    impl_ = std::make_unique<Impl>(dev);
    impl_->module = torch::jit::load(path, dev);
    impl_->module.eval();
    deviceName_ = device;
}

TorchModelEvaluator::~TorchModelEvaluator() = default;

PolicyValue TorchModelEvaluator::evaluate(const Observation& obs) {
    torch::NoGradGuard nograd;

    std::vector<float> feats = encode(obs);
    auto input = torch::from_blob(
        feats.data(),
        {1, ENC_DIM},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(impl_->device);

    auto out = impl_->module.forward({input}).toTuple();
    auto policy = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
    auto value  = out->elements()[1].toTensor().to(torch::kCPU).contiguous();

    PolicyValue r;
    std::memcpy(r.policyLogits.data(), policy.data_ptr<float>(),
                ACTION_DIM * sizeof(float));
    std::memcpy(r.values.data(), value.data_ptr<float>(),
                N_PLAYERS * sizeof(float));
    return r;
}

std::vector<PolicyValue>
TorchModelEvaluator::evaluateBatch(const std::vector<Observation>& batch) {
    torch::NoGradGuard nograd;

    const int B = static_cast<int>(batch.size());
    std::vector<PolicyValue> out(B);
    if (B == 0) return out;

    std::vector<float> feats(static_cast<std::size_t>(B) * ENC_DIM);
    std::vector<float> tmp;
    for (int i = 0; i < B; ++i) {
        encodeInto(batch[i], tmp);
        std::memcpy(feats.data() + static_cast<std::size_t>(i) * ENC_DIM,
                    tmp.data(), ENC_DIM * sizeof(float));
    }
    auto input = torch::from_blob(
        feats.data(),
        {B, ENC_DIM},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(impl_->device);

    auto tup    = impl_->module.forward({input}).toTuple();
    auto policy = tup->elements()[0].toTensor().to(torch::kCPU).contiguous();
    auto value  = tup->elements()[1].toTensor().to(torch::kCPU).contiguous();

    auto policyAcc = policy.accessor<float, 2>();
    auto valueAcc  = value.accessor<float, 2>();
    for (int i = 0; i < B; ++i) {
        for (int a = 0; a < ACTION_DIM; ++a) {
            out[i].policyLogits[a] = policyAcc[i][a];
        }
        for (int p = 0; p < N_PLAYERS; ++p) {
            out[i].values[p] = valueAcc[i][p];
        }
    }
    return out;
}

} // namespace sk
