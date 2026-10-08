#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sk::solver {

// Dense ReLU network for inference, trained in PyTorch (train/deep_cfr.py)
// and exported to a small binary file:
//   "SKMLP001" | uint32 nLayers | per layer: uint32 in, uint32 out,
//   float32 weight[out][in], float32 bias[out]
// ReLU after every layer but the last. Thread-safe for concurrent forward().
class MLP {
public:
    static MLP load(const std::string& path);

    int inputDim()  const { return layers_.empty() ? 0 : layers_.front().in; }
    int outputDim() const { return layers_.empty() ? 0 : layers_.back().out; }

    // `x` holds inputDim() raw uint8 features; writes outputDim() floats.
    void forward(const std::uint8_t* x, float* out) const;

private:
    struct Layer {
        int in = 0, out = 0;
        std::vector<float> wT;   // in x out (transposed from the file's out x in)
        std::vector<float> b;
    };
    std::vector<Layer> layers_;
};

} // namespace sk::solver
