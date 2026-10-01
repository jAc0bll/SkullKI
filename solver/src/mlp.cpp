#include "sk/solver/mlp.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sk::solver {

MLP MLP::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "SKMLP001", 8) != 0) throw std::runtime_error("bad model file " + path);
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    MLP m;
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t in = 0, out = 0;
        f.read(reinterpret_cast<char*>(&in), 4);
        f.read(reinterpret_cast<char*>(&out), 4);
        Layer l;
        l.in = static_cast<int>(in);
        l.out = static_cast<int>(out);
        l.w.resize(static_cast<std::size_t>(in) * out);
        l.b.resize(out);
        f.read(reinterpret_cast<char*>(l.w.data()), l.w.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(l.b.data()), l.b.size() * sizeof(float));
        if (!f) throw std::runtime_error("truncated model file " + path);
        if (!m.layers_.empty() && m.layers_.back().out != l.in)
            throw std::runtime_error("layer size mismatch in " + path);
        m.layers_.push_back(std::move(l));
    }
    if (m.layers_.empty()) throw std::runtime_error("empty model " + path);
    const Layer& f0 = m.layers_.front();
    m.firstT_.resize(f0.w.size());
    for (int o = 0; o < f0.out; ++o)
        for (int i = 0; i < f0.in; ++i)
            m.firstT_[static_cast<std::size_t>(i) * f0.out + o] = f0.w[static_cast<std::size_t>(o) * f0.in + i];
    return m;
}

void MLP::forward(const std::uint8_t* x, float* out) const {
    // Two ping-pong buffers sized for the widest layer.
    thread_local std::vector<float> a, b;
    int width = 0;
    for (const Layer& l : layers_) width = std::max({width, l.in, l.out});
    a.resize(width);
    b.resize(width);

    // First layer: inputs are sparse small integers, so accumulate only the
    // columns of non-zero inputs.
    const Layer& first = layers_.front();
    const bool single = layers_.size() == 1;
    float* h = single ? out : a.data();
    std::copy(first.b.begin(), first.b.end(), h);
    for (int i = 0; i < first.in; ++i) {
        if (!x[i]) continue;
        const float xi = static_cast<float>(x[i]);
        const float* col = firstT_.data() + static_cast<std::size_t>(i) * first.out;
        for (int o = 0; o < first.out; ++o) h[o] += xi * col[o];
    }
    if (single) return;
    for (int o = 0; o < first.out; ++o) h[o] = h[o] > 0.0f ? h[o] : 0.0f;

    float* cur = a.data();
    float* nxt = b.data();
    for (std::size_t li = 1; li < layers_.size(); ++li) {
        const Layer& l = layers_[li];
        const bool last = li + 1 == layers_.size();
        float* dst = last ? out : nxt;
        for (int o = 0; o < l.out; ++o) {
            const float* w = l.w.data() + static_cast<std::size_t>(o) * l.in;
            float acc = l.b[o];
            for (int i = 0; i < l.in; ++i) acc += w[i] * cur[i];
            dst[o] = (last || acc > 0.0f) ? acc : 0.0f;
        }
        std::swap(cur, nxt);
    }
}

} // namespace sk::solver
