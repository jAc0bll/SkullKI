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
    std::vector<float> w;
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t in = 0, out = 0;
        f.read(reinterpret_cast<char*>(&in), 4);
        f.read(reinterpret_cast<char*>(&out), 4);
        Layer l;
        l.in = static_cast<int>(in);
        l.out = static_cast<int>(out);
        w.resize(static_cast<std::size_t>(in) * out);
        l.b.resize(out);
        f.read(reinterpret_cast<char*>(w.data()), w.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(l.b.data()), l.b.size() * sizeof(float));
        if (!f) throw std::runtime_error("truncated model file " + path);
        if (!m.layers_.empty() && m.layers_.back().out != l.in)
            throw std::runtime_error("layer size mismatch in " + path);
        // Store transposed (in x out): the forward pass then adds one
        // contiguous column per input, which vectorises without reordering
        // a floating-point sum (a row-major dot product does not, unless the
        // compiler may reassociate, i.e. -ffast-math).
        l.wT.resize(w.size());
        for (int o = 0; o < l.out; ++o)
            for (int k = 0; k < l.in; ++k)
                l.wT[static_cast<std::size_t>(k) * l.out + o] = w[static_cast<std::size_t>(o) * l.in + k];
        m.layers_.push_back(std::move(l));
    }
    if (m.layers_.empty()) throw std::runtime_error("empty model " + path);
    return m;
}

namespace {

// dst = b + W x, with W stored transposed. Zero inputs (sparse features,
// ReLU zeros) are skipped. A single output (value head) is a dot product,
// computed with 8 independent partial sums so it vectorises too.
template <class T>
void dense(const std::vector<float>& wT, const std::vector<float>& b, int in, int out,
           const T* x, float* dst)
{
    if (out == 1) {
        float acc[8] = {};
        int k = 0;
        for (; k + 8 <= in; k += 8)
            for (int j = 0; j < 8; ++j) acc[j] += wT[k + j] * static_cast<float>(x[k + j]);
        float s = b[0];
        for (int j = 0; j < 8; ++j) s += acc[j];
        for (; k < in; ++k) s += wT[k] * static_cast<float>(x[k]);
        dst[0] = s;
        return;
    }
    std::copy(b.begin(), b.end(), dst);
    for (int k = 0; k < in; ++k) {
        if (!x[k]) continue;
        const float xk = static_cast<float>(x[k]);
        const float* col = wT.data() + static_cast<std::size_t>(k) * out;
        for (int o = 0; o < out; ++o) dst[o] += xk * col[o];
    }
}

} // namespace

void MLP::forward(const std::uint8_t* x, float* out) const {
    // Two ping-pong buffers sized for the widest layer.
    thread_local std::vector<float> a, b;
    int width = 0;
    for (const Layer& l : layers_) width = std::max({width, l.in, l.out});
    a.resize(width);
    b.resize(width);

    const Layer& first = layers_.front();
    const bool single = layers_.size() == 1;
    float* h = single ? out : a.data();
    dense(first.wT, first.b, first.in, first.out, x, h);
    if (single) return;
    for (int o = 0; o < first.out; ++o) h[o] = h[o] > 0.0f ? h[o] : 0.0f;

    float* cur = a.data();
    float* nxt = b.data();
    for (std::size_t li = 1; li < layers_.size(); ++li) {
        const Layer& l = layers_[li];
        const bool last = li + 1 == layers_.size();
        float* dst = last ? out : nxt;
        dense(l.wT, l.b, l.in, l.out, cur, dst);
        if (!last)
            for (int o = 0; o < l.out; ++o) dst[o] = dst[o] > 0.0f ? dst[o] : 0.0f;
        std::swap(cur, nxt);
    }
}

} // namespace sk::solver
