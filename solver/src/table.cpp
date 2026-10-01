#include "sk/solver/table.hpp"

#include <cassert>
#include <functional>

namespace sk::solver {

void InfoNode::averageStrategy(double* out) const {
    double sum = 0.0;
    for (int a = 0; a < nA; ++a) sum += stratSum[a];
    for (int a = 0; a < nA; ++a) out[a] = (sum > 0.0) ? stratSum[a] / sum : 1.0 / nA;
}

InfosetTable::InfosetTable() {
    for (auto& s : shards_) s = std::make_unique<Shard>();
}

std::size_t InfosetTable::shardOf(const InfoKey& key) {
    // Mix the high bits in: std::hash on short strings is FNV-like.
    const std::size_t h = std::hash<InfoKey>{}(key);
    return (h ^ (h >> 17) ^ (h >> 31)) % N_SHARDS;
}

InfoNode& InfosetTable::findOrCreate(const InfoKey& key, int nA) {
    Shard& sh = *shards_[shardOf(key)];
    if (frozen_.load(std::memory_order_relaxed)) {
        auto it = sh.map.find(key);
        assert(it != sh.map.end() && "infoset missing from a frozen table");
        return it->second;
    }
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.map.find(key);
    if (it != sh.map.end()) {
        assert(it->second.nA == nA);
        return it->second;
    }
    return sh.map.emplace(key, InfoNode(nA)).first->second;
}

const InfoNode* InfosetTable::find(const InfoKey& key) const {
    const Shard& sh = *shards_[shardOf(key)];
    auto it = sh.map.find(key);
    return it == sh.map.end() ? nullptr : &it->second;
}

std::size_t InfosetTable::size() const {
    std::size_t n = 0;
    for (const auto& s : shards_) n += s->map.size();
    return n;
}

void AveragePolicy::probs(const InfoKey& key, int nA, double* out) const {
    const InfoNode* n = t_.find(key);
    if (!n) {
        for (int a = 0; a < nA; ++a) out[a] = 1.0 / nA;
        return;
    }
    assert(n->nA == nA);
    n->averageStrategy(out);
}

} // namespace sk::solver
