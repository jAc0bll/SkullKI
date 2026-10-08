// WebAssembly entry points for the web app (built by scripts/build_wasm.sh).
//
//   sk_load(round, path)  load the round's average-strategy net from the
//                         in-memory file system (JS writes it there first)
//   sk_spot(text)         spotQuery() on "round=.. me=.. hand=.. bids=.. play=.."
//                         with the loaded net of that round; returns JSON

#include "sk/solver/spot.hpp"

#include <array>
#include <memory>
#include <string>

using namespace sk::solver;

namespace {
std::array<std::unique_ptr<MLP>, sk::MAX_ROUND + 1> nets;
std::string answer;
}

extern "C" {

int sk_load(int round, const char* path) {
    if (round < 1 || round > sk::MAX_ROUND) return 0;
    try {
        nets[round] = std::make_unique<MLP>(MLP::load(path));
    } catch (...) {
        return 0;
    }
    return 1;
}

const char* sk_spot(const char* text) {
    SpotInput in;
    std::string err;
    if (!parseSpot(text, in, err)) {
        answer = "{\"ok\":false,\"error\":\"" + err + "\"}";
        return answer.c_str();
    }
    const MLP* net = in.round >= 1 && in.round <= sk::MAX_ROUND ? nets[in.round].get() : nullptr;
    answer = spotQuery(in, net);
    return answer.c_str();
}

}
