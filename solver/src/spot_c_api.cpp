// Plain C entry points of the spot solver, shared by the WebAssembly build
// (scripts/build_wasm.sh) and the iOS app (app/modules/sk-solver).
//
//   sk_load(round, path)  load the round's average-strategy net from a file
//   sk_spot(text)         spotQuery() on "round=.. me=.. hand=.. bids=.. play=.."
//                         or spotDirect() on "mode=direct ...", or a full game
//                         ("game new|view|act|bot|next|drop ...", session.hpp), with the
//                         loaded net of that round; returns JSON (valid until
//                         the next call)

#include "sk/solver/spot_c_api.h"
#include "sk/solver/session.hpp"
#include "sk/solver/spot.hpp"

#include <array>
#include <memory>
#include <string>

using namespace sk::solver;

namespace {
std::array<std::unique_ptr<MLP>, sk::MAX_ROUND + 1> nets;
std::string answer;

const MLP* netFor(int round) {
    return round >= 1 && round <= sk::MAX_ROUND ? nets[round].get() : nullptr;
}
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
    const std::string t = text;
    std::string err;
    if (t.rfind("game ", 0) == 0) {
        std::array<const MLP*, sk::MAX_ROUND + 1> view{};
        for (int r = 0; r <= sk::MAX_ROUND; ++r) view[r] = nets[r].get();
        answer = gameCommand(t, view);
    } else if (t.rfind("mode=direct", 0) == 0) {
        DirectInput in;
        answer = parseDirect(t, in, err) ? spotDirect(in, netFor(in.round))
                                         : "{\"ok\":false,\"error\":\"" + err + "\"}";
    } else {
        SpotInput in;
        answer = parseSpot(t, in, err) ? spotQuery(in, netFor(in.round))
                                       : "{\"ok\":false,\"error\":\"" + err + "\"}";
    }
    return answer.c_str();
}

}
