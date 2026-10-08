#!/usr/bin/env bash
# Build the spot solver as WebAssembly for the app's web version:
# app/modules/sk-solver/src/web/sk-web.js (one CommonJS file, wasm inlined,
# so Metro can bundle it). The iOS app compiles the same C++ natively.
# Needs the Emscripten SDK (https://emscripten.org), e.g. in ~/emsdk:
#   source ~/emsdk/emsdk_env.sh && bash scripts/build_wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EMXX=${EMXX:-em++}
OUT=app/modules/sk-solver/src/web
mkdir -p "$OUT"
common=(-std=c++20 -O3 -msimd128 -fwasm-exceptions -Iengine/include -Isolver/include
    engine/src/game.cpp engine/src/rules.cpp engine/src/scoring.cpp
    solver/src/round.cpp solver/src/encoding.cpp solver/src/mlp.cpp solver/src/spot.cpp
    solver/src/spot_c_api.cpp solver/src/session.cpp
    -sMODULARIZE=1 -sEXPORT_NAME=createSk -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=1 -sFORCE_FILESYSTEM=1
    -sSINGLE_FILE=1 -sEXPORTED_FUNCTIONS=_sk_load,_sk_spot -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,FS)
"$EMXX" "${common[@]}" -sENVIRONMENT=web -o "$OUT/sk-web.js"
# Same build for Node, used by app/tools/check-wasm.mjs.
"$EMXX" "${common[@]}" -sENVIRONMENT=node -o app/tools/sk-node.cjs
ls -l "$OUT" app/tools/sk-node.cjs
