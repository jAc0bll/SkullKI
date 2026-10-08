#!/usr/bin/env bash
# Build the spot solver for the browser: web/src/wasm/sk.js + sk.wasm.
# Needs the Emscripten SDK (https://emscripten.org), e.g. in ~/emsdk:
#   source ~/emsdk/emsdk_env.sh && bash scripts/build_wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EMXX=${EMXX:-em++}
OUT=web/src/wasm
mkdir -p "$OUT"
"$EMXX" -std=c++20 -O3 -msimd128 -fwasm-exceptions \
    -Iengine/include -Isolver/include \
    engine/src/game.cpp engine/src/rules.cpp engine/src/scoring.cpp \
    solver/src/round.cpp solver/src/encoding.cpp solver/src/mlp.cpp solver/src/spot.cpp \
    web/wasm/spot_wasm.cpp \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSk \
    -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=1 -sFORCE_FILESYSTEM=1 \
    -sEXPORTED_FUNCTIONS=_sk_load,_sk_spot -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,FS \
    -o "$OUT/sk.js"
ls -l "$OUT"
