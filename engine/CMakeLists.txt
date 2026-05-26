add_library(sk_engine STATIC
    src/rules.cpp
    src/scoring.cpp
    src/game.cpp
    src/observation.cpp
    src/encoder.cpp
)

target_include_directories(sk_engine PUBLIC include)
target_compile_features(sk_engine PUBLIC cxx_std_20)
