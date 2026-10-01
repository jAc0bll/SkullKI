add_library(sk_solver STATIC
    src/round.cpp
    src/abstraction.cpp
    src/feature_abstraction.cpp
    src/deals.cpp
    src/table.cpp
    src/dcfr_update.cpp
    src/cfr.cpp
    src/mccfr.cpp
    src/best_response.cpp
)

target_include_directories(sk_solver PUBLIC include)
target_link_libraries(sk_solver PUBLIC sk_engine)
target_compile_features(sk_solver PUBLIC cxx_std_20)
