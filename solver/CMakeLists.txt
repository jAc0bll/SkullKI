add_library(sk_solver STATIC
    src/round.cpp
    src/deals.cpp
    src/table.cpp
    src/cfr.cpp
    src/best_response.cpp
)

target_include_directories(sk_solver PUBLIC include)
target_link_libraries(sk_solver PUBLIC sk_engine)
target_compile_features(sk_solver PUBLIC cxx_std_20)
