add_executable(sk_tournament tournament.cpp)
target_link_libraries(sk_tournament PRIVATE sk_engine sk_agents sk_search)
target_compile_features(sk_tournament PRIVATE cxx_std_20)

if(SK_BUILD_TORCH AND TARGET sk_torch)
    add_executable(benchmark_nn benchmark_nn.cpp)
    target_link_libraries(benchmark_nn PRIVATE sk_engine sk_agents sk_torch)
    target_compile_features(benchmark_nn PRIVATE cxx_std_20)
endif()
