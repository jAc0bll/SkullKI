add_library(sk_agents STATIC
    src/random_agent.cpp
    src/heuristic_agent.cpp
)

target_include_directories(sk_agents PUBLIC include)
target_link_libraries(sk_agents PUBLIC sk_engine)
target_compile_features(sk_agents PUBLIC cxx_std_20)
