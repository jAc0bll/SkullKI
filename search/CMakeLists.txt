add_library(sk_search STATIC
    src/determinize.cpp
    src/ismcts.cpp
)

target_include_directories(sk_search PUBLIC include)
target_link_libraries(sk_search PUBLIC sk_engine sk_agents)
target_compile_features(sk_search PUBLIC cxx_std_20)
