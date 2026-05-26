add_executable(sk_search_tests
    test_determinize.cpp
    test_ismcts.cpp
)
target_link_libraries(sk_search_tests PRIVATE sk_engine sk_agents sk_search Catch2::Catch2WithMain)
catch_discover_tests(sk_search_tests)
