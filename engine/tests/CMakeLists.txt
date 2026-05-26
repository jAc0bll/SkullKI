add_executable(sk_tests
    test_cards.cpp
    test_rules.cpp
    test_scoring.cpp
    test_game.cpp
    test_fuzz.cpp
    test_observation.cpp
    test_action_index.cpp
)

target_link_libraries(sk_tests PRIVATE sk_engine Catch2::Catch2WithMain)
catch_discover_tests(sk_tests)
