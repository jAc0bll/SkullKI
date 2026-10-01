add_executable(sk_solver_tests
    test_solver.cpp
)

target_link_libraries(sk_solver_tests PRIVATE sk_solver Catch2::Catch2WithMain)
catch_discover_tests(sk_solver_tests)
