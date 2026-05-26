add_library(sk_torch STATIC
    src/torch_model.cpp
    src/torch_nn_agent.cpp
    src/neural_mcts.cpp
    src/belief_model.cpp
)

target_include_directories(sk_torch PUBLIC include)
target_link_libraries(sk_torch
    PUBLIC sk_engine sk_agents sk_search
    PUBLIC "${TORCH_LIBRARIES}"
)
target_compile_features(sk_torch PUBLIC cxx_std_20)

# Some PyTorch DLLs (c10.dll, torch_cpu.dll, ...) live in the python site-packages
# and need to be discoverable by executables that link sk_torch. We don't copy them;
# the user is expected to set PATH or run from a shell that has torch importable.
# The required directory is "${TORCH_INSTALL_PREFIX}/lib" on Windows.
if(WIN32 AND DEFINED TORCH_INSTALL_PREFIX)
    set(SK_TORCH_DLL_DIR "${TORCH_INSTALL_PREFIX}/lib" CACHE PATH "" FORCE)
    message(STATUS "[Torch] Runtime DLL dir: ${SK_TORCH_DLL_DIR}")
endif()
