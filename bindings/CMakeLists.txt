include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(skullking pyengine.cpp)
target_link_libraries(skullking PRIVATE sk_engine sk_agents sk_search)
target_compile_features(skullking PRIVATE cxx_std_20)

# When LibTorch is available, also expose TorchModelEvaluator, BeliefEvaluator,
# and NeuralMCTSAgent. Keeps the no-torch build path working.
if(SK_BUILD_TORCH AND TARGET sk_torch)
    target_link_libraries(skullking PRIVATE sk_torch)
    target_compile_definitions(skullking PRIVATE SK_HAS_TORCH)
endif()

# Put the produced .pyd next to a stable path so the demo script can import it.
set_target_properties(skullking PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
