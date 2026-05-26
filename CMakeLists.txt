cmake_minimum_required(VERSION 3.20)
project(skullking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# On Linux the Python module is a shared library; everything linked into it
# (including our static engine/agents/search/nn_torch libs) must be PIC.
# Harmless on Windows.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    # Covers both real MSVC and clang-cl (which CMake also reports as MSVC).
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /bigobj)
    add_compile_options("$<$<CONFIG:Release>:/O2>")
    # clang-cl ignores /arch:AVX2 if not supported; safe to pass.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options("$<$<CONFIG:Release>:/clang:-march=native>")
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
    add_compile_options("$<$<CONFIG:Release>:-O3>")
    add_compile_options("$<$<CONFIG:Release>:-march=native>")
    add_compile_options("$<$<CONFIG:Debug>:-O0>")
    add_compile_options("$<$<CONFIG:Debug>:-g>")
    if(NOT WIN32)
        add_compile_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
        add_link_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
    endif()
endif()

option(SK_BUILD_TESTS  "Build unit tests"                     ON)
option(SK_BUILD_TOOLS  "Build CLI tools (tournament etc.)"   ON)
option(SK_BUILD_PYTHON "Build the Python extension module"   ON)
option(SK_BUILD_TORCH  "Build LibTorch-backed C++ NN inference (TorchScript)" ON)

# Locate LibTorch via the installed PyTorch package so we don't carry a
# duplicate ~200 MB binary in the repo. Users on vast.ai install torch
# with CUDA and we pick the same one up automatically.
if(SK_BUILD_TORCH)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_FOUND)
        message(STATUS "[Torch] Python3 not found; disabling SK_BUILD_TORCH.")
        set(SK_BUILD_TORCH OFF)
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE TORCH_FIND_RC
        )
        if(NOT TORCH_FIND_RC EQUAL 0)
            message(STATUS "[Torch] python -c 'import torch' failed; disabling SK_BUILD_TORCH.")
            set(SK_BUILD_TORCH OFF)
        else()
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
            find_package(Torch REQUIRED)
            message(STATUS "[Torch] Using LibTorch via ${TORCH_CMAKE_PREFIX}")

        endif()
    endif()
endif()

add_subdirectory(engine)
add_subdirectory(agents)
add_subdirectory(search)
if(SK_BUILD_TORCH)
    add_subdirectory(nn_torch)
endif()

if(SK_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(SK_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(SK_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)

    add_subdirectory(engine/tests)
    add_subdirectory(agents/tests)
    add_subdirectory(search/tests)
endif()
