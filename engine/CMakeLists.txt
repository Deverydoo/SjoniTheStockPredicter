# ============================================================================
# ArgusTrader C++ Engine - CMake Build Configuration
# ============================================================================

cmake_minimum_required(VERSION 3.20)

project(ArgusTrader
    VERSION 1.0.0
    DESCRIPTION "High-performance NASDAQ trading engine"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================

option(ARGUS_BUILD_TESTS "Build unit tests" OFF)
option(ARGUS_ENABLE_CUDA "Enable CUDA/TensorRT (for later)" OFF)

# ============================================================================
# C++ Standard
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")

# ============================================================================
# Compiler Flags
# ============================================================================

if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_options("$<$<CONFIG:Release>:/O2>")
    add_compile_options("$<$<CONFIG:Debug>:/Od;/Zi>")
    add_definitions(-D_WIN32_WINNT=0x0601)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options("$<$<CONFIG:Release>:-O3;-march=native>")
    add_compile_options("$<$<CONFIG:Debug>:-g;-O0>")
endif()

# ============================================================================
# Dependencies
# ============================================================================

find_package(Threads REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)

# nlohmann_json
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# spdlog
find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# ZeroMQ (optional) - Try vcpkg first, then conda environment
option(ARGUS_ENABLE_ZMQ "Enable ZeroMQ support" ON)

set(ZMQ_FOUND FALSE)
if(ARGUS_ENABLE_ZMQ)
    # Try vcpkg/system first
    find_package(cppzmq CONFIG QUIET)
    if(cppzmq_FOUND)
        set(ZMQ_FOUND TRUE)
    else()
        # Try conda environment
        set(CONDA_ZMQ_PATH "C:/Users/craig/miniconda3/envs/py310/Library")
        if(EXISTS "${CONDA_ZMQ_PATH}/include/zmq.h")
            message(STATUS "Found ZeroMQ in conda environment")
            set(ZMQ_FOUND TRUE)
            set(ZMQ_INCLUDE_DIR "${CONDA_ZMQ_PATH}/include")
            set(ZMQ_LIBRARY "${CONDA_ZMQ_PATH}/lib/libzmq.lib")
        endif()
    endif()
endif()

# ONNX Runtime (optional) - for ML model inference
option(ARGUS_ENABLE_ONNX "Enable ONNX Runtime support" ON)

set(ONNX_FOUND FALSE)
if(ARGUS_ENABLE_ONNX)
    # Check for local ONNX Runtime installation (GPU version preferred)
    set(ONNX_ROOT "D:/Vibe_Projects/The Trader/onnx")
    set(ONNX_GPU_PATH "${ONNX_ROOT}/onnxruntime-win-x64-gpu-1.23.2")
    set(ONNX_CPU_PATH "${ONNX_ROOT}/onnxruntime-win-x64-1.23.2")

    if(EXISTS "${ONNX_GPU_PATH}/include/onnxruntime_cxx_api.h")
        message(STATUS "Found ONNX Runtime GPU at ${ONNX_GPU_PATH}")
        set(ONNX_FOUND TRUE)
        set(ONNX_INCLUDE_DIR "${ONNX_GPU_PATH}/include")
        set(ONNX_LIB_DIR "${ONNX_GPU_PATH}/lib")
        set(ONNX_LIBRARY "${ONNX_LIB_DIR}/onnxruntime.lib")
        set(ONNX_HAS_CUDA TRUE)
    elseif(EXISTS "${ONNX_CPU_PATH}/include/onnxruntime_cxx_api.h")
        message(STATUS "Found ONNX Runtime CPU at ${ONNX_CPU_PATH}")
        set(ONNX_FOUND TRUE)
        set(ONNX_INCLUDE_DIR "${ONNX_CPU_PATH}/include")
        set(ONNX_LIB_DIR "${ONNX_CPU_PATH}/lib")
        set(ONNX_LIBRARY "${ONNX_LIB_DIR}/onnxruntime.lib")
        set(ONNX_HAS_CUDA FALSE)
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================

set(ARGUS_SOURCES
    src/main.cpp
)

# ============================================================================
# Executable
# ============================================================================

add_executable(argus_engine ${ARGUS_SOURCES})

target_include_directories(argus_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(argus_engine PRIVATE
    Threads::Threads
    Boost::system
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

if(ZMQ_FOUND)
    if(cppzmq_FOUND)
        target_link_libraries(argus_engine PRIVATE cppzmq)
    else()
        # Use conda ZMQ
        target_include_directories(argus_engine PRIVATE ${ZMQ_INCLUDE_DIR})
        target_link_libraries(argus_engine PRIVATE ${ZMQ_LIBRARY})
    endif()
    target_compile_definitions(argus_engine PRIVATE ARGUS_ZMQ_ENABLED)
    message(STATUS "ZeroMQ: ENABLED")
else()
    message(STATUS "ZeroMQ: NOT FOUND (optional)")
endif()

if(ONNX_FOUND)
    target_include_directories(argus_engine PRIVATE ${ONNX_INCLUDE_DIR})
    target_link_libraries(argus_engine PRIVATE ${ONNX_LIBRARY})
    target_compile_definitions(argus_engine PRIVATE ARGUS_ONNX_ENABLED)
    if(ONNX_HAS_CUDA)
        target_compile_definitions(argus_engine PRIVATE ARGUS_ONNX_CUDA)
    endif()
    message(STATUS "ONNX Runtime: ENABLED (CUDA: ${ONNX_HAS_CUDA})")
else()
    message(STATUS "ONNX Runtime: NOT FOUND (optional)")
endif()

# ============================================================================
# Tests (optional)
# ============================================================================

if(ARGUS_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(argus_tests tests/test_main.cpp)
    target_include_directories(argus_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(argus_tests PRIVATE GTest::gtest GTest::gtest_main Threads::Threads)

    include(GoogleTest)
    gtest_discover_tests(argus_tests)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "============================================")
message(STATUS "ArgusTrader Build Configuration")
message(STATUS "============================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Boost:        ${Boost_VERSION}")
message(STATUS "OpenSSL:      ${OPENSSL_VERSION}")
message(STATUS "============================================")
