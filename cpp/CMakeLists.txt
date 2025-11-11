cmake_minimum_required(VERSION 3.16)
project(ocfa_faceid_sdk VERSION 1.0.1 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ARM NEON optimization (only for Linux ARM, not macOS)
if((CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=hard")
    add_definitions(-DUSE_NEON)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # macOS ARM64 (Apple Silicon) - NEON support is implicit
    add_definitions(-DUSE_NEON)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_ONNXRUNTIME "Use ONNX Runtime inference engine" ON)
option(USE_NNIE "Use Hi3516CV610 NNIE inference engine" OFF)
option(USE_OPENCV "Use OpenCV for image processing" OFF)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Source files
set(SDK_SOURCES
    # Core
    src/core/sdk.cpp
    src/core/errors.cpp

    # Utils
    src/utils/image_utils.cpp
    src/utils/math_utils.cpp
    src/utils/neon_utils.cpp
)

# Inference engine sources
if(USE_ONNXRUNTIME)
    list(APPEND SDK_SOURCES src/inference/onnx_engine.cpp)
    add_definitions(-DUSE_ONNX_RUNTIME)
endif()

if(USE_NNIE)
    list(APPEND SDK_SOURCES src/inference/nnie_engine.cpp)
    add_definitions(-DUSE_NNIE)
endif()

list(APPEND SDK_SOURCES src/inference/engine_factory.cpp)

# Create SDK library
add_library(ocfa_face_sdk SHARED ${SDK_SOURCES})
add_library(ocfa_face_sdk_static STATIC ${SDK_SOURCES})

# Set library output name
set_target_properties(ocfa_face_sdk PROPERTIES OUTPUT_NAME "ocfa_face")
set_target_properties(ocfa_face_sdk_static PROPERTIES OUTPUT_NAME "ocfa_face")

# Find third-party libraries

# ONNX Runtime
if(USE_ONNXRUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
            /usr/local/include/onnxruntime
            /usr/include/onnxruntime
    )

    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib
            /usr/local/lib
            /usr/lib
    )

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(ocfa_face_sdk ${ONNXRUNTIME_LIB})
        target_link_libraries(ocfa_face_sdk_static ${ONNXRUNTIME_LIB})
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    else()
        message(FATAL_ERROR "ONNX Runtime not found")
    endif()
endif()

# Hi3516CV610 NNIE SDK
if(USE_NNIE)
    find_path(NNIE_INCLUDE_DIR hi_nnie.h
        PATHS
            $ENV{NNIE_SDK_PATH}/include
            /usr/local/include/nnie
    )

    find_library(NNIE_LIB nnie
        PATHS
            $ENV{NNIE_SDK_PATH}/lib
            /usr/local/lib
    )

    if(NNIE_INCLUDE_DIR AND NNIE_LIB)
        include_directories(${NNIE_INCLUDE_DIR})
        target_link_libraries(ocfa_face_sdk ${NNIE_LIB})
        target_link_libraries(ocfa_face_sdk_static ${NNIE_LIB})
        message(STATUS "NNIE SDK found: ${NNIE_LIB}")
    else()
        message(FATAL_ERROR "NNIE SDK not found. Set NNIE_SDK_PATH environment variable")
    endif()
endif()

# OpenCV (optional)
if(USE_OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ocfa_face_sdk ${OpenCV_LIBS})
    target_link_libraries(ocfa_face_sdk_static ${OpenCV_LIBS})
    add_definitions(-DUSE_OPENCV)
endif()

# pthread
find_package(Threads REQUIRED)
target_link_libraries(ocfa_face_sdk Threads::Threads)
target_link_libraries(ocfa_face_sdk_static Threads::Threads)

# Math library
target_link_libraries(ocfa_face_sdk m)
target_link_libraries(ocfa_face_sdk_static m)

# Install targets
install(TARGETS ocfa_face_sdk ocfa_face_sdk_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include/ocfa_face)

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "====== OCFA FaceID SDK Configuration ======")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Use ONNX Runtime: ${USE_ONNXRUNTIME}")
message(STATUS "Use NNIE: ${USE_NNIE}")
message(STATUS "Use OpenCV: ${USE_OPENCV}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "=========================================")
message(STATUS "")
