# Examples CMakeLists.txt

# Basic example (database operations only)
add_executable(demo_basic demo_basic.cpp)
target_link_libraries(demo_basic ocfa_face_sdk)

# Recognition example (complete pipeline)
add_executable(demo_recognition demo_recognition.cpp)
target_link_libraries(demo_recognition ocfa_face_sdk)

# NEON benchmark
add_executable(benchmark_neon benchmark_neon.cpp)
target_link_libraries(benchmark_neon ocfa_face_sdk)

# Install examples
install(TARGETS demo_basic demo_recognition benchmark_neon
    RUNTIME DESTINATION bin/examples
)
