/**
 * @file benchmark_neon.cpp
 * @brief Benchmark NEON-optimized functions vs. standard implementations
 */

#include "../src/utils/neon_utils.h"
#include "../src/utils/math_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

using namespace ocfa::utils;

// Helper to generate random feature vector
void generate_random_feature(float* feature, int dim) {
    for (int i = 0; i < dim; ++i) {
        feature[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

// Benchmark function template
template<typename Func>
double benchmark(const char* name, Func&& func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = static_cast<double>(duration.count()) / iterations;
    double avg_time_ms = avg_time_us / 1000.0;

    printf("%-40s: %.3f µs (%.6f ms)\n", name, avg_time_us, avg_time_ms);

    return avg_time_us;
}

void benchmark_l2_normalize() {
    printf("\n=== L2 Normalize (512-dim) ===\n");

    const int dim = 512;
    const int iterations = 10000;

    std::vector<float> feature_standard(dim);
    std::vector<float> feature_neon(dim);

    generate_random_feature(feature_standard.data(), dim);
    std::memcpy(feature_neon.data(), feature_standard.data(), dim * sizeof(float));

    // Benchmark standard implementation
    double time_standard = benchmark("Standard L2Normalize", [&]() {
        generate_random_feature(feature_standard.data(), dim);
        L2Normalize(feature_standard.data(), dim);
    }, iterations);

    // Benchmark NEON implementation
    double time_neon = benchmark("NEON L2Normalize", [&]() {
        generate_random_feature(feature_neon.data(), dim);
        neon::L2NormalizeNeon(feature_neon.data(), dim);
    }, iterations);

    // Calculate speedup
    double speedup = time_standard / time_neon;
    printf("Speedup: %.2fx\n", speedup);

    // Verify correctness
    generate_random_feature(feature_standard.data(), dim);
    std::memcpy(feature_neon.data(), feature_standard.data(), dim * sizeof(float));

    L2Normalize(feature_standard.data(), dim);
    neon::L2NormalizeNeon(feature_neon.data(), dim);

    float max_diff = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = std::abs(feature_standard[i] - feature_neon[i]);
        max_diff = std::max(max_diff, diff);
    }
    printf("Max difference: %.10f (should be < 1e-6)\n", max_diff);
}

void benchmark_cosine_similarity() {
    printf("\n=== Cosine Similarity (512-dim) ===\n");

    const int dim = 512;
    const int iterations = 10000;

    std::vector<float> feat1(dim);
    std::vector<float> feat2(dim);

    generate_random_feature(feat1.data(), dim);
    generate_random_feature(feat2.data(), dim);
    L2Normalize(feat1.data(), dim);
    L2Normalize(feat2.data(), dim);

    float sim_standard = 0.0f;
    float sim_neon = 0.0f;

    // Benchmark standard implementation
    double time_standard = benchmark("Standard CosineSimilarity", [&]() {
        sim_standard = CosineSimilarity(feat1.data(), feat2.data(), dim);
    }, iterations);

    // Benchmark NEON implementation
    double time_neon = benchmark("NEON CosineSimilarity", [&]() {
        sim_neon = neon::CosineSimilarityNeon(feat1.data(), feat2.data(), dim);
    }, iterations);

    // Calculate speedup
    double speedup = time_standard / time_neon;
    printf("Speedup: %.2fx\n", speedup);

    // Verify correctness
    printf("Standard result: %.6f\n", sim_standard);
    printf("NEON result: %.6f\n", sim_neon);
    printf("Difference: %.10f (should be < 1e-6)\n", std::abs(sim_standard - sim_neon));
}

void benchmark_batch_similarity() {
    printf("\n=== Batch Cosine Similarity (512-dim, 1000 users) ===\n");

    const int dim = 512;
    const int num_users = 1000;
    const int iterations = 100;

    std::vector<float> query(dim);
    std::vector<float> database(num_users * dim);
    std::vector<float> similarities_standard(num_users);
    std::vector<float> similarities_neon(num_users);

    generate_random_feature(query.data(), dim);
    L2Normalize(query.data(), dim);

    for (int i = 0; i < num_users; ++i) {
        generate_random_feature(database.data() + i * dim, dim);
        L2Normalize(database.data() + i * dim, dim);
    }

    // Benchmark standard implementation (sequential)
    double time_standard = benchmark("Standard Batch (sequential)", [&]() {
        for (int i = 0; i < num_users; ++i) {
            similarities_standard[i] = CosineSimilarity(query.data(), database.data() + i * dim, dim);
        }
    }, iterations);

    // Benchmark NEON implementation
    double time_neon = benchmark("NEON Batch", [&]() {
        neon::BatchCosineSimilarityNeon(
            query.data(),
            database.data(),
            num_users,
            dim,
            similarities_neon.data()
        );
    }, iterations);

    // Calculate speedup
    double speedup = time_standard / time_neon;
    printf("Speedup: %.2fx\n", speedup);

    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < num_users; ++i) {
        float diff = std::abs(similarities_standard[i] - similarities_neon[i]);
        max_diff = std::max(max_diff, diff);
    }
    printf("Max difference: %.10f (should be < 1e-6)\n", max_diff);
}

int main() {
    printf("========================================\n");
    printf("OCFA Face SDK - NEON Benchmark\n");
    printf("========================================\n");

    // Check NEON availability
    if (neon::IsNeonAvailable()) {
        printf("NEON: Available ✓\n");
    } else {
        printf("NEON: Not available (running fallback implementations)\n");
    }

    printf("\nNote: Running on ARM NEON-enabled CPU will show significant speedup\n");
    printf("On x86/x64, both implementations will use the same fallback code\n");

    // Run benchmarks
    benchmark_l2_normalize();
    benchmark_cosine_similarity();
    benchmark_batch_similarity();

    printf("\n========================================\n");
    printf("Benchmark completed!\n");
    printf("========================================\n");

    return 0;
}
