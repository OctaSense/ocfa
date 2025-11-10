/**
 * @file neon_utils.h
 * @brief ARM NEON SIMD optimized utility functions
 */

#ifndef OCFA_NEON_UTILS_H
#define OCFA_NEON_UTILS_H

#include <cstdint>

namespace ocfa {
namespace utils {
namespace neon {

/**
 * @brief Check if NEON is available at runtime
 */
bool IsNeonAvailable();

/**
 * @brief NEON-optimized L2 normalization
 * @param feature Feature vector to normalize (in-place)
 * @param dim Feature dimension (must be multiple of 4)
 */
void L2NormalizeNeon(float* feature, int dim);

/**
 * @brief NEON-optimized cosine similarity (dot product of L2-normalized vectors)
 * @param feat1 First feature vector (L2-normalized)
 * @param feat2 Second feature vector (L2-normalized)
 * @param dim Feature dimension (must be multiple of 4)
 * @return Similarity score [0.0, 1.0]
 */
float CosineSimilarityNeon(const float* feat1, const float* feat2, int dim);

/**
 * @brief NEON-optimized batch cosine similarity
 * @param query Query feature (L2-normalized)
 * @param features Database features (L2-normalized, row-major)
 * @param num_features Number of features in database
 * @param dim Feature dimension (must be multiple of 4)
 * @param similarities Output similarities array (pre-allocated)
 */
void BatchCosineSimilarityNeon(
    const float* query,
    const float* features,
    int num_features,
    int dim,
    float* similarities
);

/**
 * @brief NEON-optimized vector addition
 * @param a First vector
 * @param b Second vector
 * @param result Output vector (can be same as a or b)
 * @param dim Vector dimension (must be multiple of 4)
 */
void VectorAddNeon(const float* a, const float* b, float* result, int dim);

/**
 * @brief NEON-optimized element-wise multiplication
 * @param a First vector
 * @param b Second vector
 * @param result Output vector (can be same as a or b)
 * @param dim Vector dimension (must be multiple of 4)
 */
void VectorMulNeon(const float* a, const float* b, float* result, int dim);

} // namespace neon
} // namespace utils
} // namespace ocfa

#endif // OCFA_NEON_UTILS_H
