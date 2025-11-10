/**
 * @file math_utils.h
 * @brief Math utility functions header
 */

#ifndef OCFA_MATH_UTILS_H
#define OCFA_MATH_UTILS_H

namespace ocfa {
namespace utils {

/**
 * @brief L2 normalize a feature vector in-place
 */
void L2Normalize(float* feature, int dim);

/**
 * @brief Compute cosine similarity between two vectors
 */
float CosineSimilarity(const float* feat1, const float* feat2, int dim);

/**
 * @brief Compute Euclidean distance
 */
float EuclideanDistance(const float* feat1, const float* feat2, int dim);

/**
 * @brief Softmax activation
 */
void Softmax(float* data, int dim);

} // namespace utils
} // namespace ocfa

#endif // OCFA_MATH_UTILS_H
