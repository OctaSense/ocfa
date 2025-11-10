/**
 * @file math_utils.cpp
 * @brief Math utility functions
 */

#include <cmath>
#include <algorithm>

namespace ocfa {
namespace utils {

/**
 * @brief L2 normalize a feature vector in-place
 */
void L2Normalize(float* feature, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm += feature[i] * feature[i];
    }
    norm = std::sqrt(norm);

    if (norm > 1e-10f) {
        for (int i = 0; i < dim; ++i) {
            feature[i] /= norm;
        }
    }
}

/**
 * @brief Compute cosine similarity between two vectors
 */
float CosineSimilarity(const float* feat1, const float* feat2, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += feat1[i] * feat2[i];
    }
    return std::max(0.0f, std::min(1.0f, dot));
}

/**
 * @brief Compute Euclidean distance
 */
float EuclideanDistance(const float* feat1, const float* feat2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Softmax activation
 */
void Softmax(float* data, int dim) {
    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < dim; ++i) {
        max_val = std::max(max_val, data[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    // Normalize
    for (int i = 0; i < dim; ++i) {
        data[i] /= sum;
    }
}

} // namespace utils
} // namespace ocfa
