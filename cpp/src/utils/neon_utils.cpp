/**
 * @file neon_utils.cpp
 * @brief ARM NEON SIMD optimized utility functions implementation
 */

#include "neon_utils.h"
#include <cmath>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace ocfa {
namespace utils {
namespace neon {

bool IsNeonAvailable() {
#ifdef __ARM_NEON
    return true;
#else
    return false;
#endif
}

#ifdef __ARM_NEON

void L2NormalizeNeon(float* feature, int dim) {
    // Compute sum of squares using NEON
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 4 elements at a time
    for (; i + 3 < dim; i += 4) {
        float32x4_t vec = vld1q_f32(feature + i);
        sum_vec = vmlaq_f32(sum_vec, vec, vec);  // sum += vec * vec
    }

    // Horizontal sum of sum_vec
    float32x2_t sum_low = vget_low_f32(sum_vec);
    float32x2_t sum_high = vget_high_f32(sum_vec);
    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
    float sum = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Handle remaining elements (if dim not multiple of 4)
    for (; i < dim; ++i) {
        sum += feature[i] * feature[i];
    }

    // Compute norm and reciprocal
    float norm = std::sqrt(sum);
    if (norm < 1e-10f) {
        return;  // Avoid division by zero
    }

    float inv_norm = 1.0f / norm;
    float32x4_t inv_norm_vec = vdupq_n_f32(inv_norm);

    // Normalize using NEON
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vec = vld1q_f32(feature + i);
        vec = vmulq_f32(vec, inv_norm_vec);
        vst1q_f32(feature + i, vec);
    }

    // Handle remaining elements
    for (; i < dim; ++i) {
        feature[i] *= inv_norm;
    }
}

float CosineSimilarityNeon(const float* feat1, const float* feat2, int dim) {
    // Compute dot product using NEON
    float32x4_t dot_vec = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 16 elements at a time (4 NEON registers)
    for (; i + 15 < dim; i += 16) {
        // Load 4x4 = 16 elements
        float32x4_t a0 = vld1q_f32(feat1 + i);
        float32x4_t b0 = vld1q_f32(feat2 + i);
        float32x4_t a1 = vld1q_f32(feat1 + i + 4);
        float32x4_t b1 = vld1q_f32(feat2 + i + 4);
        float32x4_t a2 = vld1q_f32(feat1 + i + 8);
        float32x4_t b2 = vld1q_f32(feat2 + i + 8);
        float32x4_t a3 = vld1q_f32(feat1 + i + 12);
        float32x4_t b3 = vld1q_f32(feat2 + i + 12);

        // Multiply-accumulate
        dot_vec = vmlaq_f32(dot_vec, a0, b0);
        dot_vec = vmlaq_f32(dot_vec, a1, b1);
        dot_vec = vmlaq_f32(dot_vec, a2, b2);
        dot_vec = vmlaq_f32(dot_vec, a3, b3);
    }

    // Process remaining 4-element chunks
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(feat1 + i);
        float32x4_t b = vld1q_f32(feat2 + i);
        dot_vec = vmlaq_f32(dot_vec, a, b);
    }

    // Horizontal sum
    float32x2_t dot_low = vget_low_f32(dot_vec);
    float32x2_t dot_high = vget_high_f32(dot_vec);
    float32x2_t dot_pair = vadd_f32(dot_low, dot_high);
    float dot = vget_lane_f32(vpadd_f32(dot_pair, dot_pair), 0);

    // Handle remaining elements
    for (; i < dim; ++i) {
        dot += feat1[i] * feat2[i];
    }

    // Clamp to [0, 1]
    return std::max(0.0f, std::min(1.0f, dot));
}

void BatchCosineSimilarityNeon(
    const float* query,
    const float* features,
    int num_features,
    int dim,
    float* similarities
) {
    // Compute similarity for each feature in the database
    for (int i = 0; i < num_features; ++i) {
        similarities[i] = CosineSimilarityNeon(query, features + i * dim, dim);
    }
}

void VectorAddNeon(const float* a, const float* b, float* result, int dim) {
    int i = 0;

    // Process 16 elements at a time
    for (; i + 15 < dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vaddq_f32(a0, b0));
        vst1q_f32(result + i + 4, vaddq_f32(a1, b1));
        vst1q_f32(result + i + 8, vaddq_f32(a2, b2));
        vst1q_f32(result + i + 12, vaddq_f32(a3, b3));
    }

    // Process 4 elements at a time
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(result + i, vaddq_f32(va, vb));
    }

    // Handle remaining elements
    for (; i < dim; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorMulNeon(const float* a, const float* b, float* result, int dim) {
    int i = 0;

    // Process 16 elements at a time
    for (; i + 15 < dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(result + i, vmulq_f32(a0, b0));
        vst1q_f32(result + i + 4, vmulq_f32(a1, b1));
        vst1q_f32(result + i + 8, vmulq_f32(a2, b2));
        vst1q_f32(result + i + 12, vmulq_f32(a3, b3));
    }

    // Process 4 elements at a time
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(result + i, vmulq_f32(va, vb));
    }

    // Handle remaining elements
    for (; i < dim; ++i) {
        result[i] = a[i] * b[i];
    }
}

#else

// Fallback implementations for non-NEON platforms

void L2NormalizeNeon(float* feature, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += feature[i] * feature[i];
    }
    float norm = std::sqrt(sum);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; ++i) {
            feature[i] /= norm;
        }
    }
}

float CosineSimilarityNeon(const float* feat1, const float* feat2, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += feat1[i] * feat2[i];
    }
    return std::max(0.0f, std::min(1.0f, dot));
}

void BatchCosineSimilarityNeon(
    const float* query,
    const float* features,
    int num_features,
    int dim,
    float* similarities
) {
    for (int i = 0; i < num_features; ++i) {
        similarities[i] = CosineSimilarityNeon(query, features + i * dim, dim);
    }
}

void VectorAddNeon(const float* a, const float* b, float* result, int dim) {
    for (int i = 0; i < dim; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorMulNeon(const float* a, const float* b, float* result, int dim) {
    for (int i = 0; i < dim; ++i) {
        result[i] = a[i] * b[i];
    }
}

#endif // __ARM_NEON

} // namespace neon
} // namespace utils
} // namespace ocfa
