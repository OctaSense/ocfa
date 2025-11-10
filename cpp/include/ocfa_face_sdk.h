/**
 * @file ocfa_face_sdk.h
 * @brief OCFA Face SDK - Main API Header
 *
 * High-performance face recognition SDK with RGB-IR liveness detection,
 * feature extraction, and face matching capabilities.
 *
 * @copyright Copyright (c) 2025 OCTA Team
 * @version 1.0.0
 */

#ifndef OCFA_FACE_SDK_H
#define OCFA_FACE_SDK_H

#include "ocfa_types.h"
#include "ocfa_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

//========== Initialization and Cleanup ==========

/**
 * @brief Initialize face recognition SDK
 *
 * @param config SDK configuration
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note Must be called before any other SDK functions
 */
int ocfa_init(const ocfa_config_t* config);

/**
 * @brief Release SDK resources
 *
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note Should be called when SDK is no longer needed
 */
int ocfa_release(void);

//========== Face Recognition Pipeline ==========

/**
 * @brief Face recognition pipeline (liveness -> quality -> feature extraction)
 *
 * @param rgb_image RGB image data (720p, BGR format)
 * @param ir_image IR image data (720p, grayscale)
 * @param width Image width (e.g., 1280)
 * @param height Image height (e.g., 720)
 * @param result Output recognition result (includes feature vector)
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note This function performs the complete recognition pipeline:
 *       1. Preprocessing
 *       2. Liveness detection
 *       3. Quality assessment
 *       4. Feature extraction
 *       5. Feature fusion
 *       Result contains feature vector, not comparison result
 */
int ocfa_recognize(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    ocfa_recognition_result_t* result
);

//========== Step-by-step Interfaces (Optional) ==========

/**
 * @brief Liveness detection only
 *
 * @param rgb_image RGB image data
 * @param ir_image IR image data
 * @param width Image width
 * @param height Image height
 * @param liveness_score Output liveness confidence [0.0, 1.0]
 * @return OCFA_SUCCESS on success, error code otherwise
 */
int ocfa_detect_liveness(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* liveness_score
);

/**
 * @brief Quality assessment only
 *
 * @param rgb_image RGB image data
 * @param width Image width
 * @param height Image height
 * @param quality_score Output quality score [0.0, 1.0]
 * @return OCFA_SUCCESS on success, error code otherwise
 */
int ocfa_assess_quality(
    const uint8_t* rgb_image,
    int width,
    int height,
    float* quality_score
);

/**
 * @brief Feature extraction only
 *
 * @param rgb_image RGB image data
 * @param ir_image IR image data
 * @param width Image width
 * @param height Image height
 * @param feature Output feature vector (512-dim)
 * @return OCFA_SUCCESS on success, error code otherwise
 */
int ocfa_extract_feature(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* feature
);

//========== Feature Comparison ==========

/**
 * @brief Compare two feature vectors (1:1 verification)
 *
 * @param feature1 First feature vector (512-dim)
 * @param feature2 Second feature vector (512-dim)
 * @return Similarity score [0.0, 1.0], higher = more similar
 *
 * @note Features should be L2 normalized
 */
float ocfa_compare_feature(
    const float* feature1,
    const float* feature2
);

//========== Feature Database Management ==========

/**
 * @brief Add user to feature database
 *
 * @param user_id User ID (16 bytes)
 * @param feature User feature vector (512-dim)
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note User metadata (name, permissions) managed by caller
 */
int ocfa_add_user(
    const uint8_t user_id[OCFA_USER_ID_LEN],
    const float* feature
);

/**
 * @brief Update user feature
 *
 * @param user_id User ID (16 bytes)
 * @param feature New feature vector (512-dim)
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note Returns OCFA_ERROR_USER_NOT_FOUND if user doesn't exist
 */
int ocfa_update_user(
    const uint8_t user_id[OCFA_USER_ID_LEN],
    const float* feature
);

/**
 * @brief Remove user from database
 *
 * @param user_id User ID (16 bytes)
 * @return OCFA_SUCCESS on success, error code otherwise
 */
int ocfa_remove_user(
    const uint8_t user_id[OCFA_USER_ID_LEN]
);

/**
 * @brief Search for most similar user (1:1 mode)
 *
 * @param query_feature Query feature vector (512-dim)
 * @param user_id Output user ID of best match (16 bytes)
 * @param similarity Output similarity score [0.0, 1.0]
 * @return OCFA_SUCCESS on success, error code otherwise
 *
 * @note Returns the single user with highest similarity
 */
int ocfa_search_user(
    const float* query_feature,
    uint8_t user_id[OCFA_USER_ID_LEN],
    float* similarity
);

/**
 * @brief Search for multiple similar users (1:N mode)
 *
 * @param query_feature Query feature vector (512-dim)
 * @param threshold Minimum similarity threshold [0.0, 1.0]
 * @param results Output results array (pre-allocated by caller)
 * @param max_results Maximum number of results (size of results array)
 * @return Number of matches found (>=0), or error code (<0)
 *
 * @note Results are sorted by similarity (descending)
 * @note Only returns users with similarity >= threshold
 */
int ocfa_search_users(
    const float* query_feature,
    float threshold,
    ocfa_search_result_t* results,
    int max_results
);

//========== Utility Functions ==========

/**
 * @brief Get SDK version string
 *
 * @return Version string (e.g., "1.0.0")
 */
const char* ocfa_get_version(void);

/**
 * @brief Get error description string
 *
 * @param error_code Error code
 * @return Error description string
 */
const char* ocfa_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // OCFA_FACE_SDK_H
