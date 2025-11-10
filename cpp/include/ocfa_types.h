/**
 * @file ocfa_types.h
 * @brief OCFA Face SDK - Type Definitions
 *
 * @copyright Copyright (c) 2025 OCTA Team
 */

#ifndef OCFA_TYPES_H
#define OCFA_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define OCFA_FEATURE_DIM 512        ///< Feature vector dimension
#define OCFA_USER_ID_LEN 16         ///< User ID length (bytes)
#define OCFA_MAX_NAME_LEN 64        ///< Maximum name length
#define OCFA_MAX_ERROR_MSG 128      ///< Maximum error message length

/**
 * @brief Face attributes (optional, if models support)
 */
typedef struct {
    bool has_age;                   ///< Has age prediction
    int age;                        ///< Age (years)

    bool has_gender;                ///< Has gender prediction
    int gender;                     ///< Gender: 0-unknown, 1-male, 2-female
    float gender_confidence;        ///< Gender confidence [0.0, 1.0]

    bool has_mask;                  ///< Has mask detection
    bool wearing_mask;              ///< Whether wearing mask
    float mask_confidence;          ///< Mask confidence [0.0, 1.0]

    bool has_expression;            ///< Has expression recognition
    int expression;                 ///< Expression: 0-neutral, 1-smile, 2-surprise, 3-other
    float expression_confidence;    ///< Expression confidence [0.0, 1.0]
} ocfa_face_attributes_t;

/**
 * @brief Recognition result (up to feature extraction)
 */
typedef struct {
    // Liveness detection result
    bool liveness_passed;           ///< Liveness check passed
    float liveness_score;           ///< Liveness confidence [0.0, 1.0]

    // Quality assessment result
    bool quality_passed;            ///< Quality check passed
    float quality_score;            ///< Quality score [0.0, 1.0]

    // Feature extraction result
    bool feature_extracted;         ///< Feature extracted successfully
    float feature[OCFA_FEATURE_DIM];///< Feature vector (512-dim)

    // Optional face attributes
    ocfa_face_attributes_t attributes; ///< Face attributes

    // Performance metrics
    uint32_t total_time_ms;         ///< Total processing time (ms)

    // Error information
    int error_code;                 ///< Error code (0 = success)
    char error_msg[OCFA_MAX_ERROR_MSG]; ///< Error message
} ocfa_recognition_result_t;

/**
 * @brief Search result (user ID + similarity)
 */
typedef struct {
    uint8_t user_id[OCFA_USER_ID_LEN]; ///< User ID (16 bytes)
    float similarity;                   ///< Similarity score [0.0, 1.0]
} ocfa_search_result_t;

/**
 * @brief SDK configuration
 */
typedef struct {
    const char* model_dir;          ///< Model directory path
    const char* config_file;        ///< Configuration file path
    float liveness_threshold;       ///< Liveness threshold [0.0, 1.0]
    float quality_threshold;        ///< Quality threshold [0.0, 1.0]
    int num_threads;                ///< Number of threads
} ocfa_config_t;

#ifdef __cplusplus
}
#endif

#endif // OCFA_TYPES_H
