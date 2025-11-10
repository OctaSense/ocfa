/**
 * @file ocfa_errors.h
 * @brief OCFA Face SDK - Error Code Definitions
 *
 * @copyright Copyright (c) 2025 OCTA Team
 */

#ifndef OCFA_ERRORS_H
#define OCFA_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error codes
 */
enum OCFAErrorCode {
    OCFA_SUCCESS = 0,                   ///< Success

    // General errors (1-99)
    OCFA_ERROR_INVALID_PARAM = 1,       ///< Invalid parameter
    OCFA_ERROR_OUT_OF_MEMORY = 2,       ///< Out of memory
    OCFA_ERROR_NOT_INITIALIZED = 3,     ///< SDK not initialized
    OCFA_ERROR_ALREADY_INITIALIZED = 4, ///< SDK already initialized
    OCFA_ERROR_FILE_NOT_FOUND = 5,      ///< File not found
    OCFA_ERROR_INVALID_FORMAT = 6,      ///< Invalid file format

    // Image processing errors (100-199)
    OCFA_ERROR_INVALID_IMAGE = 100,     ///< Invalid image data
    OCFA_ERROR_INVALID_IMAGE_SIZE = 101,///< Invalid image size
    OCFA_ERROR_PREPROCESS_FAILED = 102, ///< Preprocessing failed

    // Liveness detection errors (200-299)
    OCFA_ERROR_LIVENESS_FAILED = 200,   ///< Liveness detection failed
    OCFA_ERROR_LIVENESS_MODEL_ERROR = 201, ///< Liveness model error

    // Quality assessment errors (300-399)
    OCFA_ERROR_QUALITY_FAILED = 300,    ///< Quality assessment failed
    OCFA_ERROR_QUALITY_TOO_LOW = 301,   ///< Quality too low

    // Feature extraction errors (400-499)
    OCFA_ERROR_FEATURE_FAILED = 400,    ///< Feature extraction failed
    OCFA_ERROR_FEATURE_MODEL_ERROR = 401, ///< Feature model error

    // Database errors (500-599)
    OCFA_ERROR_USER_NOT_FOUND = 500,    ///< User not found
    OCFA_ERROR_USER_ALREADY_EXISTS = 501, ///< User already exists
    OCFA_ERROR_DATABASE_FULL = 502,     ///< Database full
    OCFA_ERROR_INVALID_USER_ID = 503,   ///< Invalid user ID

    // Inference engine errors (600-699)
    OCFA_ERROR_MODEL_LOAD_FAILED = 600, ///< Model loading failed
    OCFA_ERROR_INFERENCE_FAILED = 601,  ///< Inference failed
    OCFA_ERROR_UNSUPPORTED_DEVICE = 602,///< Unsupported device

    // Unknown error
    OCFA_ERROR_UNKNOWN = 999            ///< Unknown error
};

/**
 * @brief Get error string from error code
 *
 * @param error_code Error code
 * @return Error description string
 */
const char* ocfa_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // OCFA_ERRORS_H
