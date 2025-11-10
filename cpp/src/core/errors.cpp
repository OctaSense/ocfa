/**
 * @file errors.cpp
 * @brief Error code to string conversion
 */

#include "ocfa_errors.h"

const char* ocfa_get_error_string(int error_code) {
    switch (error_code) {
        case OCFA_SUCCESS:
            return "Success";

        // General errors
        case OCFA_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case OCFA_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case OCFA_ERROR_NOT_INITIALIZED:
            return "SDK not initialized";
        case OCFA_ERROR_ALREADY_INITIALIZED:
            return "SDK already initialized";
        case OCFA_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case OCFA_ERROR_INVALID_FORMAT:
            return "Invalid file format";

        // Image processing errors
        case OCFA_ERROR_INVALID_IMAGE:
            return "Invalid image data";
        case OCFA_ERROR_INVALID_IMAGE_SIZE:
            return "Invalid image size";
        case OCFA_ERROR_PREPROCESS_FAILED:
            return "Preprocessing failed";

        // Liveness detection errors
        case OCFA_ERROR_LIVENESS_FAILED:
            return "Liveness detection failed";
        case OCFA_ERROR_LIVENESS_MODEL_ERROR:
            return "Liveness model error";

        // Quality assessment errors
        case OCFA_ERROR_QUALITY_FAILED:
            return "Quality assessment failed";
        case OCFA_ERROR_QUALITY_TOO_LOW:
            return "Quality too low";

        // Feature extraction errors
        case OCFA_ERROR_FEATURE_FAILED:
            return "Feature extraction failed";
        case OCFA_ERROR_FEATURE_MODEL_ERROR:
            return "Feature model error";

        // Database errors
        case OCFA_ERROR_USER_NOT_FOUND:
            return "User not found";
        case OCFA_ERROR_USER_ALREADY_EXISTS:
            return "User already exists";
        case OCFA_ERROR_DATABASE_FULL:
            return "Database full";
        case OCFA_ERROR_INVALID_USER_ID:
            return "Invalid user ID";

        // Inference engine errors
        case OCFA_ERROR_MODEL_LOAD_FAILED:
            return "Model loading failed";
        case OCFA_ERROR_INFERENCE_FAILED:
            return "Inference failed";
        case OCFA_ERROR_UNSUPPORTED_DEVICE:
            return "Unsupported device";

        default:
            return "Unknown error";
    }
}
