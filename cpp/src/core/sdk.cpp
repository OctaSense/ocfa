/**
 * @file sdk.cpp
 * @brief OCFA Face SDK - Main Implementation
 */

#include "ocfa_face_sdk.h"
#include "inference_engine.h"
#include "../utils/image_utils.h"
#include "../utils/math_utils.h"
#include "../utils/neon_utils.h"
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

// Version
#define OCFA_VERSION "1.0.0"

// Global SDK state
static bool g_sdk_initialized = false;
static ocfa_config_t g_config;

// Forward declarations of internal classes
class FeatureDatabase;
static std::unique_ptr<FeatureDatabase> g_database;

// Inference engines
static std::unique_ptr<ocfa::inference::InferenceEngine> g_liveness_engine;
static std::unique_ptr<ocfa::inference::InferenceEngine> g_feature_engine;

//========== Feature Database Implementation ==========

class FeatureDatabase {
public:
    FeatureDatabase() : feature_dim_(OCFA_FEATURE_DIM) {}

    bool AddUser(const uint8_t user_id[OCFA_USER_ID_LEN], const float* feature) {
        // Check if user already exists
        for (const auto& entry : database_) {
            if (std::memcmp(entry.user_id, user_id, OCFA_USER_ID_LEN) == 0) {
                return false; // User already exists
            }
        }

        // Add new user
        Entry entry;
        std::memcpy(entry.user_id, user_id, OCFA_USER_ID_LEN);
        entry.feature.assign(feature, feature + feature_dim_);
        database_.push_back(entry);

        return true;
    }

    bool UpdateUser(const uint8_t user_id[OCFA_USER_ID_LEN], const float* feature) {
        for (auto& entry : database_) {
            if (std::memcmp(entry.user_id, user_id, OCFA_USER_ID_LEN) == 0) {
                entry.feature.assign(feature, feature + feature_dim_);
                return true;
            }
        }
        return false; // User not found
    }

    bool RemoveUser(const uint8_t user_id[OCFA_USER_ID_LEN]) {
        for (auto it = database_.begin(); it != database_.end(); ++it) {
            if (std::memcmp(it->user_id, user_id, OCFA_USER_ID_LEN) == 0) {
                database_.erase(it);
                return true;
            }
        }
        return false; // User not found
    }

    bool SearchUser(const float* query_feature, uint8_t user_id[OCFA_USER_ID_LEN], float* similarity) {
        if (database_.empty()) {
            return false;
        }

        float max_sim = -1.0f;
        int best_idx = -1;

        for (size_t i = 0; i < database_.size(); ++i) {
            float sim = ComputeSimilarity(query_feature, database_[i].feature.data());
            if (sim > max_sim) {
                max_sim = sim;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0) {
            std::memcpy(user_id, database_[best_idx].user_id, OCFA_USER_ID_LEN);
            *similarity = max_sim;
            return true;
        }

        return false;
    }

    int SearchUsers(const float* query_feature, float threshold,
                   ocfa_search_result_t* results, int max_results) {
        std::vector<std::pair<float, int>> candidates;

        // Compute similarities
        for (size_t i = 0; i < database_.size(); ++i) {
            float sim = ComputeSimilarity(query_feature, database_[i].feature.data());
            if (sim >= threshold) {
                candidates.emplace_back(sim, static_cast<int>(i));
            }
        }

        // Sort by similarity descending
        std::sort(candidates.begin(), candidates.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        // Fill results
        int count = std::min(static_cast<int>(candidates.size()), max_results);
        for (int i = 0; i < count; ++i) {
            int idx = candidates[i].second;
            std::memcpy(results[i].user_id, database_[idx].user_id, OCFA_USER_ID_LEN);
            results[i].similarity = candidates[i].first;
        }

        return count;
    }

    size_t GetUserCount() const {
        return database_.size();
    }

    void Clear() {
        database_.clear();
    }

private:
    struct Entry {
        uint8_t user_id[OCFA_USER_ID_LEN];
        std::vector<float> feature;
    };

    float ComputeSimilarity(const float* feat1, const float* feat2) const {
        // Use NEON-optimized cosine similarity if available
        if (ocfa::utils::neon::IsNeonAvailable()) {
            return ocfa::utils::neon::CosineSimilarityNeon(feat1, feat2, feature_dim_);
        } else {
            // Fallback: standard implementation
            float dot = 0.0f;
            for (int i = 0; i < feature_dim_; ++i) {
                dot += feat1[i] * feat2[i];
            }
            return std::max(0.0f, std::min(1.0f, dot));
        }
    }

    int feature_dim_;
    std::vector<Entry> database_;
};

//========== SDK Implementation ==========

int ocfa_init(const ocfa_config_t* config) {
    if (g_sdk_initialized) {
        return OCFA_ERROR_ALREADY_INITIALIZED;
    }

    if (config == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    // Copy configuration
    g_config = *config;

    // Initialize feature database
    g_database = std::make_unique<FeatureDatabase>();

    // Initialize inference engines
    std::string engine_type = "onnx"; // Default to ONNX Runtime
    // You can make this configurable through config

    g_liveness_engine = ocfa::inference::CreateEngine(engine_type, g_config.num_threads);
    g_feature_engine = ocfa::inference::CreateEngine(engine_type, g_config.num_threads);

    if (!g_liveness_engine || !g_feature_engine) {
        g_database.reset();
        return OCFA_ERROR_UNSUPPORTED_DEVICE;
    }

    // Load liveness model (MiniFASNet)
    std::string liveness_model_path = std::string(g_config.model_dir) + "/minifasnet_int8.onnx";
    if (!g_liveness_engine->LoadModel(liveness_model_path)) {
        g_liveness_engine.reset();
        g_feature_engine.reset();
        g_database.reset();
        return OCFA_ERROR_MODEL_LOAD_FAILED;
    }

    // Load feature extraction model (ArcFace-R34)
    std::string feature_model_path = std::string(g_config.model_dir) + "/arcface_r34_int8.onnx";
    if (!g_feature_engine->LoadModel(feature_model_path)) {
        g_liveness_engine.reset();
        g_feature_engine.reset();
        g_database.reset();
        return OCFA_ERROR_MODEL_LOAD_FAILED;
    }

    g_sdk_initialized = true;

    return OCFA_SUCCESS;
}

int ocfa_release(void) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    // Release resources
    g_database.reset();
    g_liveness_engine.reset();
    g_feature_engine.reset();

    g_sdk_initialized = false;

    return OCFA_SUCCESS;
}

int ocfa_recognize(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    ocfa_recognition_result_t* result
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (rgb_image == nullptr || ir_image == nullptr || result == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    // Initialize result
    std::memset(result, 0, sizeof(ocfa_recognition_result_t));

    auto start_time = std::chrono::steady_clock::now();

    // 1. Liveness detection
    float liveness_score = 0.0f;
    int ret = ocfa_detect_liveness(rgb_image, ir_image, width, height, &liveness_score);
    result->liveness_score = liveness_score;
    result->liveness_passed = (ret == OCFA_SUCCESS && liveness_score >= g_config.liveness_threshold);

    if (ret != OCFA_SUCCESS) {
        result->error_code = ret;
        snprintf(result->error_msg, OCFA_MAX_ERROR_MSG, "Liveness detection failed");
        return ret;
    }

    // 2. Quality assessment
    float quality_score = 0.0f;
    ret = ocfa_assess_quality(rgb_image, width, height, &quality_score);
    result->quality_score = quality_score;
    result->quality_passed = (ret == OCFA_SUCCESS && quality_score >= g_config.quality_threshold);

    if (ret != OCFA_SUCCESS) {
        result->error_code = ret;
        snprintf(result->error_msg, OCFA_MAX_ERROR_MSG, "Quality assessment failed");
        return ret;
    }

    // 3. Feature extraction
    ret = ocfa_extract_feature(rgb_image, ir_image, width, height, result->feature);
    result->feature_extracted = (ret == OCFA_SUCCESS);

    if (ret != OCFA_SUCCESS) {
        result->error_code = ret;
        snprintf(result->error_msg, OCFA_MAX_ERROR_MSG, "Feature extraction failed");
        return ret;
    }

    // Calculate total time
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result->total_time_ms = static_cast<uint32_t>(duration.count());

    result->error_code = OCFA_SUCCESS;
    return OCFA_SUCCESS;
}

int ocfa_detect_liveness(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* liveness_score
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (rgb_image == nullptr || ir_image == nullptr || liveness_score == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    // Preprocess RGB image (resize to 80x80, normalize)
    constexpr int input_size = 80;
    std::vector<uint8_t> rgb_resized(input_size * input_size * 3);
    ocfa::utils::ResizeImage(rgb_image, width, height, 3,
                             rgb_resized.data(), input_size, input_size);

    std::vector<float> rgb_normalized(input_size * input_size * 3);
    float rgb_mean[3] = {0.485f, 0.456f, 0.406f};
    float rgb_std[3] = {0.229f, 0.224f, 0.225f};
    ocfa::utils::NormalizeAndConvertToCHW(rgb_resized.data(), rgb_normalized.data(),
                                          input_size, input_size, 3,
                                          rgb_mean, rgb_std);

    // Preprocess IR image (resize to 80x80, normalize)
    std::vector<uint8_t> ir_resized(input_size * input_size);
    ocfa::utils::ResizeImage(ir_image, width, height, 1,
                             ir_resized.data(), input_size, input_size);

    std::vector<float> ir_normalized(input_size * input_size);
    float ir_mean[1] = {0.5f};
    float ir_std[1] = {0.5f};
    ocfa::utils::NormalizeAndConvertToCHW(ir_resized.data(), ir_normalized.data(),
                                          input_size, input_size, 1,
                                          ir_mean, ir_std);

    // Prepare inputs for MiniFASNet (RGB + IR)
    ocfa::inference::Tensor rgb_tensor;
    rgb_tensor.shape = {1, 3, input_size, input_size};
    rgb_tensor.data = rgb_normalized;

    ocfa::inference::Tensor ir_tensor;
    ir_tensor.shape = {1, 1, input_size, input_size};
    ir_tensor.data = ir_normalized;

    std::vector<ocfa::inference::Tensor> inputs = {rgb_tensor, ir_tensor};
    std::vector<ocfa::inference::Tensor> outputs;

    // Run inference
    if (!g_liveness_engine->Run(inputs, outputs)) {
        return OCFA_ERROR_INFERENCE_FAILED;
    }

    if (outputs.empty()) {
        return OCFA_ERROR_LIVENESS_FAILED;
    }

    // Output is [1, 2] - [fake_score, real_score]
    // Apply softmax and get real_score
    float fake = outputs[0].data[0];
    float real = outputs[0].data[1];
    float exp_fake = std::exp(fake);
    float exp_real = std::exp(real);
    float sum = exp_fake + exp_real;

    *liveness_score = exp_real / sum;

    return OCFA_SUCCESS;
}

int ocfa_assess_quality(
    const uint8_t* rgb_image,
    int width,
    int height,
    float* quality_score
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (rgb_image == nullptr || quality_score == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    // Simple quality assessment based on image sharpness (Laplacian variance)
    // Higher variance = sharper image = better quality

    // Calculate Laplacian operator on grayscale
    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Convert RGB to grayscale (assuming BGR format)
        float b = rgb_image[i * 3 + 0];
        float g = rgb_image[i * 3 + 1];
        float r = rgb_image[i * 3 + 2];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }

    // Compute Laplacian
    std::vector<float> laplacian(width * height, 0.0f);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float val = -4.0f * gray[idx] +
                        gray[idx - 1] + gray[idx + 1] +
                        gray[idx - width] + gray[idx + width];
            laplacian[idx] = val;
        }
    }

    // Compute variance of Laplacian
    float mean = 0.0f;
    for (float val : laplacian) {
        mean += val;
    }
    mean /= laplacian.size();

    float variance = 0.0f;
    for (float val : laplacian) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= laplacian.size();

    // Normalize to [0, 1] range (empirically determined threshold)
    // Variance typically ranges from 0 to ~500 for 720p images
    *quality_score = std::min(1.0f, variance / 500.0f);

    return OCFA_SUCCESS;
}

int ocfa_extract_feature(
    const uint8_t* rgb_image,
    const uint8_t* ir_image,
    int width,
    int height,
    float* feature
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (rgb_image == nullptr || ir_image == nullptr || feature == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    // Preprocess RGB image for ArcFace (resize to 112x112, normalize)
    constexpr int input_size = 112;
    std::vector<uint8_t> rgb_resized(input_size * input_size * 3);
    ocfa::utils::ResizeImage(rgb_image, width, height, 3,
                             rgb_resized.data(), input_size, input_size);

    std::vector<float> rgb_normalized(input_size * input_size * 3);
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
    ocfa::utils::NormalizeAndConvertToCHW(rgb_resized.data(), rgb_normalized.data(),
                                          input_size, input_size, 3,
                                          mean, std);

    // Prepare input tensor
    ocfa::inference::Tensor input_tensor;
    input_tensor.shape = {1, 3, input_size, input_size};
    input_tensor.data = rgb_normalized;

    ocfa::inference::Tensor output_tensor;

    // Run inference
    if (!g_feature_engine->Run(input_tensor, output_tensor)) {
        return OCFA_ERROR_INFERENCE_FAILED;
    }

    if (output_tensor.data.size() != OCFA_FEATURE_DIM) {
        return OCFA_ERROR_FEATURE_FAILED;
    }

    // Copy feature and L2 normalize
    std::copy(output_tensor.data.begin(), output_tensor.data.end(), feature);

    // Use NEON-optimized L2 normalize if available
    if (ocfa::utils::neon::IsNeonAvailable()) {
        ocfa::utils::neon::L2NormalizeNeon(feature, OCFA_FEATURE_DIM);
    } else {
        ocfa::utils::L2Normalize(feature, OCFA_FEATURE_DIM);
    }

    return OCFA_SUCCESS;
}

float ocfa_compare_feature(const float* feature1, const float* feature2) {
    if (feature1 == nullptr || feature2 == nullptr) {
        return 0.0f;
    }

    // Use NEON-optimized cosine similarity if available
    if (ocfa::utils::neon::IsNeonAvailable()) {
        return ocfa::utils::neon::CosineSimilarityNeon(feature1, feature2, OCFA_FEATURE_DIM);
    } else {
        // Fallback: standard implementation
        float dot = 0.0f;
        for (int i = 0; i < OCFA_FEATURE_DIM; ++i) {
            dot += feature1[i] * feature2[i];
        }
        return std::max(0.0f, std::min(1.0f, dot));
    }
}

//========== Database Management ==========

int ocfa_add_user(const uint8_t user_id[OCFA_USER_ID_LEN], const float* feature) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (user_id == nullptr || feature == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    bool success = g_database->AddUser(user_id, feature);
    return success ? OCFA_SUCCESS : OCFA_ERROR_USER_ALREADY_EXISTS;
}

int ocfa_update_user(const uint8_t user_id[OCFA_USER_ID_LEN], const float* feature) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (user_id == nullptr || feature == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    bool success = g_database->UpdateUser(user_id, feature);
    return success ? OCFA_SUCCESS : OCFA_ERROR_USER_NOT_FOUND;
}

int ocfa_remove_user(const uint8_t user_id[OCFA_USER_ID_LEN]) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (user_id == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    bool success = g_database->RemoveUser(user_id);
    return success ? OCFA_SUCCESS : OCFA_ERROR_USER_NOT_FOUND;
}

int ocfa_search_user(
    const float* query_feature,
    uint8_t user_id[OCFA_USER_ID_LEN],
    float* similarity
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (query_feature == nullptr || user_id == nullptr || similarity == nullptr) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    bool success = g_database->SearchUser(query_feature, user_id, similarity);
    return success ? OCFA_SUCCESS : OCFA_ERROR_USER_NOT_FOUND;
}

int ocfa_search_users(
    const float* query_feature,
    float threshold,
    ocfa_search_result_t* results,
    int max_results
) {
    if (!g_sdk_initialized) {
        return OCFA_ERROR_NOT_INITIALIZED;
    }

    if (query_feature == nullptr || results == nullptr || max_results <= 0) {
        return OCFA_ERROR_INVALID_PARAM;
    }

    return g_database->SearchUsers(query_feature, threshold, results, max_results);
}

//========== Utility Functions ==========

const char* ocfa_get_version(void) {
    return OCFA_VERSION;
}
