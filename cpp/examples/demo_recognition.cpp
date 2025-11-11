/**
 * @file demo_recognition.cpp
 * @brief Complete face recognition example with RGB-IR images
 */

#include "ocfa_face_sdk.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// Helper function to print user ID
void print_user_id(const uint8_t user_id[OCFA_USER_ID_LEN]) {
    for (int i = 0; i < OCFA_USER_ID_LEN; ++i) {
        printf("%02x", user_id[i]);
    }
}

// Helper function to generate random user ID
void generate_user_id(uint8_t user_id[OCFA_USER_ID_LEN]) {
    for (int i = 0; i < OCFA_USER_ID_LEN; ++i) {
        user_id[i] = static_cast<uint8_t>(rand() % 256);
    }
}

// Create dummy 720p RGB image (BGR format)
std::vector<uint8_t> create_dummy_rgb_image(int width, int height) {
    std::vector<uint8_t> image(width * height * 3);

    // Generate gradient pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            image[idx + 0] = static_cast<uint8_t>((x * 255) / width);       // B
            image[idx + 1] = static_cast<uint8_t>((y * 255) / height);      // G
            image[idx + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height)); // R
        }
    }

    return image;
}

// Create dummy 720p IR image (grayscale)
std::vector<uint8_t> create_dummy_ir_image(int width, int height) {
    std::vector<uint8_t> image(width * height);

    // Generate checkerboard pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            image[idx] = ((x / 32) + (y / 32)) % 2 ? 200 : 50;
        }
    }

    return image;
}

int main(int argc, char** argv) {
    printf("OCFA Face SDK - Recognition Example\\n");
    printf("Version: %s\\n\\n", ocfa_get_version());

    // 1. Initialize SDK
    printf("Initializing SDK...\\n");

    ocfa_config_t config;
    config.model_dir = "../../models";
    config.config_file = "../../configs/default_config.json";
    config.liveness_threshold = 0.90f;
    config.quality_threshold = 0.50f;
    config.num_threads = 2;

    int ret = ocfa_init(&config);
    if (ret != OCFA_SUCCESS) {
        printf("Failed to initialize SDK: %s\\n", ocfa_get_error_string(ret));
        printf("Note: Make sure ONNX models exist in %s/\\n", config.model_dir);
        printf("      Expected files: minifasnet_int8.onnx, arcface_r34_int8.onnx\\n");
        return 1;
    }

    printf("SDK initialized successfully\\n\\n");

    // 2. Create test images (in real usage, load from camera or file)
    printf("Creating test images (1280x720)...\\n");
    const int width = 1280;
    const int height = 720;

    auto rgb_image = create_dummy_rgb_image(width, height);
    auto ir_image = create_dummy_ir_image(width, height);
    printf("Test images created\\n\\n");

    // 3. Perform recognition with livecheck=1 (enable liveness detection)
    printf("Performing face recognition (livecheck=1, with liveness detection)...\\n");

    ocfa_recognition_result_t result;
    int livecheck = 1;  // 1 = enable RGB+IR liveness detection
    ret = ocfa_recognize(rgb_image.data(), ir_image.data(), width, height, livecheck, &result);

    if (ret == OCFA_SUCCESS) {
        printf("Recognition successful!\\n");
        printf("  Liveness: %.3f (threshold: %.2f) - %s\\n",
               result.liveness_score,
               config.liveness_threshold,
               result.liveness_passed ? "PASSED" : "FAILED");
        printf("  Quality:  %.3f (threshold: %.2f) - %s\\n",
               result.quality_score,
               config.quality_threshold,
               result.quality_passed ? "PASSED" : "FAILED");
        printf("  Feature:  %s\\n",
               result.feature_extracted ? "EXTRACTED" : "FAILED");
        printf("  Total time: %u ms\\n", result.total_time_ms);

        if (result.feature_extracted) {
            printf("  Feature vector (first 10 dims): ");
            for (int i = 0; i < 10; ++i) {
                printf("%.4f ", result.feature[i]);
            }
            printf("...\\n");
        }
    } else {
        printf("Recognition failed: %s\\n", ocfa_get_error_string(ret));
        printf("Error message: %s\\n", result.error_msg);
    }

    // 3b. Test bypass mode (livecheck=0)
    printf("\\nPerforming face recognition (livecheck=0, bypass liveness detection)...\\n");

    ocfa_recognition_result_t result_bypass;
    livecheck = 0;  // 0 = bypass liveness detection
    ret = ocfa_recognize(rgb_image.data(), ir_image.data(), width, height, livecheck, &result_bypass);

    if (ret == OCFA_SUCCESS) {
        printf("Recognition successful (bypass mode)!\\n");
        printf("  Liveness: %.3f (bypassed)\\n", result_bypass.liveness_score);
        printf("  Quality:  %.3f (threshold: %.2f) - %s\\n",
               result_bypass.quality_score,
               config.quality_threshold,
               result_bypass.quality_passed ? "PASSED" : "FAILED");
        printf("  Feature:  %s\\n",
               result_bypass.feature_extracted ? "EXTRACTED" : "FAILED");
        printf("  Total time: %u ms\\n", result_bypass.total_time_ms);

        if (result.total_time_ms > 0 && result_bypass.total_time_ms > 0) {
            float speedup = (float)(result.total_time_ms - result_bypass.total_time_ms) / result.total_time_ms * 100.0f;
            if (speedup > 0) {
                printf("  Performance gain: %.1f%% faster (bypassed liveness detection)\\n", speedup);
            }
        }
    } else {
        printf("Recognition failed (bypass mode): %s\\n", ocfa_get_error_string(ret));
        printf("Error message: %s\\n", result_bypass.error_msg);
    }

    // 4. Test database operations (if feature extracted from bypass mode)
    if (ret == OCFA_SUCCESS && result_bypass.feature_extracted) {
        printf("\\nTesting database operations...\\n");

        // Add user to database
        uint8_t user_id[OCFA_USER_ID_LEN];
        generate_user_id(user_id);

        printf("Adding user: ");
        print_user_id(user_id);
        printf("\\n");

        ret = ocfa_add_user(user_id, result_bypass.feature);
        if (ret != OCFA_SUCCESS) {
            printf("Failed to add user: %s\\n", ocfa_get_error_string(ret));
        } else {
            printf("User added successfully\\n");

            // Search for the same user
            printf("\\nSearching for user...\\n");

            uint8_t matched_id[OCFA_USER_ID_LEN];
            float similarity;

            ret = ocfa_search_user(result_bypass.feature, matched_id, &similarity);
            if (ret == OCFA_SUCCESS) {
                printf("Found user: ");
                print_user_id(matched_id);
                printf("\\nSimilarity: %.3f\\n", similarity);

                // Verify it's the same user
                if (memcmp(user_id, matched_id, OCFA_USER_ID_LEN) == 0) {
                    printf("✓ Matched correct user!\\n");
                } else {
                    printf("✗ Matched different user\\n");
                }
            } else {
                printf("Search failed: %s\\n", ocfa_get_error_string(ret));
            }

            // Test 1:N search with threshold
            printf("\\nTesting 1:N search (threshold=0.5)...\\n");
            ocfa_search_result_t results[10];
            int count = ocfa_search_users(result_bypass.feature, 0.5f, results, 10);

            if (count > 0) {
                printf("Found %d matches:\\n", count);
                for (int i = 0; i < count; ++i) {
                    printf("  %d. ", i + 1);
                    print_user_id(results[i].user_id);
                    printf(" (similarity: %.3f)\\n", results[i].similarity);
                }
            } else if (count == 0) {
                printf("No matches found\\n");
            } else {
                printf("Search failed: %s\\n", ocfa_get_error_string(count));
            }
        }
    }

    // 5. Test feature comparison
    printf("\\nTesting feature comparison...\\n");
    if (result_bypass.feature_extracted) {
        float self_sim = ocfa_compare_feature(result_bypass.feature, result_bypass.feature);
        printf("Self-similarity: %.3f (should be ~1.0)\\n", self_sim);
    }

    // 6. Cleanup
    printf("\\nCleaning up...\\n");
    ret = ocfa_release();
    if (ret != OCFA_SUCCESS) {
        printf("Failed to release SDK: %s\\n", ocfa_get_error_string(ret));
        return 1;
    }

    printf("SDK released successfully\\n");
    printf("\\nExample completed!\\n");

    return 0;
}
