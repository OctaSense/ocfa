/**
 * @file demo_basic.cpp
 * @brief Basic usage example of OCFA Face SDK (C++)
 */

#include "ocfa_face_sdk.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

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

int main(int argc, char** argv) {
    printf("OCFA Face SDK - Basic Example\n");
    printf("Version: %s\n\n", ocfa_get_version());

    // 1. Initialize SDK
    printf("Initializing SDK...\n");

    ocfa_config_t config;
    config.model_dir = "../../models";
    config.config_file = "../../configs/default_config.json";
    config.liveness_threshold = 0.90f;
    config.quality_threshold = 0.50f;
    config.num_threads = 2;

    int ret = ocfa_init(&config);
    if (ret != OCFA_SUCCESS) {
        printf("Failed to initialize SDK: %s\n", ocfa_get_error_string(ret));
        return 1;
    }

    printf("SDK initialized successfully\n\n");

    // 2. TODO: Load test images (placeholder)
    printf("Note: Image loading not implemented in this example\n");
    printf("In real usage, load RGB and IR images here\n\n");

    // For demonstration, we'll just test the database functions
    // with dummy features

    // 3. Test feature database
    printf("Testing feature database...\n");

    // Generate dummy feature
    float feature[OCFA_FEATURE_DIM];
    for (int i = 0; i < OCFA_FEATURE_DIM; ++i) {
        feature[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Normalize feature (simplified)
    float norm = 0.0f;
    for (int i = 0; i < OCFA_FEATURE_DIM; ++i) {
        norm += feature[i] * feature[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < OCFA_FEATURE_DIM; ++i) {
        feature[i] /= norm;
    }

    // Add user
    uint8_t user_id[OCFA_USER_ID_LEN];
    generate_user_id(user_id);

    printf("Adding user: ");
    print_user_id(user_id);
    printf("\n");

    ret = ocfa_add_user(user_id, feature);
    if (ret != OCFA_SUCCESS) {
        printf("Failed to add user: %s\n", ocfa_get_error_string(ret));
    } else {
        printf("User added successfully\n");
    }

    // Search for user
    printf("\nSearching for user...\n");

    uint8_t matched_id[OCFA_USER_ID_LEN];
    float similarity;

    ret = ocfa_search_user(feature, matched_id, &similarity);
    if (ret == OCFA_SUCCESS) {
        printf("Found user: ");
        print_user_id(matched_id);
        printf("\nSimilarity: %.3f\n", similarity);

        // Verify it's the same user
        if (memcmp(user_id, matched_id, OCFA_USER_ID_LEN) == 0) {
            printf("✓ Matched correct user!\n");
        }
    } else {
        printf("Search failed: %s\n", ocfa_get_error_string(ret));
    }

    // Test feature comparison
    printf("\nTesting feature comparison...\n");
    float self_sim = ocfa_compare_feature(feature, feature);
    printf("Self-similarity: %.3f (should be ~1.0)\n", self_sim);

    // 4. Cleanup
    printf("\nCleaning up...\n");
    ret = ocfa_release();
    if (ret != OCFA_SUCCESS) {
        printf("Failed to release SDK: %s\n", ocfa_get_error_string(ret));
        return 1;
    }

    printf("SDK released successfully\n");
    printf("\nExample completed!\n");

    return 0;
}
