"""
OCFA Face SDK - Basic Usage Example

This example demonstrates basic face recognition workflow.
"""

import cv2
import numpy as np
from ocfa.sdk import OCFAFaceSDK
from ocfa.utils import generate_user_id, bytes_to_user_id


def main():
    """Basic usage example"""

    # 1. Initialize SDK
    print("Initializing SDK...")
    sdk = OCFAFaceSDK(config_path='../../configs/default_config.json')
    print(f"SDK initialized: {sdk.get_stats()}")

    # 2. Load test images
    print("\nLoading test images...")
    rgb_image = cv2.imread('test_rgb.jpg')
    ir_image = cv2.imread('test_ir.jpg', cv2.IMREAD_GRAYSCALE)

    if rgb_image is None or ir_image is None:
        print("Error: Could not load test images")
        print("Please place test_rgb.jpg and test_ir.jpg in this directory")
        return

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # 3. Face recognition (extract feature)
    print("\nPerforming face recognition...")
    result = sdk.recognize(rgb_image, ir_image)

    print(f"Liveness: {'PASSED' if result.liveness_passed else 'FAILED'} "
          f"(score: {result.liveness_score:.3f})")
    print(f"Quality: {'PASSED' if result.quality_passed else 'FAILED'} "
          f"(score: {result.quality_score:.3f})")
    print(f"Feature extracted: {result.feature_extracted}")
    print(f"Processing time: {result.total_time_ms:.1f} ms")

    if not result.feature_extracted:
        print(f"Error: {result.error_msg}")
        return

    # 4. Register user
    user_id = generate_user_id()
    user_id_str = bytes_to_user_id(user_id)
    print(f"\nRegistering user: {user_id_str}")

    success = sdk.add_user(user_id, result.feature)
    if success:
        print("User registered successfully")
    else:
        print("Failed to register user")

    print(f"Total users in database: {sdk.get_user_count()}")

    # 5. Search for user (verification)
    print("\nSearching for user...")
    matched_id, similarity = sdk.search_user(result.feature)

    if matched_id:
        matched_id_str = bytes_to_user_id(matched_id)
        print(f"Found user: {matched_id_str}")
        print(f"Similarity: {similarity:.3f}")

        if similarity >= 0.70:
            print("✓ Verification PASSED")
        else:
            print("✗ Verification FAILED (similarity too low)")
    else:
        print("No user found in database")

    # 6. Search multiple users (identification)
    print("\nSearching top 3 similar users...")
    matches = sdk.search_users(result.feature, threshold=0.60, max_results=3)

    if matches:
        print(f"Found {len(matches)} matches:")
        for i, (uid, sim) in enumerate(matches, 1):
            uid_str = bytes_to_user_id(uid)
            print(f"  {i}. User: {uid_str}, Similarity: {sim:.3f}")
    else:
        print("No matches found")

    # 7. Compare two features directly
    print("\nDirect feature comparison...")
    # Compare the same feature to itself (should be ~1.0)
    self_similarity = sdk.compare_features(result.feature, result.feature)
    print(f"Self-similarity: {self_similarity:.3f}")

    # 8. SDK statistics
    print("\nSDK Statistics:")
    stats = sdk.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
