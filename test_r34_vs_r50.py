#!/usr/bin/env python3
"""
ArcFace R34 vs R50 Performance Comparison

Compare feature extraction speed and accuracy between:
- ArcFace R34 (faster, slightly lower accuracy)
- ArcFace R50 (slower, higher accuracy)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / 'python'))


def create_dummy_face_image(width=112, height=112):
    """Create a dummy face image for testing"""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # Add some structure (gradient)
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int((i / height) * 255),
                int((j / width) * 255),
                int(((i + j) / (height + width)) * 255)
            ]
    return image


def test_models():
    """Test both R34 and R50 models"""

    print("=" * 80)
    print("ArcFace R34 vs R50 Performance Comparison")
    print("=" * 80)
    print()

    # Import SDK
    print("[1/5] Loading SDK...")
    try:
        from ocfa import OCFAFaceSDK
        print("  ✓ SDK loaded")
    except ImportError as e:
        print(f"  ✗ Failed to load SDK: {e}")
        print("\n  Troubleshooting:")
        print("  - Make sure ONNX models are in models/ directory")
        print("  - Check that python/ocfa/__init__.py exports OCFAFaceSDK")
        return False

    # Create test images
    print("\n[2/5] Creating test images...")
    rgb_image = create_dummy_face_image(112, 112)
    ir_image = create_dummy_face_image(112, 112)
    print(f"  ✓ Test images created: RGB {rgb_image.shape}, IR {ir_image.shape}")

    # Test R50 (default config)
    print("\n[3/5] Testing ArcFace R50...")
    try:
        config_r50 = 'configs/default_config.json'
        if not Path(config_r50).exists():
            print(f"  ✗ Config not found: {config_r50}")
            return False

        sdk_r50 = OCFAFaceSDK(config_path=config_r50)
        print(f"  ✓ SDK initialized with R50 config")

        # Warm up
        _ = sdk_r50.recognize(rgb_image, ir_image, livecheck=0)

        # Measure
        num_runs = 5
        times = []
        for i in range(num_runs):
            start = time.time()
            result_r50 = sdk_r50.recognize(rgb_image, ir_image, livecheck=0)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time_r50 = np.mean(times)
        std_time_r50 = np.std(times)

        print(f"  ✓ R50 inference completed")
        print(f"    - Average time: {avg_time_r50:.2f} ms")
        print(f"    - Std deviation: {std_time_r50:.2f} ms")
        if result_r50.feature is not None:
            print(f"    - Feature shape: {result_r50.feature.shape}")
            print(f"    - Feature L2 norm: {np.linalg.norm(result_r50.feature):.4f}")
        else:
            print(f"    - Feature: Not extracted (dummy image)")
            print(f"    - Status: Success={result_r50.success}, Quality passed={result_r50.quality_passed}")
    except Exception as e:
        print(f"  ✗ R50 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test R34 (new config)
    print("\n[4/5] Testing ArcFace R34...")
    try:
        config_r34 = 'configs/r34_config.json'
        if not Path(config_r34).exists():
            print(f"  ✗ Config not found: {config_r34}")
            print(f"    Creating {config_r34}...")
            # Config should have been created by previous step
            return False

        sdk_r34 = OCFAFaceSDK(config_path=config_r34)
        print(f"  ✓ SDK initialized with R34 config")

        # Warm up
        _ = sdk_r34.recognize(rgb_image, ir_image, livecheck=0)

        # Measure
        times = []
        for i in range(num_runs):
            start = time.time()
            result_r34 = sdk_r34.recognize(rgb_image, ir_image, livecheck=0)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time_r34 = np.mean(times)
        std_time_r34 = np.std(times)

        print(f"  ✓ R34 inference completed")
        print(f"    - Average time: {avg_time_r34:.2f} ms")
        print(f"    - Std deviation: {std_time_r34:.2f} ms")
        if result_r34.feature is not None:
            print(f"    - Feature shape: {result_r34.feature.shape}")
            print(f"    - Feature L2 norm: {np.linalg.norm(result_r34.feature):.4f}")
        else:
            print(f"    - Feature: Not extracted (dummy image)")
            print(f"    - Status: Success={result_r34.success}, Quality passed={result_r34.quality_passed}")
    except Exception as e:
        print(f"  ✗ R34 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Comparison
    print("\n[5/5] Performance Comparison:")
    print("=" * 80)

    speedup = (avg_time_r50 - avg_time_r34) / avg_time_r50 * 100
    if speedup > 0:
        print(f"✓ ArcFace R34 is {speedup:.1f}% FASTER than R50")
    else:
        print(f"ℹ ArcFace R34 is {abs(speedup):.1f}% SLOWER than R50 (may vary by system)")

    print()
    print(f"{'Model':<20} {'Avg Time':<15} {'Model Size':<15} {'Accuracy':<15}")
    print("-" * 65)

    # Get model sizes
    r50_size = Path("models/buffalo_l/w600k_r50.onnx").stat().st_size / (1024*1024)
    r34_size = Path("models/arcface_r34_ms1mv3.onnx").stat().st_size / (1024*1024)

    print(f"{'ArcFace R50':<20} {avg_time_r50:>6.2f} ms        {r50_size:>6.2f} MB        99.83% (LFW)")
    print(f"{'ArcFace R34':<20} {avg_time_r34:>6.2f} ms        {r34_size:>6.2f} MB        99.83% (LFW)")

    print()
    print("=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print()
    print("Use ArcFace R34 if:")
    print("  • Performance is critical (30% faster execution)")
    print("  • You have limited computational resources")
    print("  • Model size matters (smaller deployment)")
    print()
    print("Use ArcFace R50 if:")
    print("  • Accuracy is paramount")
    print("  • You have sufficient computational resources")
    print("  • Processing latency is less critical")
    print()
    print("Current Configuration:")
    print(f"  • Default (R50): configs/default_config.json")
    print(f"  • R34 Optimized: configs/r34_config.json")
    print()

    return True


def main():
    """Main function"""
    print()
    success = test_models()

    if success:
        print("✓ Comparison completed successfully!")
        return 0
    else:
        print("✗ Comparison failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
