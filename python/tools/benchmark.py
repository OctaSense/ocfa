#!/usr/bin/env python3
"""
OCFA Face SDK - Performance Benchmark Tool

Benchmark the SDK performance on test datasets.
"""

import argparse
import time
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocfa import OCFAFaceSDK


def benchmark_liveness(sdk, rgb_images, ir_images):
    """Benchmark liveness detection"""
    print("\n=== Liveness Detection Benchmark ===")

    times = []
    for rgb, ir in zip(rgb_images, ir_images):
        start = time.perf_counter()
        score = sdk.liveness_detector.detect(rgb, ir)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    print(f"Samples: {len(times)}")
    print(f"Average: {np.mean(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Std Dev: {np.std(times):.2f} ms")

    return {
        'average': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }


def benchmark_quality(sdk, rgb_images):
    """Benchmark quality assessment"""
    print("\n=== Quality Assessment Benchmark ===")

    times = []
    for rgb in rgb_images:
        start = time.perf_counter()
        score = sdk.quality_assessor.assess(rgb)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    print(f"Samples: {len(times)}")
    print(f"Average: {np.mean(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Std Dev: {np.std(times):.2f} ms")

    return {
        'average': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }


def benchmark_feature_extraction(sdk, rgb_images, ir_images):
    """Benchmark feature extraction"""
    print("\n=== Feature Extraction Benchmark ===")

    times = []
    for rgb, ir in zip(rgb_images, ir_images):
        start = time.perf_counter()
        feature = sdk.feature_extractor.extract(rgb, ir)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    print(f"Samples: {len(times)}")
    print(f"Average: {np.mean(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Std Dev: {np.std(times):.2f} ms")

    return {
        'average': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }


def benchmark_recognition(sdk, rgb_images, ir_images):
    """Benchmark full recognition pipeline"""
    print("\n=== Full Recognition Pipeline Benchmark ===")

    times = []
    success_count = 0

    for rgb, ir in zip(rgb_images, ir_images):
        start = time.perf_counter()
        result = sdk.recognize(rgb, ir)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

        if result.success:
            success_count += 1

    print(f"Samples: {len(times)}")
    print(f"Success Rate: {success_count}/{len(times)} ({100*success_count/len(times):.1f}%)")
    print(f"Average: {np.mean(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Std Dev: {np.std(times):.2f} ms")
    print(f"Throughput: {1000/np.mean(times):.2f} fps")

    return {
        'average': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times),
        'throughput': 1000 / np.mean(times),
        'success_rate': success_count / len(times)
    }


def load_test_images(data_dir, num_samples=100):
    """Load test images from directory"""
    data_path = Path(data_dir)

    print(f"\nLoading test images from {data_path}...")

    # Create dummy test images for now
    # In real usage, load from actual image files
    rgb_images = []
    ir_images = []

    for i in range(num_samples):
        # 720p dummy images
        rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        ir = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
        rgb_images.append(rgb)
        ir_images.append(ir)

    print(f"Loaded {len(rgb_images)} RGB-IR image pairs")

    return rgb_images, ir_images


def main():
    parser = argparse.ArgumentParser(description='OCFA Face SDK Performance Benchmark')
    parser.add_argument('--config', default='../../configs/default_config.json',
                        help='Path to config file')
    parser.add_argument('--data', default='../../data/test',
                        help='Path to test data directory')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to benchmark')
    parser.add_argument('--output', default='benchmark_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--module', choices=['liveness', 'quality', 'feature', 'recognition', 'all'],
                        default='all', help='Which module to benchmark')

    args = parser.parse_args()

    print("=" * 60)
    print("OCFA Face SDK - Performance Benchmark")
    print("=" * 60)

    # Initialize SDK
    print(f"\nInitializing SDK with config: {args.config}")
    try:
        sdk = OCFAFaceSDK(config_path=args.config)
        print("SDK initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SDK: {e}")
        return 1

    # Load test data
    rgb_images, ir_images = load_test_images(args.data, args.samples)

    # Run benchmarks
    results = {}

    if args.module in ['liveness', 'all']:
        results['liveness'] = benchmark_liveness(sdk, rgb_images, ir_images)

    if args.module in ['quality', 'all']:
        results['quality'] = benchmark_quality(sdk, rgb_images)

    if args.module in ['feature', 'all']:
        results['feature_extraction'] = benchmark_feature_extraction(sdk, rgb_images, ir_images)

    if args.module in ['recognition', 'all']:
        results['full_pipeline'] = benchmark_recognition(sdk, rgb_images, ir_images)

    # Save results
    print(f"\n\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
