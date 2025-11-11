#!/usr/bin/env python3
"""
OCFA Face SDK - Validate Quantized Models

Tests quantized models to ensure they produce valid outputs.
"""

import argparse
import sys
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path


def load_test_image(image_path, size=(112, 112)):
    """Load and preprocess test image"""
    img = cv2.imread(image_path)
    if img is None:
        # Create synthetic test image
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def test_model(model_path, input_name='input.1', num_tests=10):
    """
    Test a model with random/synthetic inputs

    Args:
        model_path: Path to ONNX model
        input_name: Input tensor name
        num_tests: Number of test runs

    Returns:
        dict with test results
    """
    print(f"\nTesting model: {model_path}")
    print(f"  Input name: {input_name}")

    # Load model
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return {'success': False, 'error': str(e)}

    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"  Input shape: {input_info.shape}")
    print(f"  Output shape: {output_info.shape}")
    print(f"  Output name: {output_info.name}")

    # Test with multiple inputs
    outputs = []
    errors = []

    for i in range(num_tests):
        try:
            # Create random input
            test_input = np.random.rand(1, 3, 112, 112).astype(np.float32)

            # Run inference
            result = session.run([output_info.name], {input_name: test_input})
            output = result[0]

            outputs.append(output)

        except Exception as e:
            errors.append(str(e))

    if errors:
        print(f"  ✗ Errors occurred: {len(errors)}/{num_tests}")
        print(f"    First error: {errors[0]}")
        return {'success': False, 'error_count': len(errors), 'errors': errors[:3]}

    # Analyze outputs
    outputs = np.array(outputs)

    print(f"  ✓ All {num_tests} tests passed")
    print(f"  Output statistics:")
    print(f"    Shape: {outputs[0].shape}")
    print(f"    Mean: {np.mean(outputs):.6f}")
    print(f"    Std:  {np.std(outputs):.6f}")
    print(f"    Min:  {np.min(outputs):.6f}")
    print(f"    Max:  {np.max(outputs):.6f}")

    # Check L2 norm for feature vectors
    if outputs.shape[-1] == 512:  # Face recognition features
        norms = np.linalg.norm(outputs, axis=-1)
        print(f"    L2 norms: mean={np.mean(norms):.6f}, std={np.std(norms):.6f}")

        # Check if normalized
        if np.allclose(norms, 1.0, atol=0.01):
            print(f"    ✓ Features are L2-normalized")
        else:
            print(f"    ⚠ Features are NOT L2-normalized (may need post-processing)")

    return {
        'success': True,
        'num_tests': num_tests,
        'output_shape': outputs[0].shape,
        'mean': float(np.mean(outputs)),
        'std': float(np.std(outputs)),
        'min': float(np.min(outputs)),
        'max': float(np.max(outputs))
    }


def compare_models(fp32_model, int8_model, input_name='input.1', num_tests=10):
    """
    Compare FP32 and INT8 models

    Args:
        fp32_model: Path to FP32 model
        int8_model: Path to INT8 model
        input_name: Input tensor name
        num_tests: Number of test runs
    """
    print(f"\n{'='*60}")
    print("Comparing FP32 vs INT8 Models")
    print(f"{'='*60}")

    # Load models
    session_fp32 = ort.InferenceSession(fp32_model)
    session_int8 = ort.InferenceSession(int8_model)

    output_name_fp32 = session_fp32.get_outputs()[0].name
    output_name_int8 = session_int8.get_outputs()[0].name

    # Compare outputs
    diffs = []
    cosine_sims = []

    for i in range(num_tests):
        # Create random input
        test_input = np.random.rand(1, 3, 112, 112).astype(np.float32)

        # Run both models
        output_fp32 = session_fp32.run([output_name_fp32], {input_name: test_input})[0]
        output_int8 = session_int8.run([output_name_int8], {input_name: test_input})[0]

        # Compute difference
        diff = np.abs(output_fp32 - output_int8)
        diffs.append(diff)

        # Compute cosine similarity for feature vectors
        if output_fp32.shape[-1] == 512:
            # Normalize
            fp32_norm = output_fp32 / (np.linalg.norm(output_fp32) + 1e-8)
            int8_norm = output_int8 / (np.linalg.norm(output_int8) + 1e-8)

            # Cosine similarity
            cos_sim = np.dot(fp32_norm.flatten(), int8_norm.flatten())
            cosine_sims.append(cos_sim)

    diffs = np.array(diffs)

    print(f"\nAbsolute difference (FP32 - INT8):")
    print(f"  Mean: {np.mean(diffs):.6f}")
    print(f"  Max:  {np.max(diffs):.6f}")
    print(f"  Std:  {np.std(diffs):.6f}")

    if cosine_sims:
        cosine_sims = np.array(cosine_sims)
        print(f"\nCosine similarity:")
        print(f"  Mean: {np.mean(cosine_sims):.6f}")
        print(f"  Min:  {np.min(cosine_sims):.6f}")
        print(f"  Max:  {np.max(cosine_sims):.6f}")

        if np.mean(cosine_sims) > 0.99:
            print(f"  ✓ Excellent similarity (>0.99)")
        elif np.mean(cosine_sims) > 0.95:
            print(f"  ✓ Good similarity (>0.95)")
        else:
            print(f"  ⚠ Low similarity (<0.95) - quantization may have degraded accuracy")


def main():
    parser = argparse.ArgumentParser(description='Validate quantized ONNX models')
    parser.add_argument('--fp32-model', default='models/buffalo_l/w600k_r50.onnx',
                        help='Path to FP32 model')
    parser.add_argument('--int8-model', default='models/w600k_r50_int8_static.onnx',
                        help='Path to INT8 model')
    parser.add_argument('--input-name', default='input.1',
                        help='Input tensor name')
    parser.add_argument('--num-tests', type=int, default=10,
                        help='Number of test runs')
    parser.add_argument('--compare', action='store_true',
                        help='Compare FP32 and INT8 models')

    args = parser.parse_args()

    print(f"{'='*60}")
    print("OCFA Face SDK - Model Validation")
    print(f"{'='*60}")

    # Test FP32 model
    fp32_results = test_model(args.fp32_model, args.input_name, args.num_tests)

    # Test INT8 model
    int8_results = test_model(args.int8_model, args.input_name, args.num_tests)

    # Compare if requested
    if args.compare and fp32_results['success'] and int8_results['success']:
        compare_models(args.fp32_model, args.int8_model, args.input_name, args.num_tests)

    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"FP32 Model: {'✓ PASS' if fp32_results['success'] else '✗ FAIL'}")
    print(f"INT8 Model: {'✓ PASS' if int8_results['success'] else '✗ FAIL'}")

    if fp32_results['success'] and int8_results['success']:
        print(f"\n✓ All models validated successfully!")
        return 0
    else:
        print(f"\n✗ Some models failed validation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
