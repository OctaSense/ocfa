#!/usr/bin/env python3
"""
OCFA Face SDK - Generate Calibration Data for Quantization

Creates synthetic face-like images for INT8 quantization calibration.
For production use, replace with real face images from your dataset.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import cv2


def generate_synthetic_face(size=112, seed=None):
    """
    Generate a synthetic face-like image

    Args:
        size: Image size (square)
        seed: Random seed for reproducibility

    Returns:
        BGR image array
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base image with skin-tone color
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Skin tone background (RGB: 200, 160, 130 -> BGR)
    skin_color = np.array([130, 160, 200], dtype=np.uint8)
    img[:, :] = skin_color

    # Add some texture/variation
    noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add darker regions for eyes (approximate positions)
    eye_y = int(size * 0.4)
    eye_left_x = int(size * 0.35)
    eye_right_x = int(size * 0.65)
    eye_radius = int(size * 0.08)

    cv2.circle(img, (eye_left_x, eye_y), eye_radius, (50, 50, 50), -1)
    cv2.circle(img, (eye_right_x, eye_y), eye_radius, (50, 50, 50), -1)

    # Add mouth region
    mouth_y = int(size * 0.75)
    mouth_center_x = int(size * 0.5)
    mouth_width = int(size * 0.2)
    mouth_height = int(size * 0.05)

    cv2.ellipse(img, (mouth_center_x, mouth_y), (mouth_width, mouth_height),
                0, 0, 180, (80, 60, 60), -1)

    # Add noise region for nose
    nose_y = int(size * 0.6)
    nose_x = int(size * 0.5)
    cv2.circle(img, (nose_x, nose_y), int(size * 0.05), (160, 140, 120), -1)

    # Apply slight gaussian blur for smoothness
    img = cv2.GaussianBlur(img, (5, 5), 1.0)

    return img


def generate_synthetic_ir(size=112, seed=None):
    """
    Generate a synthetic IR (infrared) image

    IR images typically have different characteristics than RGB

    Args:
        size: Image size (square)
        seed: Random seed

    Returns:
        Grayscale image array
    """
    if seed is not None:
        np.random.seed(seed + 10000)  # Different seed for IR

    # IR images tend to have less contrast
    img = np.ones((size, size), dtype=np.uint8) * 160

    # Add noise
    noise = np.random.randint(-15, 15, (size, size), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add darker regions for facial features
    eye_y = int(size * 0.4)
    eye_left_x = int(size * 0.35)
    eye_right_x = int(size * 0.65)
    eye_radius = int(size * 0.08)

    cv2.circle(img, (eye_left_x, eye_y), eye_radius, 100, -1)
    cv2.circle(img, (eye_right_x, eye_y), eye_radius, 100, -1)

    # Apply blur
    img = cv2.GaussianBlur(img, (7, 7), 1.5)

    return img


def main():
    parser = argparse.ArgumentParser(
        description='Generate calibration data for INT8 quantization'
    )
    parser.add_argument('--output', default='../../data/calibration',
                        help='Output directory for calibration images')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of calibration samples to generate')
    parser.add_argument('--size', type=int, default=112,
                        help='Image size (square)')
    parser.add_argument('--dual-stream', action='store_true',
                        help='Generate dual-stream RGB-IR pairs for MiniFASNet')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("OCFA Face SDK - Calibration Data Generator")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Dual-stream (RGB+IR): {args.dual_stream}")
    print()

    # Generate samples
    for i in range(args.num_samples):
        seed = i

        if args.dual_stream:
            # Generate RGB-IR pairs
            rgb_img = generate_synthetic_face(args.size, seed)
            ir_img = generate_synthetic_ir(args.size, seed)

            rgb_path = output_dir / f"face_{i:04d}_rgb.jpg"
            ir_path = output_dir / f"face_{i:04d}_ir.jpg"

            cv2.imwrite(str(rgb_path), rgb_img)
            cv2.imwrite(str(ir_path), ir_img)

            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{args.num_samples} RGB-IR pairs...")
        else:
            # Generate RGB only
            rgb_img = generate_synthetic_face(args.size, seed)

            img_path = output_dir / f"face_{i:04d}.jpg"
            cv2.imwrite(str(img_path), rgb_img)

            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{args.num_samples} images...")

    print(f"\n✓ Generated {args.num_samples} calibration samples")
    print(f"  Location: {output_dir.absolute()}")

    if args.dual_stream:
        print(f"\n  RGB images: face_XXXX_rgb.jpg")
        print(f"  IR images: face_XXXX_ir.jpg")
    else:
        print(f"\n  Images: face_XXXX.jpg")

    print(f"\n{'='*60}")
    print("Calibration data generation completed!")
    print(f"{'='*60}")

    print(f"\nNext steps:")
    print(f"  1. (Optional) Replace synthetic images with real face images")
    print(f"  2. Run quantization: python quantization.py \\")
    print(f"       --model <model.onnx> \\")
    print(f"       --output <model_int8.onnx> \\")
    print(f"       --calib-data {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
