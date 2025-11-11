"""
OCFA Face SDK - INT8 Quantization Tool

Quantize ONNX models to INT8 using Post-Training Quantization (PTQ).
Uses calibration dataset for optimal quantization parameters.
"""

import argparse
import os
import sys
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
import cv2
from pathlib import Path
from typing import List
import glob


class FaceCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for face recognition models

    Reads calibration images and feeds them to the quantization process.
    """

    def __init__(self,
                 calibration_data_dir: str,
                 input_name: str = 'input',
                 batch_size: int = 1,
                 input_shape: tuple = (1, 3, 112, 112)):
        """
        Initialize calibration data reader

        Args:
            calibration_data_dir: Directory containing calibration images
            input_name: Model input name
            batch_size: Batch size
            input_shape: Input tensor shape
        """
        self.calibration_data_dir = calibration_data_dir
        self.input_name = input_name
        self.batch_size = batch_size
        self.input_shape = input_shape

        # Load image paths
        self.image_paths = self._load_image_paths()
        self.current_index = 0

        print(f"Loaded {len(self.image_paths)} calibration images")

    def _load_image_paths(self) -> List[str]:
        """Load all image paths from calibration directory"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []

        for ext in image_extensions:
            pattern = os.path.join(self.calibration_data_dir, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))

        return sorted(image_paths)

    def get_next(self) -> dict:
        """
        Get next calibration batch

        Returns:
            Dictionary with input name and data
        """
        if self.current_index >= len(self.image_paths):
            return None

        # Load and preprocess image
        img_path = self.image_paths[self.current_index]
        image = self._load_and_preprocess(img_path)

        self.current_index += 1

        return {self.input_name: image}

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target size
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        image = cv2.resize(image, (target_w, target_h))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image


class DualStreamCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for dual-stream models (RGB + IR)

    For MiniFASNet liveness detection model.
    """

    def __init__(self,
                 calibration_data_dir: str,
                 rgb_input_name: str = 'rgb_input',
                 ir_input_name: str = 'ir_input',
                 batch_size: int = 1):
        """
        Initialize dual-stream calibration data reader

        Args:
            calibration_data_dir: Directory containing RGB and IR image pairs
            rgb_input_name: RGB input name
            ir_input_name: IR input name
            batch_size: Batch size
        """
        self.calibration_data_dir = calibration_data_dir
        self.rgb_input_name = rgb_input_name
        self.ir_input_name = ir_input_name
        self.batch_size = batch_size

        # Load image pairs
        self.image_pairs = self._load_image_pairs()
        self.current_index = 0

        print(f"Loaded {len(self.image_pairs)} RGB-IR image pairs")

    def _load_image_pairs(self) -> List[tuple]:
        """Load RGB-IR image pairs"""
        rgb_pattern = os.path.join(self.calibration_data_dir, '*_rgb.jpg')
        rgb_paths = glob.glob(rgb_pattern)

        pairs = []
        for rgb_path in rgb_paths:
            # Find corresponding IR image
            ir_path = rgb_path.replace('_rgb.jpg', '_ir.jpg')
            if os.path.exists(ir_path):
                pairs.append((rgb_path, ir_path))

        return sorted(pairs)

    def get_next(self) -> dict:
        """Get next calibration batch"""
        if self.current_index >= len(self.image_pairs):
            return None

        rgb_path, ir_path = self.image_pairs[self.current_index]

        # Load and preprocess RGB
        rgb_image = self._load_rgb(rgb_path)

        # Load and preprocess IR
        ir_image = self._load_ir(ir_path)

        self.current_index += 1

        return {
            self.rgb_input_name: rgb_image,
            self.ir_input_name: ir_image
        }

    def _load_rgb(self, image_path: str) -> np.ndarray:
        """Load and preprocess RGB image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def _load_ir(self, image_path: str) -> np.ndarray:
        """Load and preprocess IR image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add channel dim
        image = np.expand_dims(image, axis=0)  # Add batch dim
        return image


def quantize_model_dynamic(input_model_path: str, output_model_path: str):
    """
    Apply dynamic quantization (simpler, no calibration needed)

    Args:
        input_model_path: Input ONNX model path
        output_model_path: Output quantized model path
    """
    print("\nApplying dynamic INT8 quantization...")
    print(f"Input: {input_model_path}")
    print(f"Output: {output_model_path}")

    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QInt8
    )

    print("✓ Dynamic quantization completed")


def quantize_model_static(input_model_path: str,
                          output_model_path: str,
                          calibration_data_reader: CalibrationDataReader):
    """
    Apply static quantization (PTQ with calibration)

    Args:
        input_model_path: Input ONNX model path
        output_model_path: Output quantized model path
        calibration_data_reader: Calibration data reader
    """
    print("\nApplying static INT8 quantization (PTQ)...")
    print(f"Input: {input_model_path}")
    print(f"Output: {output_model_path}")

    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantType.QInt8,
        per_channel=False
    )

    print("✓ Static quantization completed")


def compare_model_sizes(original_path: str, quantized_path: str):
    """
    Compare model sizes before and after quantization

    Args:
        original_path: Original model path
        quantized_path: Quantized model path
    """
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    compression_ratio = original_size / quantized_size

    print(f"\n{'='*60}")
    print("Model Size Comparison:")
    print(f"{'='*60}")
    print(f"Original (FP32):  {original_size:.2f} MB")
    print(f"Quantized (INT8): {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")


def main():
    """Main quantization function"""
    parser = argparse.ArgumentParser(description='Quantize ONNX models to INT8')
    parser.add_argument('--model', type=str, required=True,
                       help='Input ONNX model path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output quantized model path')
    parser.add_argument('--calib-data', type=str, required=True,
                       help='Calibration data directory')
    parser.add_argument('--method', type=str, default='static',
                       choices=['static', 'dynamic'],
                       help='Quantization method')
    parser.add_argument('--model-type', type=str, default='arcface',
                       choices=['arcface', 'minifasnet'],
                       help='Model type')

    args = parser.parse_args()

    # Check input model exists
    if not os.path.exists(args.model):
        print(f"✗ Error: Input model not found: {args.model}")
        sys.exit(1)

    # Check calibration data directory
    if args.method == 'static' and not os.path.exists(args.calib_data):
        print(f"✗ Error: Calibration data directory not found: {args.calib_data}")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    try:
        if args.method == 'dynamic':
            # Dynamic quantization
            quantize_model_dynamic(args.model, args.output)

        else:
            # Static quantization with calibration
            print(f"\nPreparing calibration data from {args.calib_data}...")

            if args.model_type == 'arcface':
                # Single-stream model
                # Note: Input name may vary (e.g., 'input', 'input.1', 'data')
                # Check your model's input name first
                calibration_reader = FaceCalibrationDataReader(
                    args.calib_data,
                    input_name='input.1',  # For buffalo_l w600k_r50 model
                    input_shape=(1, 3, 112, 112)
                )
            else:
                # Dual-stream model (MiniFASNet)
                calibration_reader = DualStreamCalibrationDataReader(
                    args.calib_data,
                    rgb_input_name='rgb_input',
                    ir_input_name='ir_input'
                )

            quantize_model_static(args.model, args.output, calibration_reader)

        # Compare sizes
        compare_model_sizes(args.model, args.output)

        print(f"\n{'='*60}")
        print("Quantization completed successfully!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
