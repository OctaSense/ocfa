#!/usr/bin/env python3
"""
OCFA Face SDK - MiniFASNet PyTorch to ONNX Converter

Convert MiniFASNet PyTorch models to ONNX format.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import onnx
from onnx import shape_inference

# Import MiniFASNet architecture
from ocfa.models.minifasnet_arch import MiniFASNetV2, MiniFASNetV1SE


def load_minifasnet_model(model_path, model_type='v2', input_size=80):
    """
    Load MiniFASNet model from PyTorch checkpoint

    Args:
        model_path: Path to .pth file
        model_type: 'v2' or 'v1se'
        input_size: Input image size (default 80x80)

    Returns:
        Loaded PyTorch model
    """
    print(f"\n{'='*60}")
    print(f"Loading MiniFASNet {model_type.upper()} model")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Input size: {input_size}x{input_size}")

    # Load checkpoint first to determine kernel size
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Detect kernel size from conv_6_dw layer
    kernel_size = (7, 7)  # default
    for key in state_dict.keys():
        if 'conv_6_dw' in key and 'weight' in key:
            weight_shape = state_dict[key].shape
            if len(weight_shape) >= 4:
                kernel_size = (weight_shape[2], weight_shape[3])
                print(f"Detected kernel size from checkpoint: {kernel_size}")
                break

    # Create model based on type
    if model_type.lower() == 'v2':
        model = MiniFASNetV2(
            embedding_size=128,
            conv6_kernel=kernel_size,
            num_classes=3,  # 3 classes: real, fake, mask
            img_channel=3
        )
    elif model_type.lower() == 'v1se':
        model = MiniFASNetV1SE(
            embedding_size=128,
            conv6_kernel=kernel_size,
            num_classes=3,
            img_channel=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights (state_dict already loaded earlier for kernel size detection)
    print(f"\nLoading weights...")

    # Remove 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    print("✓ Weights loaded successfully")

    model.eval()
    return model


def export_to_onnx(model, output_path, input_size=80, opset_version=11, simplify=True):
    """
    Export MiniFASNet model to ONNX

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_size: Input image size
        opset_version: ONNX opset version
        simplify: Whether to simplify ONNX model
    """
    print(f"\n{'='*60}")
    print("Exporting to ONNX")
    print(f"{'='*60}")
    print(f"Output path: {output_path}")
    print(f"Input shape: [1, 3, {input_size}, {input_size}]")
    print(f"Opset version: {opset_version}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    print("\nExporting...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ Model exported to {output_path}")

    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)

    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX validation failed: {e}")
        return False

    # Infer shapes
    try:
        onnx_model = shape_inference.infer_shapes(onnx_model)
        print("✓ Shape inference successful")
    except Exception as e:
        print(f"⚠ Shape inference warning: {e}")

    # Print model info
    print(f"\nModel Info:")
    print(f"  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                 for dim in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")

    print(f"  Outputs:")
    for out in onnx_model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                 for dim in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, output_path)
                new_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✓ Model simplified: {file_size:.2f} MB → {new_size:.2f} MB")
            else:
                print("⚠ Simplification failed, using original model")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")

    return True


def test_onnx_inference(onnx_path, input_size=80):
    """
    Test ONNX model inference

    Args:
        onnx_path: Path to ONNX model
        input_size: Input image size
    """
    print(f"\n{'='*60}")
    print("Testing ONNX Inference")
    print(f"{'='*60}")

    try:
        import onnxruntime as ort
        import numpy as np

        # Create session
        session = ort.InferenceSession(onnx_path)

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")

        # Create random input
        test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Run inference
        print("\nRunning inference...")
        outputs = session.run([output_name], {input_name: test_input})

        print(f"✓ Inference successful")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output (first sample): {outputs[0][0]}")

        # Check if output is softmax probabilities
        if outputs[0].shape[-1] == 3:
            probs = outputs[0][0]
            predicted_class = np.argmax(probs)
            class_names = ['Real', 'Fake', 'Mask']
            print(f"\n  Predicted probabilities:")
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                marker = "←" if i == predicted_class else ""
                print(f"    {name}: {prob:.4f} {marker}")

        return True

    except ImportError:
        print("⚠ onnxruntime not installed, skipping inference test")
        print("  Install with: pip install onnxruntime")
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert MiniFASNet PyTorch model to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert MiniFASNet V2
  python convert_minifasnet.py \\
      --model-path ../../models/minifasnet_v2.pth \\
      --output ../../models/minifasnet_v2.onnx \\
      --model-type v2

  # Convert MiniFASNet V1SE
  python convert_minifasnet.py \\
      --model-path ../../models/minifasnet_v1se.pth \\
      --output ../../models/minifasnet_v1se.onnx \\
      --model-type v1se
        """
    )

    parser.add_argument('--model-path', required=True,
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--output', required=True,
                        help='Output ONNX file path')
    parser.add_argument('--model-type', default='v2', choices=['v2', 'v1se'],
                        help='Model type (default: v2)')
    parser.add_argument('--input-size', type=int, default=80,
                        help='Input image size (default: 80)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Do not simplify ONNX model')
    parser.add_argument('--no-test', action='store_true',
                        help='Do not run inference test')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    try:
        # Load model
        model = load_minifasnet_model(
            args.model_path,
            args.model_type,
            args.input_size
        )

        # Export to ONNX
        success = export_to_onnx(
            model,
            args.output,
            args.input_size,
            args.opset,
            simplify=not args.no_simplify
        )

        if not success:
            print("\n✗ Export failed")
            return 1

        # Test inference
        if not args.no_test:
            test_success = test_onnx_inference(args.output, args.input_size)
            if not test_success:
                print("\n⚠ Inference test failed, but model was exported")

        print(f"\n{'='*60}")
        print("✓ Conversion completed successfully!")
        print(f"{'='*60}")
        print(f"\nONNX model saved to: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Quantize to INT8:")
        print(f"     python quantization.py \\")
        print(f"       --model {args.output} \\")
        print(f"       --output {args.output.replace('.onnx', '_int8.onnx')} \\")
        print(f"       --method static \\")
        print(f"       --model-type minifasnet \\")
        print(f"       --calib-data ../../data/calibration")
        print(f"\n  2. Test with SDK:")
        print(f"     python -m ocfa.test_liveness")

        return 0

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
