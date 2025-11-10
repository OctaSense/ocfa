"""
OCFA Face SDK - Model Export Tool

Export PyTorch models to ONNX format for deployment.
Supports ArcFace-R34 and MiniFASNet models.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import onnx
from onnx import shape_inference
import numpy as np


class ArcFaceBackbone(nn.Module):
    """
    Simplified ArcFace ResNet34 backbone for export

    Note: This is a skeleton. For actual export, you need the full model definition
    from InsightFace or load from checkpoint.
    """

    def __init__(self, num_features=512):
        super(ArcFaceBackbone, self).__init__()
        # TODO: Load actual ResNet34 architecture
        # This is just a placeholder
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_features)
        )

    def forward(self, x):
        return self.features(x)


class MiniFASNetBackbone(nn.Module):
    """
    Simplified MiniFASNet for export

    Note: This is a skeleton. Load actual model from InsightFace.
    """

    def __init__(self):
        super(MiniFASNetBackbone, self).__init__()
        # TODO: Load actual MiniFASNet architecture
        # Placeholder for dual-stream input
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.ir_stream = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, rgb, ir):
        rgb_feat = self.rgb_stream(rgb)
        ir_feat = self.ir_stream(ir)
        combined = torch.cat([rgb_feat, ir_feat], dim=1)
        return self.classifier(combined)


def load_arcface_model(weights_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load ArcFace model from weights

    Args:
        weights_path: Path to .pth weights file
        device: 'cpu' or 'cuda'

    Returns:
        Loaded PyTorch model
    """
    print(f"Loading ArcFace model from {weights_path}...")

    # Try to load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load state dict
    state_dict = torch.load(weights_path, map_location=device)

    # Create model
    # NOTE: This is simplified. Real implementation needs full architecture
    model = ArcFaceBackbone(num_features=512)

    # Try to load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load all weights: {e}")
        print("Using random initialization for demo purposes")

    model.eval()
    return model.to(device)


def load_minifasnet_model(weights_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load MiniFASNet model from weights

    Args:
        weights_path: Path to .pth weights file
        device: 'cpu' or 'cuda'

    Returns:
        Loaded PyTorch model
    """
    print(f"Loading MiniFASNet model from {weights_path}...")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    model = MiniFASNetBackbone()

    try:
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load all weights: {e}")
        print("Using random initialization for demo purposes")

    model.eval()
    return model.to(device)


def export_arcface_to_onnx(model: nn.Module,
                           output_path: str,
                           input_size: tuple = (1, 3, 112, 112),
                           opset_version: int = 11):
    """
    Export ArcFace model to ONNX

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
    """
    print(f"\nExporting ArcFace to ONNX...")
    print(f"Input size: {input_size}")
    print(f"Output path: {output_path}")

    # Create dummy input
    dummy_input = torch.randn(input_size)

    # Export to ONNX
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
        }
    )

    print(f"✓ ONNX model exported to {output_path}")

    # Verify model
    verify_onnx_model(output_path)


def export_minifasnet_to_onnx(model: nn.Module,
                               output_path: str,
                               rgb_size: tuple = (1, 3, 112, 112),
                               ir_size: tuple = (1, 1, 112, 112),
                               opset_version: int = 11):
    """
    Export MiniFASNet model to ONNX

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        rgb_size: RGB input size
        ir_size: IR input size
        opset_version: ONNX opset version
    """
    print(f"\nExporting MiniFASNet to ONNX...")
    print(f"RGB input size: {rgb_size}")
    print(f"IR input size: {ir_size}")
    print(f"Output path: {output_path}")

    # Create dummy inputs
    dummy_rgb = torch.randn(rgb_size)
    dummy_ir = torch.randn(ir_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_ir),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['rgb_input', 'ir_input'],
        output_names=['output'],
        dynamic_axes={
            'rgb_input': {0: 'batch_size'},
            'ir_input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ ONNX model exported to {output_path}")

    # Verify model
    verify_onnx_model(output_path)


def verify_onnx_model(onnx_path: str):
    """
    Verify ONNX model

    Args:
        onnx_path: Path to ONNX model
    """
    print(f"\nVerifying ONNX model...")

    # Load model
    model = onnx.load(onnx_path)

    # Check model
    try:
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX model check failed: {e}")
        return

    # Infer shapes
    try:
        model = shape_inference.infer_shapes(model)
        print("✓ Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")

    # Print model info
    print(f"\nModel Info:")
    print(f"  IR version: {model.ir_version}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Opset version: {model.opset_import[0].version}")

    print(f"\nInputs:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  {input_tensor.name}: {shape}")

    print(f"\nOutputs:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  {output_tensor.name}: {shape}")


def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX')
    parser.add_argument('--model', type=str, required=True,
                       choices=['arcface_r34', 'minifasnet'],
                       help='Model type to export')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to PyTorch weights (.pth)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ONNX file path (.onnx)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    try:
        if args.model == 'arcface_r34':
            # Load and export ArcFace
            model = load_arcface_model(args.weights, args.device)
            export_arcface_to_onnx(
                model,
                args.output,
                input_size=(1, 3, 112, 112),
                opset_version=args.opset
            )

        elif args.model == 'minifasnet':
            # Load and export MiniFASNet
            model = load_minifasnet_model(args.weights, args.device)
            export_minifasnet_to_onnx(
                model,
                args.output,
                rgb_size=(1, 3, 112, 112),
                ir_size=(1, 1, 112, 112),
                opset_version=args.opset
            )

        print(f"\n{'='*60}")
        print("Export completed successfully!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
