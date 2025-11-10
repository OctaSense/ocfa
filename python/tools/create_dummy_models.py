"""
Create dummy ONNX models for testing purposes

This script creates minimal ONNX models with the correct input/output shapes
for testing the C++ SDK without requiring actual trained models.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import sys
from pathlib import Path


def create_arcface_model(output_path: str):
    """
    Create dummy ArcFace-R34 ONNX model

    Input: RGB image (1, 3, 112, 112)
    Output: Feature vector (1, 512)
    """
    print("Creating dummy ArcFace model...")

    # Input
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 3, 112, 112]
    )

    # Output
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 512]
    )

    # Create a simple identity-like operation
    # In reality, this would be a ResNet-34 backbone
    # For testing, we just use GlobalAveragePool + FC

    # Flatten: (1, 3, 112, 112) -> (1, 37632)
    flatten_node = helper.make_node(
        'Flatten',
        inputs=['input'],
        outputs=['flattened'],
        axis=1
    )

    # Random weight matrix: (37632, 512)
    np.random.seed(42)
    weight_data = np.random.randn(37632, 512).astype(np.float32) * 0.01
    weight_tensor = helper.make_tensor(
        'fc_weight',
        TensorProto.FLOAT,
        [37632, 512],
        weight_data.flatten().tolist()
    )

    # MatMul: (1, 37632) x (37632, 512) -> (1, 512)
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['flattened', 'fc_weight'],
        outputs=['fc_output']
    )

    # L2 Normalize
    # First compute L2 norm
    reducenorm_node = helper.make_node(
        'ReduceL2',
        inputs=['fc_output'],
        outputs=['norm'],
        axes=[1],
        keepdims=1
    )

    # Then divide by norm
    div_node = helper.make_node(
        'Div',
        inputs=['fc_output', 'norm'],
        outputs=['output']
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[flatten_node, matmul_node, reducenorm_node, div_node],
        name='ArcFace-R34-Dummy',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='OCFA-SDK')
    model.opset_import[0].version = 13

    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(f"✓ Saved to {output_path}")
    print(f"  Input: (1, 3, 112, 112) float32")
    print(f"  Output: (1, 512) float32")


def create_minifasnet_model(output_path: str):
    """
    Create dummy MiniFASNet ONNX model

    Inputs:
      - RGB: (1, 3, 80, 80)
      - IR: (1, 1, 80, 80)
    Output: (1, 2) - [fake_score, real_score]
    """
    print("\nCreating dummy MiniFASNet model...")

    # Inputs
    rgb_input = helper.make_tensor_value_info(
        'rgb', TensorProto.FLOAT, [1, 3, 80, 80]
    )
    ir_input = helper.make_tensor_value_info(
        'ir', TensorProto.FLOAT, [1, 1, 80, 80]
    )

    # Output
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 2]
    )

    # Flatten RGB: (1, 3, 80, 80) -> (1, 19200)
    flatten_rgb = helper.make_node(
        'Flatten',
        inputs=['rgb'],
        outputs=['rgb_flat'],
        axis=1
    )

    # Flatten IR: (1, 1, 80, 80) -> (1, 6400)
    flatten_ir = helper.make_node(
        'Flatten',
        inputs=['ir'],
        outputs=['ir_flat'],
        axis=1
    )

    # Concat: (1, 19200) + (1, 6400) -> (1, 25600)
    concat = helper.make_node(
        'Concat',
        inputs=['rgb_flat', 'ir_flat'],
        outputs=['concat_output'],
        axis=1
    )

    # Random weight: (25600, 2)
    np.random.seed(123)
    weight_data = np.random.randn(25600, 2).astype(np.float32) * 0.01
    weight_tensor = helper.make_tensor(
        'fc_weight',
        TensorProto.FLOAT,
        [25600, 2],
        weight_data.flatten().tolist()
    )

    # Bias: (2,) - slightly favor "real" class
    bias_data = np.array([0.0, 0.5], dtype=np.float32)
    bias_tensor = helper.make_tensor(
        'fc_bias',
        TensorProto.FLOAT,
        [2],
        bias_data.tolist()
    )

    # Gemm: (1, 25600) x (25600, 2) + (2,) -> (1, 2)
    gemm = helper.make_node(
        'Gemm',
        inputs=['concat_output', 'fc_weight', 'fc_bias'],
        outputs=['output'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[flatten_rgb, flatten_ir, concat, gemm],
        name='MiniFASNet-Dummy',
        inputs=[rgb_input, ir_input],
        outputs=[output_tensor],
        initializer=[weight_tensor, bias_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='OCFA-SDK')
    model.opset_import[0].version = 13

    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(f"✓ Saved to {output_path}")
    print(f"  Input 1 (RGB): (1, 3, 80, 80) float32")
    print(f"  Input 2 (IR): (1, 1, 80, 80) float32")
    print(f"  Output: (1, 2) float32")


def main():
    # Get models directory
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OCFA Face SDK - Dummy Model Generator")
    print("=" * 60)
    print("\nWARNING: These are dummy models for testing only!")
    print("They will NOT produce meaningful results.")
    print("Use real trained models for actual deployment.\n")

    # Create models
    arcface_path = models_dir / 'arcface_r34_int8.onnx'
    minifasnet_path = models_dir / 'minifasnet_int8.onnx'

    create_arcface_model(str(arcface_path))
    create_minifasnet_model(str(minifasnet_path))

    print("\n" + "=" * 60)
    print("Dummy models created successfully!")
    print("=" * 60)
    print(f"\nModels saved to: {models_dir}")
    print("\nYou can now test the C++ SDK:")
    print("  cd cpp/build")
    print("  ./examples/demo_recognition")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    try:
        import onnx
    except ImportError:
        print("ERROR: onnx package not installed")
        print("Install with: pip install onnx")
        sys.exit(1)

    main()
