"""
OCFA Face SDK - MiniFASNet Model Loader

This module loads MiniFASNet model for RGB-IR liveness detection.
Supports both PyTorch and ONNX Runtime inference.
"""

import numpy as np
from typing import Tuple
import os


class MiniFASNetModel:
    """
    MiniFASNet RGB-IR dual-stream liveness detection model

    Input: RGB face (112x112x3) + IR face (112x112x1)
    Output: Liveness score [0, 1] (higher = real face)
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 use_onnx: bool = True):
        """
        Initialize MiniFASNet model

        Args:
            model_path: Path to model file (.onnx or .pth)
            device: 'cpu' or 'cuda'
            use_onnx: Whether to use ONNX Runtime (True) or PyTorch (False)
        """
        self.model_path = model_path
        self.device = device
        self.use_onnx = use_onnx
        self.session = None
        self.model = None
        # MiniFASNet v2 uses 80x80 input
        self.input_size = (80, 80)

        self._load_model()

    def _load_model(self):
        """Load model from file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_onnx_model(self):
        """Load ONNX model using ONNX Runtime"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        # Assuming dual inputs: rgb_input, ir_input
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for PyTorch inference")

        # TODO: Implement PyTorch model loading
        raise NotImplementedError("PyTorch inference not yet implemented. Use ONNX.")

    def detect_liveness(self,
                        rgb_face: np.ndarray,
                        ir_face: np.ndarray) -> Tuple[float, bool]:
        """
        Detect liveness from RGB and IR face images

        Args:
            rgb_face: RGB face image (112, 112, 3), uint8, [0, 255]
            ir_face: IR face image (112, 112) or (112, 112, 1), uint8, [0, 255]

        Returns:
            (liveness_score, is_real)
            - liveness_score: float in [0, 1], higher = more likely real
            - is_real: True if classified as real face
        """
        # Preprocess
        rgb_tensor = self._preprocess_rgb(rgb_face)
        ir_tensor = self._preprocess_ir(ir_face)

        # Inference
        if self.use_onnx:
            score = self._infer_onnx(rgb_tensor, ir_tensor)
        else:
            score = self._infer_pytorch(rgb_tensor, ir_tensor)

        # Binary classification (0.5 threshold)
        is_real = score > 0.5

        return score, is_real

    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess RGB face image

        Args:
            image: RGB image (112, 112, 3)

        Returns:
            Preprocessed tensor (1, 3, 112, 112)
        """
        import cv2

        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _preprocess_ir(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess IR face image

        Args:
            image: IR image (112, 112) or (112, 112, 1)

        Returns:
            Preprocessed tensor (1, 1, 112, 112)
        """
        import cv2

        # Handle shape
        if len(image.shape) == 3:
            image = image[:, :, 0]

        if image.shape != self.input_size:
            image = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _infer_onnx(self,
                    rgb_tensor: np.ndarray,
                    ir_tensor: np.ndarray) -> float:
        """
        Run ONNX inference

        Args:
            rgb_tensor: RGB tensor (1, 3, 80, 80)
            ir_tensor: IR tensor (1, 1, 80, 80) - may not be used for single-stream models

        Returns:
            Liveness score [0, 1]
        """
        # Build input dict
        # This depends on model's input names
        # For MiniFASNet v2 (single-stream RGB): only use RGB
        inputs = {}
        if len(self.input_names) == 2:
            # Dual-stream model
            inputs[self.input_names[0]] = rgb_tensor
            inputs[self.input_names[1]] = ir_tensor
        elif len(self.input_names) == 1:
            # Single-stream model - use RGB only
            inputs[self.input_names[0]] = rgb_tensor
        else:
            raise ValueError(f"Unexpected number of inputs: {len(self.input_names)}")

        outputs = self.session.run(self.output_names, inputs)

        # Output shape: (1, 3) for [real, fake, mask] scores (MiniFASNet v2)
        # or (1, 2) for [fake, real] scores
        # or (1, 1) for single score
        output = outputs[0][0]

        if len(output) >= 3:
            # MiniFASNet v2: [real_prob, fake_prob, mask_prob]
            # Use real_prob as the liveness score
            real_score = float(output[0])
        elif len(output) == 2:
            # Binary classification: [fake_prob, real_prob] or [real_prob, fake_prob]
            # Assume second element is real_prob
            real_score = float(output[1])
        else:
            # Single score
            real_score = float(output[0])

        return np.clip(real_score, 0.0, 1.0)

    def _infer_pytorch(self,
                       rgb_tensor: np.ndarray,
                       ir_tensor: np.ndarray) -> float:
        """Run PyTorch inference"""
        # TODO: Implement PyTorch inference
        raise NotImplementedError("PyTorch inference not yet implemented")

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'use_onnx': self.use_onnx,
            'input_size': self.input_size
        }
