"""
OCFA Face SDK - ArcFace Model Loader

This module loads ArcFace-R34 model for face feature extraction.
Supports both PyTorch and ONNX Runtime inference.
"""

import numpy as np
from typing import Optional, Tuple
import os


class ArcFaceModel:
    """
    ArcFace-R34 face feature extraction model

    Input: RGB or IR face image (112x112)
    Output: 512-dim feature vector (L2 normalized)
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 use_onnx: bool = True):
        """
        Initialize ArcFace model

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
        self.input_size = (112, 112)
        self.feature_dim = 512

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

        # TODO: Configure session options
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
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for PyTorch inference")

        # TODO: Implement PyTorch model loading
        # This would require the model architecture definition
        raise NotImplementedError("PyTorch inference not yet implemented. Use ONNX.")

    def extract_feature(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim feature vector from face image

        Args:
            face_image: Face image (112, 112, 3) for RGB or (112, 112, 1) for IR
                       Value range: [0, 255], uint8

        Returns:
            Feature vector (512,), L2 normalized
        """
        # Preprocess
        input_tensor = self._preprocess(face_image)

        # Inference
        if self.use_onnx:
            feature = self._infer_onnx(input_tensor)
        else:
            feature = self._infer_pytorch(input_tensor)

        # L2 normalize
        feature = self._l2_normalize(feature)

        return feature

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input

        Steps:
        1. Resize to 112x112 if needed
        2. Convert to float32
        3. Normalize to [0, 1] or [-1, 1]
        4. Transpose to CHW format
        5. Add batch dimension

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Preprocessed tensor (1, C, 112, 112)
        """
        import cv2

        # Handle grayscale (IR) images
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            pass  # Already in correct format
        elif len(image.shape) == 3 and image.shape[2] == 3:
            pass  # RGB image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Resize if needed
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # TODO: Apply mean/std normalization if required by model
        # mean = np.array([0.5, 0.5, 0.5])
        # std = np.array([0.5, 0.5, 0.5])
        # image = (image - mean) / std

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0][0]  # Remove batch dimension

    def _infer_pytorch(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run PyTorch inference"""
        # TODO: Implement PyTorch inference
        raise NotImplementedError("PyTorch inference not yet implemented")

    def _l2_normalize(self, feature: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """L2 normalize feature vector"""
        norm = np.linalg.norm(feature)
        norm = max(norm, eps)
        return feature / norm

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'use_onnx': self.use_onnx,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim
        }
