"""
OCFA Face SDK - Feature Extraction Module

Extract face features using ArcFace-R34 model (RGB + IR dual-stream).
"""

import numpy as np
from typing import Tuple
from .models.arcface import ArcFaceModel


class FeatureExtractor:
    """Dual-stream (RGB + IR) feature extractor"""

    def __init__(self,
                 rgb_model_path: str,
                 ir_model_path: str,
                 device: str = 'cpu'):
        """
        Initialize feature extractor

        Args:
            rgb_model_path: Path to RGB ArcFace model
            ir_model_path: Path to IR ArcFace model
            device: 'cpu' or 'cuda'
        """
        self.rgb_model = ArcFaceModel(rgb_model_path, device=device, use_onnx=True)
        self.ir_model = ArcFaceModel(ir_model_path, device=device, use_onnx=True)
        self.feature_dim = 512

    def extract(self, rgb_face: np.ndarray, ir_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from RGB and IR faces

        Args:
            rgb_face: RGB face (112, 112, 3)
            ir_face: IR face (112, 112) or (112, 112, 1)

        Returns:
            (rgb_feature, ir_feature)
            - rgb_feature: 512-dim feature, L2 normalized
            - ir_feature: 512-dim feature, L2 normalized
        """
        rgb_feature = self.rgb_model.extract_feature(rgb_face)
        ir_feature = self.ir_model.extract_feature(ir_face)

        return rgb_feature, ir_feature

    def extract_rgb_only(self, rgb_face: np.ndarray) -> np.ndarray:
        """Extract RGB feature only"""
        return self.rgb_model.extract_feature(rgb_face)

    def extract_ir_only(self, ir_face: np.ndarray) -> np.ndarray:
        """Extract IR feature only"""
        return self.ir_model.extract_feature(ir_face)
