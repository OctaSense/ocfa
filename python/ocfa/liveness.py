"""
OCFA Face SDK - Liveness Detection Module

RGB-IR dual-modal liveness detection using MiniFASNet model.
"""

import numpy as np
from typing import Tuple, Optional
from .models.minifasnet import MiniFASNetModel


class LivenessDetector:
    """RGB-IR liveness detector wrapper"""

    def __init__(self, model_path: str, threshold: float = 0.90, device: str = 'cpu'):
        """
        Initialize liveness detector

        Args:
            model_path: Path to MiniFASNet model
            threshold: Liveness threshold [0, 1]
            device: 'cpu' or 'cuda'
        """
        self.model = MiniFASNetModel(model_path, device=device, use_onnx=True)
        self.threshold = threshold

    def detect(self, rgb_face: np.ndarray, ir_face: np.ndarray) -> Tuple[bool, float]:
        """
        Detect liveness from RGB and IR faces

        Args:
            rgb_face: RGB face (112, 112, 3)
            ir_face: IR face (112, 112) or (112, 112, 1)

        Returns:
            (passed, score)
            - passed: True if liveness check passed
            - score: Liveness confidence [0, 1]
        """
        score, is_real = self.model.detect_liveness(rgb_face, ir_face)
        passed = score >= self.threshold
        return passed, float(score)

    def set_threshold(self, threshold: float):
        """Set liveness threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be in [0, 1]")
        self.threshold = threshold
