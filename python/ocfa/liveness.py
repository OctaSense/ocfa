"""
OCFA Face SDK - Liveness Detection Module

RGB-IR dual-modal liveness detection using MiniFASNet model.
Includes IR face detection enhancement for presentation attack detection (PAD).
"""

import numpy as np
from typing import Tuple, Optional
from .models.minifasnet import MiniFASNetModel
from .models.ir_face_detector import IRFaceDetector, LivenessDetectorWithIRCheck


class LivenessDetector:
    """RGB-IR liveness detector wrapper with optional IR face detection enhancement"""

    def __init__(self,
                 model_path: str,
                 threshold: float = 0.90,
                 device: str = 'cpu',
                 use_ir_detection: bool = True,
                 ir_weights: Optional[Tuple[float, float]] = None):
        """
        Initialize liveness detector

        Args:
            model_path: Path to MiniFASNet model
            threshold: Liveness threshold [0, 1]
            device: 'cpu' or 'cuda'
            use_ir_detection: Whether to use IR face detection for enhanced PAD
            ir_weights: (rgb_weight, ir_weight) for IR-enhanced scoring
                       Default: (0.6, 0.4)
        """
        self.model = MiniFASNetModel(model_path, device=device, use_onnx=True)
        self.threshold = threshold
        self.use_ir_detection = use_ir_detection

        if use_ir_detection:
            # Initialize IR-enhanced detector
            self.enhanced_detector = LivenessDetectorWithIRCheck(
                self.model,
                use_ir_detection=True
            )
            # Set weights if provided
            if ir_weights:
                self.rgb_weight, self.ir_weight = ir_weights
            else:
                self.rgb_weight = 0.6
                self.ir_weight = 0.4
        else:
            self.enhanced_detector = None

    def detect(self, rgb_face: np.ndarray, ir_face: np.ndarray) -> Tuple[bool, float]:
        """
        Detect liveness from RGB and IR faces

        Enhanced with IR face detection:
        - If RGB detects face but IR doesn't → likely presentation attack (photo/screen)
        - If both detect → likely real face
        - If neither detects → ambiguous (use basic RGB score)

        Args:
            rgb_face: RGB face (112, 112, 3)
            ir_face: IR face (112, 112) or (112, 112, 1)

        Returns:
            (passed, score)
            - passed: True if liveness check passed
            - score: Liveness confidence [0, 1]
        """
        if self.use_ir_detection and self.enhanced_detector:
            # Use IR-enhanced detection
            passed, score = self.enhanced_detector.detect(rgb_face, ir_face)
        else:
            # Use basic RGB-only detection
            score, is_real = self.model.detect_liveness(rgb_face, ir_face)
            passed = score >= self.threshold

        return passed, float(score)

    def set_threshold(self, threshold: float):
        """Set liveness threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be in [0, 1]")
        self.threshold = threshold

    def set_ir_detection(self, enabled: bool):
        """Enable/disable IR face detection enhancement"""
        self.use_ir_detection = enabled

    def set_ir_weights(self, rgb_weight: float, ir_weight: float):
        """Set weights for IR-enhanced scoring"""
        if not (0.0 <= rgb_weight <= 1.0) or not (0.0 <= ir_weight <= 1.0):
            raise ValueError("Weights must be in [0, 1]")
        if not np.isclose(rgb_weight + ir_weight, 1.0, atol=0.01):
            raise ValueError("Weights must sum to approximately 1.0")
        self.rgb_weight = rgb_weight
        self.ir_weight = ir_weight
