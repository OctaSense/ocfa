"""
OCFA Face SDK - IR Face Detector

This module detects if a face is present in IR image.
Used as an additional liveness check: if RGB shows a face but IR doesn't,
it indicates a photo/screen attack (no thermal signature).
"""

import numpy as np
import cv2
from typing import Tuple


class IRFaceDetector:
    """
    IR Image Face Detector

    Detects whether a face is present in an IR image based on:
    1. Thermal contrast (face is warmer than background)
    2. Face shape/contour characteristics
    3. Histogram analysis

    This is a simple baseline detector. For production, consider using:
    - Haar Cascade trained on IR images
    - Deep learning model trained on IR face data
    - Thermal signature analysis
    """

    def __init__(self):
        """Initialize IR face detector"""
        # Threshold for determining if image has significant thermal content
        # IR images with faces have higher variance and specific intensity distribution
        self.thermal_variance_threshold = 100.0  # Adjust based on calibration
        self.face_intensity_range = (80, 200)     # Expected face intensity range (0-255)
        self.background_intensity_max = 100       # Typical background is dimmer

    def detect_face_in_ir(self, ir_image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if a face is present in IR image

        Args:
            ir_image: IR image (H, W) or (H, W, 1), uint8, [0, 255]

        Returns:
            (has_face, confidence)
            - has_face: True if face detected in IR image
            - confidence: float in [0, 1], confidence level of detection
        """
        # Normalize image shape
        if len(ir_image.shape) == 3:
            ir_image = ir_image[:, :, 0]

        if ir_image.size == 0:
            return False, 0.0

        # Convert to float for analysis
        ir_float = ir_image.astype(np.float32)

        # Compute features
        variance = np.var(ir_float)
        mean = np.mean(ir_float)
        std = np.std(ir_float)

        # Detect bright regions (likely face/thermal signature)
        bright_regions = (ir_float > 100).astype(np.uint8)
        bright_ratio = np.sum(bright_regions) / ir_image.size

        # Analyze histogram
        hist_confidence = self._analyze_histogram(ir_float)

        # Analyze edges (faces have characteristic edge patterns)
        edge_confidence = self._detect_face_edges(ir_image)

        # Compute detection confidence
        # Factors:
        # 1. Variance: real faces have higher variance than blank images
        # 2. Bright regions: faces appear brighter in IR
        # 3. Edge patterns: faces have specific edge characteristics

        variance_score = min(1.0, variance / self.thermal_variance_threshold)
        bright_score = min(1.0, bright_ratio * 3)  # Expected ~30% bright pixels

        # Weighted confidence
        confidence = (
            variance_score * 0.3 +      # Variance of thermal signature
            bright_score * 0.2 +        # Bright region ratio
            hist_confidence * 0.25 +    # Histogram characteristics
            edge_confidence * 0.25      # Edge pattern detection
        )

        # Face is detected if:
        # 1. Variance is above threshold (enough thermal variation)
        # 2. Has reasonable bright regions (warm signature)
        # 3. Edge patterns match face-like structures
        has_face = (
            variance > self.thermal_variance_threshold and
            bright_ratio > 0.1 and  # At least 10% bright pixels
            confidence > 0.4        # Overall confidence above threshold
        )

        return has_face, float(confidence)

    def _analyze_histogram(self, ir_image: np.ndarray) -> float:
        """
        Analyze histogram to detect face-like intensity distribution

        Real faces in IR typically show:
        - Peak in higher intensity range (warm areas)
        - Secondary peak in lower range (cooler background)
        - Specific distribution pattern
        """
        hist = cv2.calcHist([ir_image.astype(np.uint8)], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Calculate entropy (more uniform distribution = less likely face)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))

        # Good face images have moderate entropy (not uniform, not too concentrated)
        # Max entropy = 8 bits = 8
        entropy_score = 1.0 - abs(entropy - 4.0) / 4.0
        entropy_score = max(0.0, min(1.0, entropy_score))

        # Analyze intensity distribution
        # Faces should have bimodal distribution (face region + background)
        mean_intensity = np.mean(ir_image)
        intensity_range = np.max(ir_image) - np.min(ir_image)

        # Good contrast and reasonable intensity
        intensity_score = min(1.0, intensity_range / 100.0)

        # Combined histogram confidence
        histogram_confidence = (entropy_score * 0.6 + intensity_score * 0.4)

        return float(histogram_confidence)

    def _detect_face_edges(self, ir_image: np.ndarray) -> float:
        """
        Detect face-like edge patterns in IR image

        Faces have characteristic edge structures:
        - Circular/oval outline
        - Eye regions (higher intensity contrast)
        - Facial features
        """
        # Use Canny edge detection
        edges = cv2.Canny(ir_image, 50, 150)

        # Calculate edge density
        edge_ratio = np.sum(edges > 0) / edges.size

        # Faces have moderate edge density (not too sparse, not too dense)
        edge_score = min(1.0, edge_ratio * 15)  # Expected ~7% edges

        # Detect contours and analyze shape
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for circular-like contours (face-like shapes)
        if len(contours) == 0:
            shape_score = 0.0
        else:
            # Analyze largest contours
            areas = [cv2.contourArea(c) for c in contours]
            sorted_indices = np.argsort(areas)[::-1][:3]  # Top 3 contours

            shape_score = 0.0
            for idx in sorted_indices:
                contour = contours[idx]
                if len(contour) >= 5:  # Need at least 5 points
                    # Fit ellipse
                    ellipse = cv2.fitEllipse(contour)

                    # Check circularity (face is roughly circular)
                    (center, (width, height), angle) = ellipse
                    aspect_ratio = min(width, height) / (max(width, height) + 1e-5)

                    # Good circularity for faces: aspect ratio > 0.7
                    circularity = max(0.0, (aspect_ratio - 0.5) / 0.3)
                    shape_score = max(shape_score, min(1.0, circularity))

        # Combined edge confidence
        edge_confidence = (edge_score * 0.4 + shape_score * 0.6)

        return float(edge_confidence)

    def get_ir_quality_score(self, ir_image: np.ndarray) -> float:
        """
        Get overall quality score of IR image

        Returns:
            Quality score in [0, 1]
        """
        if len(ir_image.shape) == 3:
            ir_image = ir_image[:, :, 0]

        ir_float = ir_image.astype(np.float32)

        # Check if image is too dark (no thermal signal)
        mean = np.mean(ir_float)
        if mean < 30:
            return 0.0  # Too dark

        # Check if image is saturated (camera overexposed)
        if mean > 220:
            return 0.5  # Overexposed

        # Check contrast
        std = np.std(ir_float)
        if std < 20:
            return 0.5  # Too little contrast

        # Good quality
        return min(1.0, (std / 50.0))


class LivenessDetectorWithIRCheck:
    """
    Enhanced liveness detector that combines:
    1. MiniFASNet RGB liveness detection
    2. IR face detection (checks for thermal signature)

    Security improvement:
    - If RGB detects face but IR doesn't → attack (photo/screen)
    - If both detect face → likely real face
    - Provides better protection against presentation attacks
    """

    def __init__(self, minifasnet_detector, use_ir_detection: bool = True):
        """
        Initialize enhanced liveness detector

        Args:
            minifasnet_detector: MiniFASNetModel instance for RGB detection
            use_ir_detection: Whether to enable IR face detection enhancement
        """
        self.minifasnet = minifasnet_detector
        self.ir_detector = IRFaceDetector()
        self.use_ir_detection = use_ir_detection

    def detect(self, rgb_face: np.ndarray, ir_face: np.ndarray) -> Tuple[bool, float]:
        """
        Enhanced liveness detection with IR face verification

        Args:
            rgb_face: RGB face image (112, 112, 3)
            ir_face: IR face image (112, 112) or (112, 112, 1)

        Returns:
            (liveness_passed, liveness_score)
        """
        # Step 1: RGB-based liveness detection (MiniFASNet)
        rgb_score, _ = self.minifasnet.detect_liveness(rgb_face, ir_face)

        # Step 2: IR face detection (additional security check)
        if self.use_ir_detection:
            ir_has_face, ir_confidence = self.ir_detector.detect_face_in_ir(ir_face)

            # If RGB shows face but IR doesn't → likely attack
            # RGB score only matters if IR also detects face
            if not ir_has_face:
                # No thermal signature → photo/screen attack
                return False, rgb_score * 0.5  # Reduce score to indicate attack

            # Both RGB and IR detect face → likely real
            # Use combined confidence
            combined_score = (rgb_score * 0.6 + ir_confidence * 0.4)
        else:
            # Only use RGB detection
            combined_score = rgb_score

        # Threshold check
        liveness_passed = combined_score > 0.5

        return liveness_passed, combined_score
