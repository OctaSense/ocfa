"""
OCFA Face SDK - Feature Fusion Module

Adaptive fusion of RGB and IR features based on illumination and quality.
"""

import numpy as np
from typing import Optional
from .utils import l2_normalize, estimate_illumination


class FeatureFusion:
    """Adaptive RGB-IR feature fusion"""

    def __init__(self, config: dict):
        """
        Initialize feature fusion

        Args:
            config: Fusion configuration from OCFAConfig
        """
        self.strategy = config.get('strategy', 'adaptive')
        self.rgb_weight_high = config.get('rgb_weight_high_light', 0.8)
        self.ir_weight_high = config.get('ir_weight_high_light', 0.2)
        self.rgb_weight_medium = config.get('rgb_weight_medium_light', 0.5)
        self.ir_weight_medium = config.get('ir_weight_medium_light', 0.5)
        self.rgb_weight_low = config.get('rgb_weight_low_light', 0.2)
        self.ir_weight_low = config.get('ir_weight_low_light', 0.8)
        self.light_threshold_high = config.get('light_threshold_high', 100)
        self.light_threshold_low = config.get('light_threshold_low', 10)

    def fuse(self,
             rgb_feature: np.ndarray,
             ir_feature: np.ndarray,
             rgb_quality: float = 1.0,
             ir_quality: float = 1.0,
             illumination_lux: Optional[int] = None,
             rgb_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuse RGB and IR features adaptively

        Args:
            rgb_feature: RGB feature (512,)
            ir_feature: IR feature (512,)
            rgb_quality: RGB quality score [0, 1]
            ir_quality: IR quality score [0, 1]
            illumination_lux: Illumination in lux (optional)
            rgb_image: RGB image for estimating illumination (optional)

        Returns:
            Fused feature (512,), L2 normalized
        """
        if self.strategy == 'adaptive':
            return self._adaptive_fusion(
                rgb_feature, ir_feature,
                rgb_quality, ir_quality,
                illumination_lux, rgb_image
            )
        elif self.strategy == 'average':
            return self._average_fusion(rgb_feature, ir_feature)
        elif self.strategy == 'rgb_only':
            return rgb_feature
        elif self.strategy == 'ir_only':
            return ir_feature
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    def _adaptive_fusion(self,
                        rgb_feature: np.ndarray,
                        ir_feature: np.ndarray,
                        rgb_quality: float,
                        ir_quality: float,
                        illumination_lux: Optional[int],
                        rgb_image: Optional[np.ndarray]) -> np.ndarray:
        """
        Adaptive fusion based on illumination and quality

        Algorithm:
        1. Determine base weights from illumination
        2. Adjust weights by quality scores
        3. Normalize weights
        4. Weighted sum
        5. L2 normalize result
        """
        # Estimate illumination if not provided
        if illumination_lux is None:
            if rgb_image is not None:
                illumination_lux = estimate_illumination(rgb_image)
            else:
                illumination_lux = 50  # Default medium light

        # Step 1: Base weights from illumination
        if illumination_lux >= self.light_threshold_high:
            alpha, beta = self.rgb_weight_high, self.ir_weight_high
        elif illumination_lux >= self.light_threshold_low:
            alpha, beta = self.rgb_weight_medium, self.ir_weight_medium
        else:
            alpha, beta = self.rgb_weight_low, self.ir_weight_low

        # Step 2: Adjust by quality
        alpha *= rgb_quality
        beta *= ir_quality

        # Step 3: Normalize weights
        weight_sum = alpha + beta
        if weight_sum > 0:
            alpha /= weight_sum
            beta /= weight_sum
        else:
            alpha, beta = 0.5, 0.5

        # Step 4: Weighted fusion
        fused = alpha * rgb_feature + beta * ir_feature

        # Step 5: L2 normalize
        fused = l2_normalize(fused)

        return fused

    def _average_fusion(self,
                       rgb_feature: np.ndarray,
                       ir_feature: np.ndarray) -> np.ndarray:
        """Simple average fusion"""
        fused = 0.5 * rgb_feature + 0.5 * ir_feature
        return l2_normalize(fused)
