"""
OCFA Face SDK - Main SDK Interface

This module provides the main SDK interface that integrates all components.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from .config import OCFAConfig
from .preprocessing import ImagePreprocessor
from .liveness import LivenessDetector
from .quality import QualityAssessor
from .feature import FeatureExtractor
from .fusion import FeatureFusion
from .comparison import FeatureComparator
from .database import FeatureDatabase
from .utils import Timer


class FaceRecognitionResult:
    """Face recognition result"""

    def __init__(self):
        self.liveness_passed = False
        self.liveness_score = 0.0
        self.quality_passed = False
        self.quality_score = 0.0
        self.feature_extracted = False
        self.feature = None
        self.rgb_feature = None
        self.ir_feature = None
        self.total_time_ms = 0.0
        self.error_code = 0
        self.error_msg = ""
        self.quality_details = {}


class OCFAFaceSDK:
    """
    OCFA Face Recognition SDK

    Main interface for face recognition with RGB-IR liveness detection.
    """

    def __init__(self, config: Optional[OCFAConfig] = None, config_path: Optional[str] = None):
        """
        Initialize SDK

        Args:
            config: OCFAConfig instance
            config_path: Path to config JSON file (alternative to config)
        """
        if config is None:
            if config_path:
                self.config = OCFAConfig(config_path)
            else:
                self.config = OCFAConfig()  # Use default
        else:
            self.config = config

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all SDK components"""
        # Preprocessing
        preproc_config = self.config.get('preprocessing', {})
        self.preprocessor = ImagePreprocessor(preproc_config)

        # Liveness detection
        liveness_model = self.config.liveness_model_path
        liveness_threshold = self.config.liveness_threshold
        self.liveness_detector = LivenessDetector(
            liveness_model,
            threshold=liveness_threshold,
            device=self.config.device
        )

        # Quality assessment
        quality_config = self.config.get('quality', {})
        self.quality_assessor = QualityAssessor(quality_config)

        # Feature extraction
        rgb_model = self.config.feature_model_rgb_path
        ir_model = self.config.feature_model_ir_path
        self.feature_extractor = FeatureExtractor(
            rgb_model,
            ir_model,
            device=self.config.device
        )

        # Feature fusion
        fusion_config = self.config.get('fusion', {})
        self.feature_fusion = FeatureFusion(fusion_config)

        # Feature comparison
        self.comparator = FeatureComparator()

        # Feature database
        self.database = FeatureDatabase(feature_dim=512)

    def recognize(self,
                  rgb_image: np.ndarray,
                  ir_image: np.ndarray) -> FaceRecognitionResult:
        """
        Complete recognition pipeline (to feature extraction)

        Pipeline:
        1. Preprocessing
        2. Liveness detection
        3. Quality assessment
        4. Feature extraction
        5. Feature fusion

        Args:
            rgb_image: RGB image (H, W, 3), uint8
            ir_image: IR image (H, W) or (H, W, 1), uint8

        Returns:
            FaceRecognitionResult
        """
        result = FaceRecognitionResult()
        timer = Timer()

        with timer:
            try:
                # Step 1: Preprocessing
                rgb_preprocessed = self.preprocessor.preprocess_rgb(rgb_image)
                ir_preprocessed = self.preprocessor.preprocess_ir(ir_image)

                # Crop to face size
                face_size = self.config.face_size
                rgb_face = self.preprocessor.crop_face(rgb_preprocessed, face_size)
                ir_face = self.preprocessor.crop_face(ir_preprocessed, face_size)

                # Step 2: Liveness detection
                liveness_passed, liveness_score = self.liveness_detector.detect(
                    rgb_face, ir_face
                )
                result.liveness_passed = liveness_passed
                result.liveness_score = liveness_score

                if not liveness_passed:
                    result.error_code = 1
                    result.error_msg = "Liveness detection failed"
                    return result

                # Step 3: Quality assessment
                quality_passed, quality_score, quality_details = self.quality_assessor.assess(
                    rgb_face
                )
                result.quality_passed = quality_passed
                result.quality_score = quality_score
                result.quality_details = quality_details

                if not quality_passed:
                    result.error_code = 2
                    result.error_msg = "Quality assessment failed"
                    return result

                # Step 4: Feature extraction
                rgb_feature, ir_feature = self.feature_extractor.extract(rgb_face, ir_face)
                result.rgb_feature = rgb_feature
                result.ir_feature = ir_feature

                # Step 5: Feature fusion
                fused_feature = self.feature_fusion.fuse(
                    rgb_feature,
                    ir_feature,
                    rgb_quality=quality_score,
                    ir_quality=1.0,  # IR quality always assumed good
                    rgb_image=rgb_image
                )
                result.feature = fused_feature
                result.feature_extracted = True

            except Exception as e:
                result.error_code = -1
                result.error_msg = str(e)

        result.total_time_ms = timer.get_elapsed_ms()
        return result

    # Feature comparison methods

    def compare_features(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Compare two features (1:1)

        Args:
            feature1: First feature (512,)
            feature2: Second feature (512,)

        Returns:
            Similarity [0, 1]
        """
        return self.comparator.compare(feature1, feature2)

    # Feature database methods

    def add_user(self, user_id: bytes, feature: np.ndarray) -> bool:
        """
        Add user to feature database

        Args:
            user_id: 16-byte user ID
            feature: Feature vector (512,)

        Returns:
            True if added successfully
        """
        return self.database.add_user(user_id, feature)

    def update_user(self, user_id: bytes, feature: np.ndarray) -> bool:
        """Update user feature"""
        return self.database.update_user(user_id, feature)

    def remove_user(self, user_id: bytes) -> bool:
        """Remove user from database"""
        return self.database.remove_user(user_id)

    def search_user(self, query_feature: np.ndarray) -> Tuple[Optional[bytes], float]:
        """
        Search for most similar user

        Returns:
            (user_id, similarity) or (None, 0.0) if not found
        """
        return self.database.search_user(query_feature)

    def search_users(self,
                     query_feature: np.ndarray,
                     threshold: float = 0.70,
                     max_results: int = 5) -> List[Tuple[bytes, float]]:
        """
        Search for multiple similar users

        Returns:
            List of (user_id, similarity) sorted by similarity descending
        """
        return self.database.search_users(query_feature, threshold, max_results)

    def get_user_count(self) -> int:
        """Get number of users in database"""
        return self.database.get_user_count()

    def clear_database(self):
        """Clear all users from database"""
        self.database.clear()

    # Utility methods

    def get_version(self) -> str:
        """Get SDK version"""
        return "1.0.0"

    def get_config(self) -> OCFAConfig:
        """Get configuration"""
        return self.config

    def get_stats(self) -> Dict[str, Any]:
        """Get SDK statistics"""
        return {
            'version': self.get_version(),
            'user_count': self.get_user_count(),
            'memory_usage_bytes': self.database.get_memory_usage(),
            'device': self.config.device,
            'liveness_threshold': self.config.liveness_threshold,
            'quality_threshold': self.config.quality_threshold
        }
