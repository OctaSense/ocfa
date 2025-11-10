"""
OCFA Face SDK - Feature Comparison Module

Compare face features using cosine similarity.
"""

import numpy as np
from typing import List, Tuple
from .utils import cosine_similarity, batch_cosine_similarity


class FeatureComparator:
    """Feature comparison using cosine similarity"""

    def __init__(self):
        """Initialize feature comparator"""
        pass

    def compare(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Compare two features (1:1 matching)

        Args:
            feature1: First feature (512,), L2 normalized
            feature2: Second feature (512,), L2 normalized

        Returns:
            Similarity score [0, 1]
        """
        return cosine_similarity(feature1, feature2)

    def compare_batch(self, query: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Compare query feature against multiple features (1:N matching)

        Args:
            query: Query feature (512,), L2 normalized
            features: Feature matrix (N, 512), L2 normalized

        Returns:
            Similarity scores (N,), [0, 1]
        """
        return batch_cosine_similarity(query, features)

    def verify(self, feature1: np.ndarray, feature2: np.ndarray, threshold: float = 0.70) -> Tuple[bool, float]:
        """
        Verify if two features match

        Args:
            feature1: First feature
            feature2: Second feature
            threshold: Similarity threshold

        Returns:
            (is_match, similarity)
        """
        similarity = self.compare(feature1, feature2)
        is_match = similarity >= threshold
        return is_match, similarity
