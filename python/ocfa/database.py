"""
OCFA Face SDK - Feature Database Management

In-memory feature database for user management and search.
"""

import numpy as np
from typing import List, Tuple, Optional
from .utils import batch_cosine_similarity, generate_user_id


class FeatureDatabase:
    """In-memory feature database"""

    def __init__(self, feature_dim: int = 512):
        """
        Initialize feature database

        Args:
            feature_dim: Feature dimension (default: 512)
        """
        self.feature_dim = feature_dim
        self.user_ids: List[bytes] = []  # List of 16-byte user IDs
        self.features: List[np.ndarray] = []  # List of feature vectors

    def add_user(self, user_id: bytes, feature: np.ndarray) -> bool:
        """
        Add user to database

        Args:
            user_id: 16-byte user ID
            feature: Feature vector (512,)

        Returns:
            True if added successfully, False if user already exists
        """
        if len(user_id) != 16:
            raise ValueError("User ID must be 16 bytes")

        if feature.shape != (self.feature_dim,):
            raise ValueError(f"Feature must have shape ({self.feature_dim},)")

        # Check if user already exists
        if user_id in self.user_ids:
            return False

        self.user_ids.append(user_id)
        self.features.append(feature.copy())
        return True

    def update_user(self, user_id: bytes, feature: np.ndarray) -> bool:
        """
        Update user feature

        Args:
            user_id: 16-byte user ID
            feature: New feature vector (512,)

        Returns:
            True if updated, False if user not found
        """
        try:
            idx = self.user_ids.index(user_id)
            self.features[idx] = feature.copy()
            return True
        except ValueError:
            return False

    def remove_user(self, user_id: bytes) -> bool:
        """
        Remove user from database

        Args:
            user_id: 16-byte user ID

        Returns:
            True if removed, False if user not found
        """
        try:
            idx = self.user_ids.index(user_id)
            del self.user_ids[idx]
            del self.features[idx]
            return True
        except ValueError:
            return False

    def search_user(self, query_feature: np.ndarray) -> Tuple[Optional[bytes], float]:
        """
        Search for most similar user (1:1 mode)

        Args:
            query_feature: Query feature (512,)

        Returns:
            (user_id, similarity) or (None, 0.0) if database empty
        """
        if len(self.features) == 0:
            return None, 0.0

        # Compute similarities
        features_matrix = np.array(self.features)
        similarities = batch_cosine_similarity(query_feature, features_matrix)

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_user_id = self.user_ids[best_idx]

        return best_user_id, best_similarity

    def search_users(self,
                     query_feature: np.ndarray,
                     threshold: float = 0.70,
                     max_results: int = 5) -> List[Tuple[bytes, float]]:
        """
        Search for multiple similar users (1:N mode)

        Args:
            query_feature: Query feature (512,)
            threshold: Minimum similarity threshold
            max_results: Maximum number of results

        Returns:
            List of (user_id, similarity) sorted by similarity descending
        """
        if len(self.features) == 0:
            return []

        # Compute similarities
        features_matrix = np.array(self.features)
        similarities = batch_cosine_similarity(query_feature, features_matrix)

        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity descending
        sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]

        # Take top max_results
        top_indices = sorted_indices[:max_results]

        # Build result list
        results = [
            (self.user_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def get_user_count(self) -> int:
        """Get total number of users"""
        return len(self.user_ids)

    def get_user_feature(self, user_id: bytes) -> Optional[np.ndarray]:
        """
        Get feature for a specific user

        Args:
            user_id: 16-byte user ID

        Returns:
            Feature vector or None if not found
        """
        try:
            idx = self.user_ids.index(user_id)
            return self.features[idx].copy()
        except ValueError:
            return None

    def clear(self):
        """Clear all users from database"""
        self.user_ids.clear()
        self.features.clear()

    def get_all_users(self) -> List[bytes]:
        """Get list of all user IDs"""
        return self.user_ids.copy()

    def get_memory_usage(self) -> int:
        """
        Estimate memory usage in bytes

        Returns:
            Estimated memory usage
        """
        # user_ids: 16 bytes each
        # features: 512 * 4 bytes each (float32)
        user_count = len(self.user_ids)
        memory = user_count * (16 + self.feature_dim * 4)
        return memory
