"""
OCFA Face SDK - Utility Functions

This module provides common utility functions for the SDK.
"""

import numpy as np
import time
from typing import Tuple, Optional
from functools import wraps


def l2_normalize(features: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    """
    L2 normalize feature vectors

    Args:
        features: Feature array, shape (N, D) or (D,)
        axis: Normalization axis
        eps: Small epsilon to avoid division by zero

    Returns:
        L2 normalized features
    """
    norm = np.linalg.norm(features, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return features / norm


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors
    Assumes features are already L2 normalized

    Args:
        feat1: First feature vector (512,)
        feat2: Second feature vector (512,)

    Returns:
        Cosine similarity in [0, 1]
    """
    # If already L2 normalized, cosine similarity = dot product
    similarity = np.dot(feat1, feat2)
    return float(np.clip(similarity, 0.0, 1.0))


def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two feature vectors

    Args:
        feat1: First feature vector
        feat2: Second feature vector

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(feat1 - feat2))


def batch_cosine_similarity(query: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple features

    Args:
        query: Query feature vector (512,)
        features: Feature matrix (N, 512)

    Returns:
        Similarity array (N,)
    """
    # Assumes both are L2 normalized
    similarities = np.dot(features, query)
    return np.clip(similarities, 0.0, 1.0)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB

    Args:
        image: BGR image array

    Returns:
        RGB image array
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image[:, :, ::-1]
    return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR

    Args:
        image: RGB image array

    Returns:
        BGR image array
    """
    return bgr_to_rgb(image)  # Same operation


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert HWC format to CHW format

    Args:
        image: Image in HWC format (H, W, C)

    Returns:
        Image in CHW format (C, H, W)
    """
    return np.transpose(image, (2, 0, 1))


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """
    Convert CHW format to HWC format

    Args:
        image: Image in CHW format (C, H, W)

    Returns:
        Image in HWC format (H, W, C)
    """
    return np.transpose(image, (1, 2, 0))


def compute_iou(box1: Tuple[int, int, int, int],
                 box2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU (Intersection over Union) between two boxes

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU value in [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def resize_image(image: np.ndarray,
                 target_size: Tuple[int, int],
                 keep_aspect: bool = False) -> np.ndarray:
    """
    Resize image to target size

    Args:
        image: Input image
        target_size: (width, height)
        keep_aspect: Whether to keep aspect ratio

    Returns:
        Resized image
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for image resizing")

    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return result
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute average brightness of image

    Args:
        image: Input image (grayscale or RGB)

    Returns:
        Average brightness value [0, 255]
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = np.mean(image, axis=2)
    else:
        gray = image

    return float(np.mean(gray))


def estimate_illumination(image: np.ndarray) -> int:
    """
    Estimate illumination level in lux (simplified)

    Args:
        image: RGB image

    Returns:
        Estimated illumination in lux
    """
    brightness = compute_brightness(image)

    # Simple mapping from brightness to lux (empirical)
    # Low: 0-50 -> 0-10 lux
    # Medium: 50-150 -> 10-100 lux
    # High: 150-255 -> 100-1000 lux
    if brightness < 50:
        lux = int(brightness * 0.2)
    elif brightness < 150:
        lux = int(10 + (brightness - 50) * 0.9)
    else:
        lux = int(100 + (brightness - 150) * 8.5)

    return lux


class Timer:
    """Simple timer for performance measurement"""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.time() - self.start_time) * 1000  # Convert to ms

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed


def timeit(func):
    """
    Decorator to measure function execution time

    Usage:
        @timeit
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        print(f"{func.__name__} took {elapsed:.2f} ms")
        return result
    return wrapper


def bytes_to_user_id(user_id_bytes: bytes) -> str:
    """
    Convert 16-byte user ID to hex string

    Args:
        user_id_bytes: 16 bytes user ID

    Returns:
        Hex string (32 characters)
    """
    return user_id_bytes.hex()


def user_id_to_bytes(user_id_str: str) -> bytes:
    """
    Convert hex string user ID to 16 bytes

    Args:
        user_id_str: Hex string (32 characters)

    Returns:
        16 bytes user ID

    Raises:
        ValueError: If string is not valid hex or not 32 characters
    """
    if len(user_id_str) != 32:
        raise ValueError(f"User ID must be 32 hex characters, got {len(user_id_str)}")

    try:
        return bytes.fromhex(user_id_str)
    except ValueError as e:
        raise ValueError(f"Invalid hex string: {e}")


def generate_user_id() -> bytes:
    """
    Generate random 16-byte user ID

    Returns:
        Random 16 bytes
    """
    import uuid
    return uuid.uuid4().bytes


def validate_image_shape(image: np.ndarray,
                         expected_channels: Optional[int] = None) -> None:
    """
    Validate image shape

    Args:
        image: Input image array
        expected_channels: Expected number of channels (None to skip check)

    Raises:
        ValueError: If image shape is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy array")

    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")

    if expected_channels is not None:
        if len(image.shape) == 2 and expected_channels != 1:
            raise ValueError(f"Expected {expected_channels} channels, got grayscale")
        elif len(image.shape) == 3 and image.shape[2] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {image.shape[2]}")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax activation

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax output
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
