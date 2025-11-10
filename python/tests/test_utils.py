"""
Unit tests for utils module
"""

import pytest
import numpy as np
from ocfa.utils import (
    l2_normalize,
    cosine_similarity,
    batch_cosine_similarity,
    bgr_to_rgb,
    hwc_to_chw,
    chw_to_hwc,
    compute_iou,
    compute_brightness,
    Timer,
    bytes_to_user_id,
    user_id_to_bytes,
    generate_user_id,
    validate_image_shape,
    softmax
)


def test_l2_normalize():
    """Test L2 normalization"""
    # Single vector
    vec = np.array([3.0, 4.0])
    normalized = l2_normalize(vec, axis=0)
    assert np.allclose(np.linalg.norm(normalized), 1.0)

    # Matrix (multiple vectors)
    mat = np.array([[3.0, 4.0], [1.0, 1.0]])
    normalized = l2_normalize(mat, axis=1)
    for i in range(len(normalized)):
        assert np.allclose(np.linalg.norm(normalized[i]), 1.0)


def test_cosine_similarity():
    """Test cosine similarity"""
    # Identical vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    sim = cosine_similarity(vec1, vec2)
    assert np.isclose(sim, 1.0)

    # Orthogonal vectors
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert np.isclose(sim, 0.0)


def test_batch_cosine_similarity():
    """Test batch cosine similarity"""
    query = np.array([1.0, 0.0, 0.0])
    features = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ])

    sims = batch_cosine_similarity(query, features)
    assert len(sims) == 3
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)


def test_bgr_to_rgb():
    """Test BGR to RGB conversion"""
    bgr = np.array([[[0, 1, 2]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    assert np.array_equal(rgb, np.array([[[2, 1, 0]]], dtype=np.uint8))


def test_hwc_to_chw():
    """Test HWC to CHW conversion"""
    hwc = np.zeros((112, 112, 3))
    chw = hwc_to_chw(hwc)
    assert chw.shape == (3, 112, 112)


def test_chw_to_hwc():
    """Test CHW to HWC conversion"""
    chw = np.zeros((3, 112, 112))
    hwc = chw_to_hwc(chw)
    assert hwc.shape == (112, 112, 3)


def test_compute_iou():
    """Test IoU computation"""
    # Identical boxes
    box1 = (0, 0, 10, 10)
    box2 = (0, 0, 10, 10)
    iou = compute_iou(box1, box2)
    assert np.isclose(iou, 1.0)

    # Non-overlapping boxes
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)
    iou = compute_iou(box1, box2)
    assert np.isclose(iou, 0.0)

    # Partially overlapping boxes
    box1 = (0, 0, 10, 10)
    box2 = (5, 5, 15, 15)
    iou = compute_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_compute_brightness():
    """Test brightness computation"""
    # Black image
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    brightness = compute_brightness(black)
    assert np.isclose(brightness, 0.0)

    # White image
    white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    brightness = compute_brightness(white)
    assert np.isclose(brightness, 255.0)


def test_timer():
    """Test Timer context manager"""
    import time

    with Timer() as timer:
        time.sleep(0.01)

    elapsed = timer.get_elapsed_ms()
    assert elapsed >= 10.0  # At least 10ms


def test_user_id_conversion():
    """Test user ID conversion"""
    # Generate random ID
    user_id_bytes = generate_user_id()
    assert len(user_id_bytes) == 16

    # Convert to string
    user_id_str = bytes_to_user_id(user_id_bytes)
    assert len(user_id_str) == 32
    assert all(c in '0123456789abcdef' for c in user_id_str)

    # Convert back
    converted_bytes = user_id_to_bytes(user_id_str)
    assert converted_bytes == user_id_bytes


def test_user_id_to_bytes_invalid():
    """Test invalid user ID string"""
    with pytest.raises(ValueError):
        user_id_to_bytes('invalid')

    with pytest.raises(ValueError):
        user_id_to_bytes('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')


def test_validate_image_shape():
    """Test image shape validation"""
    # Valid RGB image
    rgb = np.zeros((112, 112, 3), dtype=np.uint8)
    validate_image_shape(rgb, expected_channels=3)

    # Valid grayscale image
    gray = np.zeros((112, 112), dtype=np.uint8)
    validate_image_shape(gray, expected_channels=1)

    # Invalid shape
    with pytest.raises(ValueError):
        invalid = np.zeros((112, 112, 3, 1))
        validate_image_shape(invalid)


def test_softmax():
    """Test softmax activation"""
    # Simple test
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)

    # Check sum to 1
    assert np.isclose(np.sum(result), 1.0)

    # Check all positive
    assert np.all(result > 0)

    # Check largest input has largest output
    assert np.argmax(result) == np.argmax(x)
