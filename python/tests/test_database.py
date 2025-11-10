"""
Unit tests for database module
"""

import pytest
import numpy as np
from ocfa.database import FeatureDatabase
from ocfa.utils import generate_user_id, l2_normalize


@pytest.fixture
def empty_database():
    """Create empty database"""
    return FeatureDatabase(feature_dim=512)


@pytest.fixture
def populated_database():
    """Create database with some users"""
    db = FeatureDatabase(feature_dim=512)

    # Add 5 users
    for i in range(5):
        user_id = generate_user_id()
        feature = np.random.randn(512).astype(np.float32)
        feature = l2_normalize(feature, axis=0)
        db.add_user(user_id, feature)

    return db


def test_add_user(empty_database):
    """Test adding users"""
    db = empty_database

    user_id = generate_user_id()
    feature = np.random.randn(512).astype(np.float32)
    feature = l2_normalize(feature, axis=0)

    # Add user
    success = db.add_user(user_id, feature)
    assert success == True
    assert db.get_user_count() == 1

    # Try adding same user again
    success = db.add_user(user_id, feature)
    assert success == False
    assert db.get_user_count() == 1


def test_add_user_invalid_id():
    """Test adding user with invalid ID"""
    db = FeatureDatabase(feature_dim=512)

    with pytest.raises(ValueError):
        db.add_user(b'short_id', np.random.randn(512))


def test_add_user_invalid_feature():
    """Test adding user with invalid feature"""
    db = FeatureDatabase(feature_dim=512)
    user_id = generate_user_id()

    with pytest.raises(ValueError):
        db.add_user(user_id, np.random.randn(256))  # Wrong dimension


def test_update_user(populated_database):
    """Test updating user feature"""
    db = populated_database

    # Get first user
    user_id = db.get_all_users()[0]
    old_feature = db.get_user_feature(user_id)

    # Update with new feature
    new_feature = np.random.randn(512).astype(np.float32)
    new_feature = l2_normalize(new_feature, axis=0)

    success = db.update_user(user_id, new_feature)
    assert success == True

    # Verify update
    updated_feature = db.get_user_feature(user_id)
    assert np.allclose(updated_feature, new_feature)
    assert not np.allclose(updated_feature, old_feature)


def test_update_nonexistent_user(empty_database):
    """Test updating non-existent user"""
    db = empty_database

    user_id = generate_user_id()
    feature = np.random.randn(512).astype(np.float32)

    success = db.update_user(user_id, feature)
    assert success == False


def test_remove_user(populated_database):
    """Test removing user"""
    db = populated_database
    initial_count = db.get_user_count()

    # Remove first user
    user_id = db.get_all_users()[0]
    success = db.remove_user(user_id)

    assert success == True
    assert db.get_user_count() == initial_count - 1
    assert user_id not in db.get_all_users()


def test_remove_nonexistent_user(empty_database):
    """Test removing non-existent user"""
    db = empty_database

    user_id = generate_user_id()
    success = db.remove_user(user_id)
    assert success == False


def test_search_user(populated_database):
    """Test searching for single user"""
    db = populated_database

    # Get a user's feature
    user_id = db.get_all_users()[0]
    feature = db.get_user_feature(user_id)

    # Search with same feature
    matched_id, similarity = db.search_user(feature)

    assert matched_id == user_id
    assert np.isclose(similarity, 1.0, atol=1e-5)


def test_search_user_empty_database(empty_database):
    """Test searching in empty database"""
    db = empty_database

    feature = np.random.randn(512).astype(np.float32)
    matched_id, similarity = db.search_user(feature)

    assert matched_id is None
    assert similarity == 0.0


def test_search_users(populated_database):
    """Test searching for multiple users"""
    db = populated_database

    # Get a user's feature
    user_id = db.get_all_users()[0]
    feature = db.get_user_feature(user_id)

    # Search with threshold
    results = db.search_users(feature, threshold=0.5, max_results=3)

    assert len(results) > 0
    assert results[0][0] == user_id  # First result should be exact match
    assert np.isclose(results[0][1], 1.0, atol=1e-5)

    # Check results are sorted by similarity descending
    similarities = [r[1] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_search_users_with_threshold(populated_database):
    """Test searching with high threshold"""
    db = populated_database

    feature = np.random.randn(512).astype(np.float32)
    feature = l2_normalize(feature, axis=0)

    # High threshold - likely no matches
    results = db.search_users(feature, threshold=0.99, max_results=5)
    assert len(results) <= 5


def test_get_user_feature(populated_database):
    """Test getting user feature"""
    db = populated_database

    user_id = db.get_all_users()[0]
    feature = db.get_user_feature(user_id)

    assert feature is not None
    assert feature.shape == (512,)


def test_get_nonexistent_user_feature(empty_database):
    """Test getting feature of non-existent user"""
    db = empty_database

    user_id = generate_user_id()
    feature = db.get_user_feature(user_id)

    assert feature is None


def test_clear(populated_database):
    """Test clearing database"""
    db = populated_database

    assert db.get_user_count() > 0

    db.clear()

    assert db.get_user_count() == 0
    assert len(db.get_all_users()) == 0


def test_get_all_users(populated_database):
    """Test getting all user IDs"""
    db = populated_database

    all_users = db.get_all_users()

    assert len(all_users) == db.get_user_count()
    assert all(len(uid) == 16 for uid in all_users)


def test_memory_usage(populated_database):
    """Test memory usage estimation"""
    db = populated_database

    memory = db.get_memory_usage()

    # Each user: 16 bytes (ID) + 512 * 4 bytes (feature)
    expected = db.get_user_count() * (16 + 512 * 4)

    assert memory == expected
