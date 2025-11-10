"""
OCFA Face SDK - Python Package

High-performance face recognition SDK with RGB-IR liveness detection,
feature extraction, and face matching capabilities.
"""

__version__ = '1.0.0'
__author__ = 'OCTA Team'
__license__ = 'MIT'

# Core modules
from .config import OCFAConfig, load_config
from .utils import (
    l2_normalize,
    cosine_similarity,
    batch_cosine_similarity,
    Timer,
    generate_user_id,
    bytes_to_user_id,
    user_id_to_bytes
)

# Submodules will be imported lazily to avoid circular dependencies
# and reduce initial import time

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',

    # Configuration
    'OCFAConfig',
    'load_config',

    # Utilities
    'l2_normalize',
    'cosine_similarity',
    'batch_cosine_similarity',
    'Timer',
    'generate_user_id',
    'bytes_to_user_id',
    'user_id_to_bytes',
]


def get_version() -> str:
    """Get SDK version"""
    return __version__


def print_info():
    """Print SDK information"""
    print(f"OCFA Face SDK v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nFeatures:")
    print("  - RGB-IR Liveness Detection")
    print("  - Face Feature Extraction (ArcFace-R34)")
    print("  - Face Matching (1:1 and 1:N)")
    print("  - Quality Assessment")
    print("  - Adaptive Feature Fusion")
    print("\nTarget Platform: Hi3516CV610 (ARM A17, NNIE 3.0)")
