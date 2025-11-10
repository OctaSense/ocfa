"""
Unit tests for config module
"""

import pytest
import tempfile
import json
import os
from ocfa.config import OCFAConfig, load_config


def test_default_config():
    """Test default configuration"""
    config = OCFAConfig()

    assert config.get('version') == '1.0.0'
    assert config.liveness_threshold == 0.90
    assert config.quality_threshold == 0.50
    assert config.device == 'cpu'
    assert config.num_threads == 2


def test_load_from_file():
    """Test loading config from JSON file"""
    # Create temporary config file
    config_data = {
        'version': '1.0.0',
        'models': {
            'liveness_model': 'test_liveness.onnx',
            'feature_model_rgb': 'test_rgb.onnx',
            'feature_model_ir': 'test_ir.onnx'
        },
        'thresholds': {
            'liveness': 0.85,
            'quality': 0.60
        },
        'input': {
            'rgb_width': 1280,
            'rgb_height': 720,
            'face_size': 112
        },
        'inference': {
            'device': 'cpu',
            'num_threads': 4
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = OCFAConfig(temp_path)

        assert config.liveness_threshold == 0.85
        assert config.quality_threshold == 0.60
        assert config.num_threads == 4
        assert config.liveness_model_path == 'test_liveness.onnx'

    finally:
        os.unlink(temp_path)


def test_get_nested_value():
    """Test getting nested configuration values"""
    config = OCFAConfig()

    # Test dot notation
    assert config.get('models.liveness_model') is not None
    assert config.get('thresholds.liveness') == 0.90
    assert config.get('input.face_size') == 112

    # Test default value
    assert config.get('nonexistent.key', 'default') == 'default'


def test_set_value():
    """Test setting configuration values"""
    config = OCFAConfig()

    config.set('thresholds.liveness', 0.95)
    assert config.get('thresholds.liveness') == 0.95

    config.set('new.nested.value', 123)
    assert config.get('new.nested.value') == 123


def test_save_to_file():
    """Test saving configuration to file"""
    config = OCFAConfig()
    config.set('test_key', 'test_value')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        config.save_to_file(temp_path)
        assert os.path.exists(temp_path)

        # Load and verify
        with open(temp_path, 'r') as f:
            data = json.load(f)
            assert data['test_key'] == 'test_value'

    finally:
        os.unlink(temp_path)


def test_properties():
    """Test configuration properties"""
    config = OCFAConfig()

    assert isinstance(config.liveness_threshold, float)
    assert isinstance(config.quality_threshold, float)
    assert isinstance(config.input_size, tuple)
    assert isinstance(config.face_size, int)
    assert isinstance(config.device, str)
    assert isinstance(config.num_threads, int)


def test_invalid_config_file():
    """Test handling of invalid config file"""
    with pytest.raises(FileNotFoundError):
        OCFAConfig('nonexistent_config.json')


def test_load_config_function():
    """Test load_config convenience function"""
    config = load_config()
    assert isinstance(config, OCFAConfig)
    assert config.get('version') is not None
