"""
OCFA Face SDK - Configuration Management

This module handles loading and managing SDK configuration from JSON files.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class OCFAConfig:
    """OCFA Face SDK Configuration Manager"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to JSON configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load_from_file(config_path)
        else:
            self._load_default_config()

    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from JSON file

        Args:
            config_path: Path to JSON configuration file

        Raises:
            FileNotFoundError: If config file does not exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.config_path = config_path
        self._validate_config()

    def _load_default_config(self) -> None:
        """Load default configuration"""
        self.config = {
            "version": "1.0.0",
            "platform": "generic",
            "models": {
                "liveness_model": "models/minifasnet_dual_int8.onnx",
                "feature_model_rgb": "models/arcface_r34_rgb_int8.onnx",
                "feature_model_ir": "models/arcface_r34_ir_int8.onnx"
            },
            "thresholds": {
                "liveness": 0.90,
                "quality": 0.50
            },
            "input": {
                "rgb_width": 1280,
                "rgb_height": 720,
                "ir_width": 1280,
                "ir_height": 720,
                "face_size": 112
            },
            "inference": {
                "device": "cpu",
                "num_threads": 2,
                "batch_size": 1
            },
            "preprocessing": {
                "enable_denoise": True,
                "enable_enhancement": True,
                "enable_color_correction": True,
                "enable_histogram_equalization": True
            },
            "fusion": {
                "strategy": "adaptive",
                "rgb_weight_high_light": 0.8,
                "ir_weight_high_light": 0.2,
                "rgb_weight_medium_light": 0.5,
                "ir_weight_medium_light": 0.5,
                "rgb_weight_low_light": 0.2,
                "ir_weight_low_light": 0.8,
                "light_threshold_high": 100,
                "light_threshold_low": 10
            },
            "quality": {
                "blur_threshold": 100.0,
                "brightness_min": 40,
                "brightness_max": 220,
                "pose_angle_threshold": 30
            },
            "logging": {
                "level": "info",
                "save_log": True,
                "log_path": "logs/ocfa_face.log",
                "console_output": True
            },
            "performance": {
                "enable_parallel_inference": True,
                "enable_neon_optimization": False,
                "enable_pipeline": False
            }
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['version', 'models', 'thresholds', 'input', 'inference']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate model paths exist
        for model_key, model_path in self.config['models'].items():
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")

        # Validate thresholds
        if not (0.0 <= self.config['thresholds']['liveness'] <= 1.0):
            raise ValueError("Liveness threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.config['thresholds']['quality'] <= 1.0):
            raise ValueError("Quality threshold must be in [0.0, 1.0]")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., "models.liveness_model")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., "thresholds.liveness")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_to_file(self, output_path: str) -> None:
        """
        Save configuration to JSON file

        Args:
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"OCFAConfig(config_path={self.config_path})"

    def __str__(self) -> str:
        return json.dumps(self.config, indent=2, ensure_ascii=False)

    # Convenience properties
    @property
    def liveness_threshold(self) -> float:
        """Get liveness detection threshold"""
        return self.get('thresholds.liveness', 0.90)

    @property
    def quality_threshold(self) -> float:
        """Get quality assessment threshold"""
        return self.get('thresholds.quality', 0.50)

    @property
    def liveness_model_path(self) -> str:
        """Get liveness model path"""
        return self.get('models.liveness_model', '')

    @property
    def feature_model_rgb_path(self) -> str:
        """Get RGB feature model path"""
        return self.get('models.feature_model_rgb', '')

    @property
    def feature_model_ir_path(self) -> str:
        """Get IR feature model path"""
        return self.get('models.feature_model_ir', '')

    @property
    def input_size(self) -> tuple:
        """Get input image size (width, height)"""
        return (self.get('input.rgb_width', 1280),
                self.get('input.rgb_height', 720))

    @property
    def face_size(self) -> int:
        """Get face crop size"""
        return self.get('input.face_size', 112)

    @property
    def device(self) -> str:
        """Get inference device"""
        return self.get('inference.device', 'cpu')

    @property
    def num_threads(self) -> int:
        """Get number of inference threads"""
        return self.get('inference.num_threads', 2)


def load_config(config_path: Optional[str] = None) -> OCFAConfig:
    """
    Load configuration from file or use default

    Args:
        config_path: Path to JSON configuration file

    Returns:
        OCFAConfig instance
    """
    return OCFAConfig(config_path)
