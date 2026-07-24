"""Configuration management for Acoustic Phase Optimizer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default_config.yaml"


class Config:
    """Hierarchical configuration with YAML file support."""

    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = self._defaults()
        if config_path:
            self.load(config_path)
        elif DEFAULT_CONFIG_PATH.exists():
            self.load(str(DEFAULT_CONFIG_PATH))

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "system": {
                "sample_rate": 48000,
                "bit_depth": 24,
                "buffer_size": 1024,
                "asio_driver": None,
            },
            "measurement": {
                "sweep_type": "log",
                "sweep_duration": 5.0,
                "sweep_start_freq": 20.0,
                "sweep_end_freq": 20000.0,
                "mls_order": 15,
                "mls_sequence_length": 32767,
                "averaging": 3,
                "pre_delay": 0.1,
                "post_delay": 0.2,
                "output_level_db": -12.0,
            },
            "acoustic": {
                "speed_of_sound": 343.0,
                "max_reflection_order": 3,
                "absorption_model": "eyring",
                "temperature": 20.0,
                "humidity": 50.0,
            },
            "weather": {
                "location": "",
                "latitude": 0.0,
                "longitude": 0.0,
                "enabled": False,
            },
            "optimization": {
                "default_algorithm": "genetic",
                "max_iterations": 1000,
                "population_size": 100,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "initial_temperature": 100.0,
                "cooling_rate": 0.95,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-6,
                "objective_weights": {
                    "phase_coherence": 1.0,
                    "magnitude_flatness": 0.5,
                    "destructive_interference": 1.5,
                    "delay_alignment": 1.0,
                    "rt60_deviation": 0.3,
                },
            },
            "filters": {
                "fir_length": 512,
                "iir_order": 4,
                "crossover_slope_db_per_octave": 24,
                "eq_bands": 31,
                "min_phase_fir": True,
            },
            "visualization": {
                "fft_size": 16384,
                "colormap": "inferno",
                "resolution_3d": 64,
                "refresh_rate_hz": 10,
            },
            "hardware": {
                "mixer": {
                    "type": "allen_heath_qu16",
                    "channels": 16,
                    "control_protocol": "tcp",
                    "address": "192.168.1.100",
                    "port": 51325,
                },
                "dsp": {
                    "type": "dbx_venus360",
                    "address": "192.168.1.101",
                    "control_port": 23,
                },
                "microphone": {
                    "type": "usb_measurement",
                    "calibration_file": None,
                    "sensitivity_db": 0.0,
                },
            },
            "zones": {
                "left_main": {"enabled": True, "x": -10.0, "y": 0.0, "z": 2.0},
                "right_main": {"enabled": True, "x": 10.0, "y": 0.0, "z": 2.0},
                "center": {"enabled": False, "x": 0.0, "y": -1.0, "z": 1.5},
                "subwoofer": {"enabled": True, "x": 0.0, "y": 2.0, "z": 0.0},
                "front_fill": {"enabled": False, "x": 0.0, "y": 8.0, "z": 0.3},
                "delay_left": {"enabled": False, "x": -15.0, "y": 20.0, "z": 4.0},
                "delay_right": {"enabled": False, "x": 15.0, "y": 20.0, "z": 4.0},
            },
        }

    def load(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Config file not found: {path}")
            return
        with open(path_obj, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            self._deep_merge(self._data, user_config)
        logger.info(f"Loaded configuration from {path}")

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {path}")

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        current = self._data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def set(self, *keys_and_value: Any) -> None:
        if len(keys_and_value) < 2:
            raise ValueError("Need at least one key and a value")
        *keys, value = keys_and_value
        current = self._data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @data.setter
    def data(self, value: Dict[str, Any]) -> None:
        self._data = value

    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        config = cls()
        config._data = data
        return config
