"""Constants for Franka robot evaluation.

This module provides:
- Fixed constants (ACTION_DIM, STATE_DIM)
- Camera-related constants loaded from camera_config.yaml
- Robot-related constants loaded from real_env_config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.franka.utils import get_nested
from examples.franka.utils import load_yaml_config

# Path to config files
_CAMERA_CONFIG_FILE = Path(__file__).parent / "camera_config.yaml"
_REAL_ENV_CONFIG_FILE = Path(__file__).parent / "real_env_config.yaml"

# Cached config data
_camera_config: dict[str, Any] | None = None
_real_env_config: dict[str, Any] | None = None


def _load_camera_config() -> dict[str, Any]:
    """Load camera configuration from YAML file."""
    global _camera_config
    if _camera_config is None:
        _camera_config = load_yaml_config(_CAMERA_CONFIG_FILE)
    return _camera_config or {}


def _load_real_env_config() -> dict[str, Any]:
    """Load real_env configuration from YAML file."""
    global _real_env_config
    if _real_env_config is None:
        _real_env_config = load_yaml_config(_REAL_ENV_CONFIG_FILE)
    return _real_env_config or {}


def _get_camera_nested(keys: list[str], default: Any) -> Any:
    """Get nested value from camera config or return default."""
    return get_nested(_load_camera_config(), keys, default)


def _get_env_nested(keys: list[str], default: Any) -> Any:
    """Get nested value from real_env config or return default."""
    return get_nested(_load_real_env_config(), keys, default)


def reload_config(config_path: str | Path | None = None) -> None:
    """Reload camera configuration from a specific file or default location.

    Args:
        config_path: Path to config file. If None, uses default location.
    """
    global _camera_config, _CAMERA_CONFIG_FILE
    if config_path is not None:
        _CAMERA_CONFIG_FILE = Path(config_path)
    _camera_config = None
    _load_camera_config()


def reload_real_env_config(config_path: str | Path | None = None) -> None:
    """Reload real_env configuration from a specific file or default location.

    Args:
        config_path: Path to config file. If None, uses default location.
    """
    global _real_env_config, _REAL_ENV_CONFIG_FILE
    if config_path is not None:
        _REAL_ENV_CONFIG_FILE = Path(config_path)
    _real_env_config = None
    _load_real_env_config()


# =============================================================================
# Fixed constants (not from config)
# =============================================================================

# Action dimensions
# action: [x, y, z, qw, qx, qy, qz, gripper]
ACTION_DIM: int = 8

# State dimensions
# state: [TCP pose (7D) + gripper (1D) + wrench (6D)]
STATE_DIM: int = 14


# =============================================================================
# Camera constants (from camera_config.yaml)
# =============================================================================

# Camera service client
CAMERA_HOST: str = _get_camera_nested(["camera_client", "host"], "127.0.0.1")
CAMERA_PORT: int = _get_camera_nested(["camera_client", "port"], 5050)
CAMERA_TIMEOUT_S: float = _get_camera_nested(["camera_client", "timeout_s"], 0.1)

# Image dimensions (must match training)
IMAGE_HEIGHT: int = _get_camera_nested(["image", "height"], 224)
IMAGE_WIDTH: int = _get_camera_nested(["image", "width"], 224)


# =============================================================================
# Robot constants (from real_env_config.yaml)
# =============================================================================

# Robot connection
ROBOT_IP: str = _get_env_nested(["robot", "ip"], "172.16.0.2")

# Workspace bounds (meters)
_DEFAULT_WS_MIN = [0.2, -0.5, 0.0]
_DEFAULT_WS_MAX = [0.8, 0.5, 0.6]
_ws_min = _get_env_nested(["workspace_bounds", "min"], _DEFAULT_WS_MIN)
_ws_max = _get_env_nested(["workspace_bounds", "max"], _DEFAULT_WS_MAX)
WORKSPACE_BOUNDS: tuple[np.ndarray, np.ndarray] = (
    np.array(_ws_min, dtype=np.float32),
    np.array(_ws_max, dtype=np.float32),
)

# End-effector transform (NE_T_EE) - 4x4 row-major matrix flattened to 16 elements
_DEFAULT_EE_TRANSFORM = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]
DEFAULT_EE_TRANSFORM: list[float] = _get_env_nested(["ee_transform"], _DEFAULT_EE_TRANSFORM)

# Evaluation parameters
MAX_EPISODE_TIME: float = _get_env_nested(["evaluation", "max_episode_time"], 30.0)
DEFAULT_PROMPT: str = _get_env_nested(["evaluation", "default_prompt"], "open the can with the screwdriver")
