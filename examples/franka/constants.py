"""Constants for Franka robot evaluation.

Values can be loaded from camera_config.yaml if available, otherwise defaults are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Path to default config file
_CONFIG_FILE = Path(__file__).parent / "camera_config.yaml"

# Cached config data
_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    global _config
    if _config is None:
        if _CONFIG_FILE.exists():
            with _CONFIG_FILE.open("r", encoding="utf-8") as f:
                _config = yaml.safe_load(f) or {}
        else:
            _config = {}
    return _config or {}


def _get_nested(keys: list[str], default: Any) -> Any:
    """Get nested value from config or return default."""
    cfg = _load_config()
    for key in keys:
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg


def reload_config(config_path: str | Path | None = None) -> None:
    """Reload configuration from a specific file or default location.

    Args:
        config_path: Path to config file. If None, uses default location.
    """
    global _config, _CONFIG_FILE
    if config_path is not None:
        _CONFIG_FILE = Path(config_path)
    _config = None
    _load_config()


# Robot connection
ROBOT_IP: str = _get_nested(["robot_server", "ip"], "127.0.0.1")
ROBOT_PORT: int = _get_nested(["robot_server", "port"], 8888)

# Control parameters
CONTROL_FPS: float = _get_nested(["recording", "fps"], 30.0)
DT: float = 1.0 / CONTROL_FPS

# Action dimensions
# action: [x, y, z, qw, qx, qy, qz, gripper]
ACTION_DIM: int = 8
STATE_DIM: int = 14  # TCP pose (7) + additional state (7)

# Workspace bounds (m)
_ws_min = _get_nested(["evaluation", "workspace_bounds", "min"], [0.2, -0.5, 0.0])
_ws_max = _get_nested(["evaluation", "workspace_bounds", "max"], [0.8, 0.5, 0.6])
WORKSPACE_BOUNDS: tuple[np.ndarray, np.ndarray] = (
    np.array(_ws_min, dtype=np.float32),
    np.array(_ws_max, dtype=np.float32),
)

# Maximum TCP speed (m/s)
MAX_POS_SPEED: float = _get_nested(["evaluation", "max_pos_speed"], 0.5)

# Gripper
GRIPPER_OPEN: float = _get_nested(["evaluation", "gripper_open"], 1.0)
GRIPPER_CLOSE: float = _get_nested(["evaluation", "gripper_close"], 0.0)

# Camera service
CAMERA_HOST: str = _get_nested(["evaluation", "camera_host"], "127.0.0.1")
CAMERA_PORT: int = _get_nested(["evaluation", "camera_port"], 5050)
CAMERA_TIMEOUT_S: float = _get_nested(["evaluation", "camera_timeout_s"], 0.1)

# Image dimensions (must match training)
IMAGE_HEIGHT: int = _get_nested(["evaluation", "image_height"], 224)
IMAGE_WIDTH: int = _get_nested(["evaluation", "image_width"], 224)

# Episode defaults
MAX_EPISODE_TIME: float = _get_nested(["evaluation", "max_episode_time"], 30.0)
NUM_EPISODES: int = _get_nested(["evaluation", "num_episodes"], 10)

# Default prompt
DEFAULT_PROMPT: str = _get_nested(["evaluation", "default_prompt"], "open the can with the screwdriver")

# Robot reset parameters
# Default joint position for reset (7 joint angles in radians)
_default_joint = _get_nested(
    ["evaluation", "default_joint_position"],
    [-0.26134401, 0.46399827, -0.02856101, -2.23260865, -0.00302741, 2.67803179, 0.5054156],
)
DEFAULT_JOINT_POSITION: np.ndarray = np.array(_default_joint, dtype=np.float64)

# Move speed factor for joint motion (0.0-1.0)
DEFAULT_MOVE_SPEED_FACTOR: float = _get_nested(["evaluation", "move_speed_factor"], 0.3)

# Gripper parameters for grasping
GRIPPER_OPEN_WIDTH: float = _get_nested(["evaluation", "gripper_open_width"], 0.078)  # meters
GRIPPER_GRASP_WIDTH: float = _get_nested(["evaluation", "gripper_grasp_width"], 0.015)  # meters
GRIPPER_VELOCITY: float = _get_nested(["evaluation", "gripper_velocity"], 0.1)  # m/s
GRIPPER_FORCE: float = _get_nested(["evaluation", "gripper_force"], 30.0)  # Newtons
