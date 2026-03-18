"""Shared utilities for Franka robot evaluation scripts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from openpi.shared.yaml_config import load_yaml_mapping

# Numerical constants
QUATERNION_NORM_EPSILON: float = 1e-9
THREAD_JOIN_TIMEOUT_S: float = 2.0
IMPEDANCE_STARTUP_DELAY_S: float = 0.02
GRIPPER_GRASP_EPSILON: float = 0.01


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion, falling back to identity if near zero."""
    q = np.asarray(quat)
    norm = np.linalg.norm(q)
    if norm < QUATERNION_NORM_EPSILON:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def align_quaternion_sign(quat: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Align quaternion sign to reference to avoid discontinuities."""
    if reference is not None and float(np.dot(quat, reference)) < 0.0:
        return -quat
    return quat


def quat_to_rotmat(quat: Iterable[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        quat: Quaternion in (w, x, y, z) order.

    Returns:
        3x3 rotation matrix.

    Raises:
        ValueError: If quaternion shape is not (4,).
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm < QUATERNION_NORM_EPSILON:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def get_nested(cfg: dict[str, Any], keys: list[str], default: Any) -> Any:
    """Get nested value from config dict or return default.

    Args:
        cfg: Configuration dictionary.
        keys: List of keys to traverse.
        default: Default value if key path not found.

    Returns:
        Value at key path or default.
    """
    for key in keys:
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config file, raising if missing or invalid."""
    return load_yaml_mapping(path)
