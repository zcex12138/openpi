"""Visualize offline Franka TCP trajectory and action target frames with a slider.

Usage:
    uv run examples/franka/visualize_offline_trajectory.py \
        --records /home/mpi/workspace/yhx/openpi/policy_records
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_records_path(records_path: Path, *, config: str | None, episode: int | None) -> Path:
    if records_path.exists() and records_path.is_file():
        return records_path

    base_dir = records_path if records_path.is_dir() else records_path.parent

    candidate = base_dir / "records.npy"
    if candidate.exists():
        return candidate

    if episode is not None:
        candidate = base_dir / f"records_ep_{episode:04d}.npy"
        if candidate.exists():
            return candidate
    else:
        candidates = sorted(base_dir.glob("records_ep_*.npy"))
        if candidates:
            logger.info("Using split episode file: %s", candidates[0])
            return candidates[0]

    config_dirs: list[Path] = []
    if config is not None:
        candidate_dir = base_dir / config
        if candidate_dir.exists():
            config_dirs = [candidate_dir]
        else:
            logger.warning("Config directory not found: %s", candidate_dir)

    if not config_dirs:
        if list(base_dir.glob("episode_*/records.npy")):
            config_dirs = [base_dir]
        else:
            config_dirs = [p for p in base_dir.iterdir() if p.is_dir()]

    if not config_dirs:
        raise FileNotFoundError(f"No records found under {base_dir}")

    if len(config_dirs) > 1 and config is None:
        logger.info("Multiple configs found, using first: %s", config_dirs[0].name)
    config_dir = config_dirs[0]

    if episode is not None:
        candidate = config_dir / f"episode_{episode:03d}" / "records.npy"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Records file not found: {candidate}")

    candidates = sorted(config_dir.glob("episode_*/records.npy"))
    if candidates:
        logger.info("Using episode records file: %s", candidates[0])
        return candidates[0]

    raise FileNotFoundError(f"No episode records found under {config_dir}")


def _as_float_array(value: Iterable[float], *, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < size:
        raise ValueError(f"Expected at least {size} elements, got {arr.size}")
    return arr[:size]


def _quat_to_rotmat(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size < 4:
        raise ValueError(f"Expected quaternion with 4 elements, got {q.size}")

    if np.any(np.isnan(q[:4])):
        return np.full((3, 3), np.nan, dtype=np.float64)

    q = q[:4]
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)

    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _extract_poses(
    records: np.ndarray, *, max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tcp_positions = []
    tcp_quaternions = []
    target_positions = []
    target_quaternions = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {idx} is not a dict, got {type(record)}")

        state = record.get("inputs/observation/state")
        if state is None:
            raise ValueError(f"Record {idx} missing inputs/observation/state")

        tcp_positions.append(_as_float_array(state, size=3))
        tcp_quaternions.append(_as_float_array(state[3:], size=4))

        actions = record.get("outputs/actions")
        if actions is None:
            target_positions.append(np.full((max_steps, 3), np.nan, dtype=np.float64))
            target_quaternions.append(np.full((max_steps, 4), np.nan, dtype=np.float64))
            continue

        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        step_positions = np.full((max_steps, 3), np.nan, dtype=np.float64)
        step_quats = np.full((max_steps, 4), np.nan, dtype=np.float64)
        for step_idx in range(min(max_steps, actions.shape[0])):
            step = actions[step_idx]
            if step.shape[0] >= 7:
                step_positions[step_idx] = _as_float_array(step, size=3)
                step_quats[step_idx] = _as_float_array(step[3:], size=4)

        target_positions.append(step_positions)
        target_quaternions.append(step_quats)

    return (
        np.stack(tcp_positions, axis=0),
        np.stack(tcp_quaternions, axis=0),
        np.stack(target_positions, axis=0),
        np.stack(target_quaternions, axis=0),
    )


def _compute_bounds(
    tcp_positions: np.ndarray, target_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = [tcp_positions]
    if target_positions.ndim == 3:
        flat_targets = target_positions.reshape(-1, 3)
        target_valid = flat_targets[~np.isnan(flat_targets).any(axis=1)]
    else:
        target_valid = target_positions[~np.isnan(target_positions).any(axis=1)]

    if target_valid.size > 0:
        points.append(target_valid)

    all_points = np.concatenate(points, axis=0)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    margin = 0.05
    return min_vals - margin, max_vals + margin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize offline TCP trajectory and action target frames with a slider."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("policy_records"),
        help="Path to records.npy or records directory (default: policy_records)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to load (0-based). For episode folders, uses episode_###.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置名子目录（例如 pi05_franka_screwdriver_lora）",
    )
    parser.add_argument(
        "--use-workspace",
        action="store_true",
        help="Use workspace bounds from constants instead of data bounds.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Offline Trajectory vs Action Target",
        help="Figure title.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=3,
        help="Number of action steps to visualize as target frames (default: 3).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Coordinate frame axis length in meters (default: 0.08).",
    )
    args = parser.parse_args()

    try:
        records_path = _resolve_records_path(args.records, config=args.config, episode=args.episode)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    records = np.load(records_path, allow_pickle=True)
    if records.size == 0:
        logger.error("Empty records file: %s", args.records)
        sys.exit(1)

    if args.target_steps < 1:
        logger.error("target-steps must be >= 1")
        sys.exit(1)

    tcp_positions, tcp_quats, target_positions, target_quats = _extract_poses(
        records, max_steps=args.target_steps
    )
    n_frames = tcp_positions.shape[0]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    plt.subplots_adjust(bottom=0.18)

    # Bounds
    if args.use_workspace:
        lower, upper = constants.WORKSPACE_BOUNDS
    else:
        lower, upper = _compute_bounds(tcp_positions, target_positions)

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True, alpha=0.3)

    if hasattr(ax, "set_box_aspect"):
        span = upper - lower
        max_span = float(np.max(span))
        if max_span > 0:
            ax.set_box_aspect((span[0] / max_span, span[1] / max_span, span[2] / max_span))

    # Plot full paths (faint)
    ax.plot(
        tcp_positions[:, 0],
        tcp_positions[:, 1],
        tcp_positions[:, 2],
        color="#aaaaaa",
        linewidth=1,
        alpha=0.5,
        label="TCP full",
    )

    primary_targets = target_positions[:, 0, :]
    target_valid = primary_targets[~np.isnan(primary_targets).any(axis=1)]
    if target_valid.size > 0:
        ax.plot(
            target_valid[:, 0],
            target_valid[:, 1],
            target_valid[:, 2],
            color="#fdb462",
            linewidth=1,
            alpha=0.5,
            label="Target step1 full",
        )

    # Dynamic lines and points
    (tcp_line,) = ax.plot([], [], [], color="#377eb8", linewidth=2, label="TCP")
    (target_line,) = ax.plot([], [], [], color="#ff7f0e", linewidth=2, label="Target step1")
    (tcp_point,) = ax.plot([], [], [], "o", color="#377eb8", markersize=5)
    (target_point,) = ax.plot([], [], [], "o", color="#ff7f0e", markersize=5)

    # Coordinate frames
    axis_colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    tcp_frame_lines = []
    for color in axis_colors:
        (line,) = ax.plot([], [], [], color=color, linewidth=2.2, alpha=0.95)
        tcp_frame_lines.append(line)

    target_frame_lines = []
    for step_idx in range(args.target_steps):
        step_lines = []
        alpha = max(0.2, 0.8 - 0.2 * step_idx)
        for color in axis_colors:
            (line,) = ax.plot(
                [],
                [],
                [],
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=alpha,
            )
            step_lines.append(line)
        target_frame_lines.append(step_lines)

    status_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    ax.legend(loc="upper right", fontsize=9)

    slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        "Frame",
        valmin=0,
        valmax=n_frames - 1,
        valinit=0,
        valstep=1,
    )

    def _update_frame_lines(
        lines: list, position: np.ndarray, rotation: np.ndarray, *, axis_length: float
    ) -> None:
        axes = rotation[:, 0:3]
        for axis_vec, line in zip(axes.T, lines):
            end = position + axis_length * axis_vec
            line.set_data([position[0], end[0]], [position[1], end[1]])
            line.set_3d_properties([position[2], end[2]])

    def _clear_frame_lines(lines: list) -> None:
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

    def _update(index: int) -> None:
        idx = int(index)
        tcp_hist = tcp_positions[: idx + 1]
        tcp_line.set_data(tcp_hist[:, 0], tcp_hist[:, 1])
        tcp_line.set_3d_properties(tcp_hist[:, 2])
        tcp_point.set_data([tcp_positions[idx, 0]], [tcp_positions[idx, 1]])
        tcp_point.set_3d_properties([tcp_positions[idx, 2]])

        target_hist = primary_targets[: idx + 1]
        target_mask = ~np.isnan(target_hist).any(axis=1)
        if np.any(target_mask):
            target_hist = target_hist[target_mask]
            target_line.set_data(target_hist[:, 0], target_hist[:, 1])
            target_line.set_3d_properties(target_hist[:, 2])
        else:
            target_line.set_data([], [])
            target_line.set_3d_properties([])

        if not np.isnan(primary_targets[idx]).any():
            target_point.set_data([primary_targets[idx, 0]], [primary_targets[idx, 1]])
            target_point.set_3d_properties([primary_targets[idx, 2]])
        else:
            target_point.set_data([], [])
            target_point.set_3d_properties([])

        # TCP frame
        if not np.isnan(tcp_positions[idx]).any() and not np.isnan(tcp_quats[idx]).any():
            tcp_rot = _quat_to_rotmat(tcp_quats[idx])
            if not np.isnan(tcp_rot).any():
                _update_frame_lines(
                    tcp_frame_lines,
                    tcp_positions[idx],
                    tcp_rot,
                    axis_length=args.axis_length,
                )
            else:
                _clear_frame_lines(tcp_frame_lines)
        else:
            _clear_frame_lines(tcp_frame_lines)

        # Target frames (2-3 steps)
        for step_idx, lines in enumerate(target_frame_lines):
            pose = target_positions[idx, step_idx]
            quat = target_quats[idx, step_idx]
            if np.isnan(pose).any() or np.isnan(quat).any():
                _clear_frame_lines(lines)
                continue
            rot = _quat_to_rotmat(quat)
            if np.isnan(rot).any():
                _clear_frame_lines(lines)
                continue
            _update_frame_lines(lines, pose, rot, axis_length=args.axis_length)

        status_text.set_text(
            f"Frame {idx + 1}/{n_frames} | Targets: {args.target_steps}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    _update(0)

    logger.info("Loaded %d frames from %s", n_frames, args.records)
    plt.show()


if __name__ == "__main__":
    main()
