"""Real-time visualization of Franka target and TCP pose frames.

This script connects to the Franka robot controller and visualizes the current
TCP pose and the target pose as coordinate frames in 3D space.

Usage:
    uv run examples/franka/visualize_online_trajectory.py [--robot-ip IP]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from frankx import Affine
from frankx import Robot
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from examples.franka import constants
from examples.franka.utils import quat_to_rotmat as _quat_to_rotmat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PoseFrameVisualizer:
    """Real-time visualizer for TCP pose and target pose."""

    AXIS_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]  # x=red, y=green, z=blue

    def __init__(
        self,
        robot_ip: str = constants.ROBOT_IP,
        *,
        fps: float = 30.0,
        axis_length: float = 0.08,
        workspace_bounds: tuple[np.ndarray, np.ndarray] = constants.WORKSPACE_BOUNDS,
        elev: float = 30.0,
        azim: float = 45.0,
    ) -> None:
        self._robot_ip = robot_ip
        self._fps = fps
        self._dt = 1.0 / fps
        self._axis_length = axis_length
        self._bounds = workspace_bounds
        self._elev = elev
        self._azim = azim

        self._client = None
        self._ani = None
        self._running = False

        self._fig = plt.figure(figsize=(8, 8))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._fig.suptitle("Franka Target vs TCP Pose", fontsize=14, fontweight="bold")

        self._setup_axes()

        # Lines for current TCP frame (solid) and target frame (dashed)
        self._tcp_lines = self._init_frame_lines(linestyle="-", alpha=0.9)
        self._target_lines = self._init_frame_lines(linestyle="--", alpha=0.7)

        # Origin markers
        (self._tcp_origin,) = self._ax.plot([], [], [], "o", color="#222222", markersize=4)
        (self._target_origin,) = self._ax.plot([], [], [], "o", color="#666666", markersize=4)

        # Legend
        legend_items = [
            Line2D([0], [0], color="#222222", lw=2, label="TCP (solid)"),
            Line2D([0], [0], color="#666666", lw=2, linestyle="--", label="Target (dashed)"),
        ]
        self._ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    def _setup_axes(self) -> None:
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.grid(True, alpha=0.3)
        self._ax.view_init(elev=self._elev, azim=self._azim)

        lower, upper = self._bounds
        margin = 0.05
        self._ax.set_xlim(lower[0] - margin, upper[0] + margin)
        self._ax.set_ylim(lower[1] - margin, upper[1] + margin)
        self._ax.set_zlim(lower[2] - margin, upper[2] + margin)

        if hasattr(self._ax, "set_box_aspect"):
            span = upper - lower
            max_span = float(np.max(span))
            if max_span > 0:
                self._ax.set_box_aspect((span[0] / max_span, span[1] / max_span, span[2] / max_span))

    def _init_frame_lines(self, *, linestyle: str, alpha: float) -> list:
        lines = []
        for color in self.AXIS_COLORS:
            (line,) = self._ax.plot(
                [],
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
            )
            lines.append(line)
        return lines

    def connect(self) -> None:
        """Connect to the Franka robot."""
        logger.info("Connecting to Franka robot at %s", self._robot_ip)
        self._client = Robot(self._robot_ip)
        self._client.set_default_behavior()
        self._client.recover_from_errors()
        self._client.set_EE(constants.DEFAULT_EE_TRANSFORM)
        logger.info("Connected to Franka robot")

    def disconnect(self) -> None:
        """Disconnect from the Franka robot."""
        self._client = None
        logger.info("Disconnected from Franka robot")

    def _get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._client is None:
            return None

        try:
            state = self._client.get_state()
            affine = Affine(state.O_T_EE)
            return (
                np.asarray(affine.translation(), dtype=np.float64),
                np.asarray(affine.quaternion(), dtype=np.float64),
            )
        except Exception as e:
            logger.warning("Failed to get TCP pose: %s", e)
            return None

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._client is None:
            return None

        try:
            state = self._client.get_state()
            affine = Affine(state.O_T_EE_d)
            return (
                np.asarray(affine.translation(), dtype=np.float64),
                np.asarray(affine.quaternion(), dtype=np.float64),
            )
        except Exception as e:
            logger.warning("Failed to get target pose: %s", e)
            return None

    def _update_frame(self, lines: list, origin_marker, position: np.ndarray, rotation: np.ndarray) -> None:
        axes = rotation[:, 0:3]
        for axis_vec, line in zip(axes.T, lines):
            end = position + self._axis_length * axis_vec
            line.set_data([position[0], end[0]], [position[1], end[1]])
            line.set_3d_properties([position[2], end[2]])

        origin_marker.set_data([position[0]], [position[1]])
        origin_marker.set_3d_properties([position[2]])

    def _init_plot(self):
        return self._tcp_lines + self._target_lines + [self._tcp_origin, self._target_origin]

    def _update(self, frame: int):
        tcp_pose = self._get_tcp_pose()
        target_pose = self._get_target_pose()

        if tcp_pose is not None:
            tcp_pos, tcp_quat = tcp_pose
            tcp_rot = _quat_to_rotmat(tcp_quat)
            self._update_frame(self._tcp_lines, self._tcp_origin, tcp_pos, tcp_rot)

        if target_pose is not None:
            target_pos, target_quat = target_pose
            target_rot = _quat_to_rotmat(target_quat)
            self._update_frame(self._target_lines, self._target_origin, target_pos, target_rot)

        return self._tcp_lines + self._target_lines + [self._tcp_origin, self._target_origin]

    def run(self) -> None:
        """Start the visualization loop."""
        self._running = True

        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, stopping...")
            self._running = False
            plt.close(self._fig)

        signal.signal(signal.SIGINT, signal_handler)

        interval_ms = int(self._dt * 1000)
        self._ani = FuncAnimation(
            self._fig,
            self._update,
            init_func=self._init_plot,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        logger.info("Starting visualization (press Ctrl+C to stop)...")
        plt.show()

    def close(self) -> None:
        if self._ani is not None:
            self._ani.event_source.stop()
        plt.close(self._fig)
        self.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time visualization of Franka impedance target and TCP pose frames."
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=constants.ROBOT_IP,
        help=f"Robot controller IP address (default: {constants.ROBOT_IP})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Update frequency in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Coordinate frame axis length in meters (default: 0.08)",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Camera elevation angle for 3D view (default: 30.0)",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=45.0,
        help="Camera azimuth angle for 3D view (default: 45.0)",
    )

    args = parser.parse_args()

    visualizer = PoseFrameVisualizer(
        robot_ip=args.robot_ip,
        fps=args.fps,
        axis_length=args.axis_length,
        elev=args.elev,
        azim=args.azim,
    )

    try:
        visualizer.connect()
        visualizer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
