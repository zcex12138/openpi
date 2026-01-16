"""Real-time visualization of Franka robot 6D force/torque (wrench) data.

This script connects to the Franka robot and displays real-time plots of
the TCP wrench (6D: fx, fy, fz, tx, ty, tz) using matplotlib.

Usage:
    uv run examples/franka/visualize_wrench.py [--robot-ip IP] [--robot-port PORT]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WrenchVisualizer:
    """Real-time visualizer for Franka robot wrench data."""

    # Colors for force (fx, fy, fz) and torque (tx, ty, tz)
    FORCE_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]  # red, green, blue
    TORQUE_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]  # red, green, blue
    FORCE_LABELS = ["Fx", "Fy", "Fz"]
    TORQUE_LABELS = ["Tx", "Ty", "Tz"]

    def __init__(
        self,
        robot_ip: str = constants.ROBOT_IP,
        robot_port: int = constants.ROBOT_PORT,
        window_sec: float = 10.0,
        fps: float = 30.0,
    ) -> None:
        """Initialize the wrench visualizer.

        Args:
            robot_ip: IP address of the robot controller.
            robot_port: Port of the robot controller.
            window_sec: Time window to display in seconds.
            fps: Target update frequency in Hz.
        """
        self._robot_ip = robot_ip
        self._robot_port = robot_port
        self._window_sec = window_sec
        self._fps = fps
        self._dt = 1.0 / fps
        self._max_points = int(window_sec * fps)

        # Data buffers (deques with max length)
        self._time_data: deque[float] = deque(maxlen=self._max_points)
        self._force_data: list[deque[float]] = [deque(maxlen=self._max_points) for _ in range(3)]
        self._torque_data: list[deque[float]] = [deque(maxlen=self._max_points) for _ in range(3)]

        self._start_time: float = 0.0
        self._client = None
        self._ani = None
        self._running = False

        # Initialize plot
        self._fig, (self._ax_force, self._ax_torque) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        self._fig.suptitle("Franka Robot Wrench Visualization", fontsize=14, fontweight="bold")

        # Force lines
        self._force_lines = []
        for i, (color, label) in enumerate(zip(self.FORCE_COLORS, self.FORCE_LABELS)):
            (line,) = self._ax_force.plot([], [], color=color, label=label, linewidth=1.5)
            self._force_lines.append(line)

        # Torque lines
        self._torque_lines = []
        for i, (color, label) in enumerate(zip(self.TORQUE_COLORS, self.TORQUE_LABELS)):
            (line,) = self._ax_torque.plot([], [], color=color, label=label, linewidth=1.5)
            self._torque_lines.append(line)

        # Configure force subplot
        self._ax_force.set_ylabel("Force (N)", fontsize=12)
        self._ax_force.legend(loc="upper right", fontsize=10)
        self._ax_force.grid(True, alpha=0.3)
        self._ax_force.set_xlim(0, window_sec)

        # Configure torque subplot
        self._ax_torque.set_xlabel("Time (s)", fontsize=12)
        self._ax_torque.set_ylabel("Torque (Nm)", fontsize=12)
        self._ax_torque.legend(loc="upper right", fontsize=10)
        self._ax_torque.grid(True, alpha=0.3)
        self._ax_torque.set_xlim(0, window_sec)

        plt.tight_layout()

    def connect(self) -> None:
        """Connect to the Franka robot."""
        from robot_client import RobotClient

        logger.info("Connecting to Franka robot at %s:%s", self._robot_ip, self._robot_port)
        self._client = RobotClient(self._robot_ip, self._robot_port)
        logger.info("Connected to Franka robot")

    def disconnect(self) -> None:
        """Disconnect from the Franka robot."""
        self._client = None
        logger.info("Disconnected from Franka robot")

    def _get_wrench(self) -> np.ndarray | None:
        """Get current wrench data from robot.

        Returns:
            6D wrench [fx, fy, fz, tx, ty, tz] or None if failed.
        """
        if self._client is None:
            return None

        try:
            state = self._client.get_state()
            if state is None:
                return None
            _joint_angles, _position, _quaternion, force, _velocity = state
            return np.asarray(force, dtype=np.float32)
        except Exception as e:
            logger.warning("Failed to get robot state: %s", e)
            return None

    def _init_plot(self):
        """Initialize the plot (called by FuncAnimation)."""
        return self._force_lines + self._torque_lines

    def _update(self, frame: int):
        """Update the plot with new data (called by FuncAnimation)."""
        import time

        current_time = time.time() - self._start_time

        # Get wrench data
        wrench = self._get_wrench()
        if wrench is None:
            return self._force_lines + self._torque_lines

        # Append data
        self._time_data.append(current_time)
        for i in range(3):
            self._force_data[i].append(wrench[i])
            self._torque_data[i].append(wrench[i + 3])

        # Update lines
        time_array = np.array(self._time_data)
        for i, line in enumerate(self._force_lines):
            line.set_data(time_array, np.array(self._force_data[i]))
        for i, line in enumerate(self._torque_lines):
            line.set_data(time_array, np.array(self._torque_data[i]))

        # Update x-axis limits (scrolling window)
        if current_time > self._window_sec:
            x_min = current_time - self._window_sec
            x_max = current_time
        else:
            x_min = 0
            x_max = self._window_sec

        self._ax_force.set_xlim(x_min, x_max)
        self._ax_torque.set_xlim(x_min, x_max)

        # Auto-scale y-axis
        if len(self._time_data) > 1:
            # Force y-limits
            all_force = np.concatenate([np.array(d) for d in self._force_data])
            if len(all_force) > 0:
                f_min, f_max = all_force.min(), all_force.max()
                f_margin = max(0.1 * (f_max - f_min), 1.0)
                self._ax_force.set_ylim(f_min - f_margin, f_max + f_margin)

            # Torque y-limits
            all_torque = np.concatenate([np.array(d) for d in self._torque_data])
            if len(all_torque) > 0:
                t_min, t_max = all_torque.min(), all_torque.max()
                t_margin = max(0.1 * (t_max - t_min), 0.1)
                self._ax_torque.set_ylim(t_min - t_margin, t_max + t_margin)

        return self._force_lines + self._torque_lines

    def run(self) -> None:
        """Start the visualization."""
        import time

        self._running = True
        self._start_time = time.time()

        # Set up signal handler for clean exit
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, stopping...")
            self._running = False
            plt.close(self._fig)

        signal.signal(signal.SIGINT, signal_handler)

        # Create animation
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
        """Clean up resources."""
        if self._ani is not None:
            self._ani.event_source.stop()
        plt.close(self._fig)
        self.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time visualization of Franka robot wrench data."
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=constants.ROBOT_IP,
        help=f"Robot controller IP address (default: {constants.ROBOT_IP})",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=constants.ROBOT_PORT,
        help=f"Robot controller port (default: {constants.ROBOT_PORT})",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Time window to display in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Update frequency in Hz (default: 30.0)",
    )
    args = parser.parse_args()

    visualizer = WrenchVisualizer(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        window_sec=args.window,
        fps=args.fps,
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
