"""Low-level Franka robot environment for state collection and action execution."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from examples.franka import constants
from examples.franka.gripper_interpolator import GripperStateInterpolator

if TYPE_CHECKING:
    from robot_client import RobotClient


logger = logging.getLogger(__name__)


class FrankaRealEnv:
    """Low-level Franka robot environment (robot state + actions).

    Handles:
    - Robot communication via RobotClient
    - Robot state collection
    - Action execution with safety checks (workspace clipping, velocity limiting)

    State format: 14D [TCP pose (7D: x, y, z, qw, qx, qy, qz)
    + gripper state (1D: 0=open, 1=closed)
    + TCP wrench (6D: fx, fy, fz, tx, ty, tz)]
    Action format: 8D [position (3D: x, y, z) + quaternion (4D: qw, qx, qy, qz) + gripper (1D)]
    """

    def __init__(
        self,
        robot_ip: str = constants.ROBOT_IP,
        robot_port: int = constants.ROBOT_PORT,
        *,
        control_fps: float = constants.CONTROL_FPS,
        workspace_bounds: tuple[np.ndarray, np.ndarray] = constants.WORKSPACE_BOUNDS,
        max_pos_speed: float = constants.MAX_POS_SPEED,
        gripper_interpolation_duration: float = 1.4,
        gripper_command_interval_s: float = 1.5,
    ) -> None:
        self._robot_ip = robot_ip
        self._robot_port = robot_port
        self._control_fps = control_fps
        self._workspace_bounds = workspace_bounds
        self._max_pos_speed = max_pos_speed
        self._dt = 1.0 / control_fps
        self._gripper_command_interval_s = gripper_command_interval_s

        self._client: RobotClient | None = None
        self._last_state: np.ndarray | None = None
        self._last_action_time: float = 0.0
        self._last_gripper_command_time: float | None = None
        self._last_gripper_target: float | None = None  # 0.0=open, 1.0=closed
        self._gripper_interpolator = GripperStateInterpolator(
            interpolation_duration=gripper_interpolation_duration
        )

    def connect(self) -> None:
        """Connect to the Franka robot controller."""
        if self._client is not None:
            logger.warning("Already connected to robot")
            return

        from robot_client import RobotClient

        logger.info("Connecting to Franka robot at %s:%s", self._robot_ip, self._robot_port)
        self._client = RobotClient(self._robot_ip, self._robot_port)
        logger.info("Connected to Franka robot")

    def disconnect(self) -> None:
        """Disconnect from the Franka robot controller."""
        if self._client is not None:
            try:
                # Ensure the robot returns to a safe reset state before disabling impedance control.
                self.reset(grasp=False)
            except Exception as e:
                logger.warning("Error resetting robot before disconnect: %s", e)
            try:
                self._client.stop_impedance_control()
            except Exception as e:
                logger.warning("Error stopping impedance control during disconnect: %s", e)
            finally:
                self._client = None
        logger.info("Disconnected from Franka robot")

    def get_state(self) -> np.ndarray:
        """Get current robot state.

        Returns:
            14D state: [TCP pose (7D: x, y, z, qw, qx, qy, qz)
            + gripper state (1D: 0=open, 1=closed)
            + TCP wrench (6D: fx, fy, fz, tx, ty, tz)]
        """
        if self._client is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        state = self._client.get_state()
        if state is None:
            raise RuntimeError("Failed to read robot state from controller")

        _joint_angles, position, quaternion, force, _velocity = state
        gripper_state = self._get_interpolated_gripper_state()

        state_vec = np.concatenate([position, quaternion, [gripper_state], force]).astype(np.float32)
        self._last_state = state_vec
        return self._last_state

    def execute_action(self, action: np.ndarray) -> None:
        """Execute action on robot with safety checks.

        Args:
            action: 8D action [x, y, z, qw, qx, qy, qz, gripper]
        """
        if self._client is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (constants.ACTION_DIM,):
            raise ValueError(f"Expected action shape ({constants.ACTION_DIM},), got {action.shape}")

        # Extract position and gripper
        position = action[:3]
        quaternion = action[3:7]
        gripper = action[7]

        # Apply workspace clipping
        position_clipped = self._clip_to_workspace(position)
        if not np.allclose(position, position_clipped, atol=1e-4):
            logger.debug("Action clipped: %s -> %s", position, position_clipped)
        position = position_clipped

        # Apply velocity limiting
        if self._last_state is not None:
            current_pos = self._last_state[:3]
            position = self._limit_velocity(current_pos, position)

        # Send pose to robot (franka_control expects position + quaternion)
        self._client.send_pose(position.tolist(), quaternion.tolist(), verify=False)
        # Send gripper command separately (rate-limited + busy-aware)
        self._maybe_send_gripper_command(float(gripper))
        self._last_action_time = time.time()

    def _get_interpolated_gripper_state(self) -> float:
        """Return interpolated gripper state without reading width/force."""
        current_time = time.time()

        if self._client is not None:
            try:
                is_moving = self._client.gripper.is_moving()
                if is_moving is False and self._gripper_interpolator.is_interpolating:
                    self._gripper_interpolator.mark_early_termination()
            except Exception:
                pass

        return float(self._gripper_interpolator.get_state(current_time))

    def _compute_gripper_target(self, gripper_cmd: float) -> float:
        """Map action gripper scalar to target state (0=open, 1=closed)."""
        threshold = (constants.GRIPPER_OPEN + constants.GRIPPER_CLOSE) / 2.0
        # GRIPPER_OPEN is typically 1.0 and GRIPPER_CLOSE is 0.0.
        return 0.0 if gripper_cmd >= threshold else 1.0

    def _maybe_send_gripper_command(self, gripper_cmd: float) -> None:
        """Send gripper open/close command if allowed by busy/interval checks."""
        if self._client is None:
            return

        target_state = self._compute_gripper_target(gripper_cmd)
        if self._last_gripper_target is not None and target_state == self._last_gripper_target:
            return

        current_time = time.time()
        if self._last_gripper_command_time is not None:
            if current_time - self._last_gripper_command_time < self._gripper_command_interval_s:
                return

        try:
            is_moving = self._client.gripper.is_moving()
            if is_moving is True:
                return
        except Exception:
            pass

        if target_state == 0.0:
            success = self._client.gripper.move_async(
                constants.GRIPPER_OPEN_WIDTH,
                constants.GRIPPER_VELOCITY,
            )
        else:
            success = self._client.gripper.grasp_async(
                constants.GRIPPER_GRASP_WIDTH,
                constants.GRIPPER_VELOCITY,
                constants.GRIPPER_FORCE,
                epsilon_inner=0.01,
                epsilon_outer=0.01,
            )

        if success:
            self._last_gripper_command_time = current_time
            self._last_gripper_target = target_state
            self._gripper_interpolator.set_target(target_state, current_time)

    def _clip_to_workspace(self, position: np.ndarray) -> np.ndarray:
        """Clip position to workspace bounds."""
        lower, upper = self._workspace_bounds
        return np.clip(position, lower, upper)

    def _limit_velocity(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Limit velocity to max_pos_speed."""
        displacement = target_pos - current_pos
        distance = np.linalg.norm(displacement)
        max_distance = self._max_pos_speed * self._dt

        if distance > max_distance:
            direction = displacement / distance
            return current_pos + direction * max_distance
        return target_pos

    def reset(self, grasp: bool = True) -> None:
        """Reset robot to default position and optionally grasp the screwdriver.

        Steps:
        1. Stop impedance control (if running)
        2. Open gripper before moving joints
        3. Move robot to default joint position
        4. Close gripper to grasp screwdriver (if grasp=True)
        5. Start impedance control for Cartesian control
        """
        if self._client is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        logger.info("Resetting robot...")

        # Step 1: Stop impedance control if running
        logger.info("Stopping impedance control...")
        try:
            self._client.stop_impedance_control()
            time.sleep(0.5)
        except Exception as e:
            logger.warning("Error stopping impedance control: %s", e)

        # Step 2: Open gripper before moving joints
        logger.info("Opening gripper before moving joints...")
        try:
            success = self._client.gripper.move_async(
                constants.GRIPPER_OPEN_WIDTH,
                constants.GRIPPER_VELOCITY,
            )
            if not success:
                logger.warning("Failed to open gripper")
            time.sleep(2.0)  # Wait for gripper to finish opening
            if success:
                current_time = time.time()
                self._last_gripper_command_time = current_time
                self._last_gripper_target = 0.0
                self._gripper_interpolator.set_target(0.0, current_time)
                self._gripper_interpolator.mark_early_termination()
        except Exception as e:
            logger.warning("Error opening gripper: %s", e)

        # Step 3: Move to default joint position
        logger.info("Moving robot to default joint position...")
        try:
            success = self._client.move_joint(
                constants.DEFAULT_JOINT_POSITION.tolist(),
                speed_factor=constants.DEFAULT_MOVE_SPEED_FACTOR,
            )
            if not success:
                logger.warning("Failed to move robot to default position")
        except Exception as e:
            logger.warning("Error moving to default position: %s", e)

        # Step 4: Close gripper if requested
        if grasp:
            logger.info("Closing gripper to grasp screwdriver...")
            try:
                success = self._client.gripper.grasp_async(
                    constants.GRIPPER_GRASP_WIDTH,
                    constants.GRIPPER_VELOCITY,
                    constants.GRIPPER_FORCE,
                    epsilon_inner=0.01,
                    epsilon_outer=0.01,
                )
                if not success:
                    logger.warning("Failed to close gripper")
                time.sleep(2.0)  # Wait for gripper to finish grasping
                if success:
                    current_time = time.time()
                    self._last_gripper_command_time = current_time
                    self._last_gripper_target = 1.0
                    self._gripper_interpolator.set_target(1.0, current_time)
                    self._gripper_interpolator.mark_early_termination()
            except Exception as e:
                logger.warning("Error closing gripper: %s", e)

        # Step 5: Start impedance control for Cartesian control
        logger.info("Starting impedance control...")
        try:
            if not self._client.start_impedance_control(translational_stiffness=800.0,
                                                       rotational_stiffness=60.0,
                                                       translational_damping_ratio=0.9,
                                                       rotational_damping_ratio=0.9):
                logger.warning("Failed to start impedance control")
            time.sleep(1.0)  # Wait for impedance control to stabilize

            # Initialize impedance target at current position
            state = self._client.get_state()
            if state is not None:
                joint_angles, position, quaternion, force, velocity = state
                self._client.send_pose(position, quaternion, verify=False)
                time.sleep(0.5)
        except Exception as e:
            logger.warning("Error starting impedance control: %s", e)

        # Clear last state to avoid velocity limiting issues
        self._last_state = None

        logger.info("Robot reset complete")

    def __enter__(self) -> "FrankaRealEnv":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
