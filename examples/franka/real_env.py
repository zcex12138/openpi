"""Low-level Franka robot environment for state collection and action execution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from threading import Thread
import time
from typing import Any

from frankx import Affine, ImpedanceMotion, JointMotion, MotionData, Robot, Waypoint, WaypointMotion
import numpy as np
from examples.franka import constants
from examples.franka.gripper_interpolator import GripperStateInterpolator
from examples.franka.utils import GRIPPER_GRASP_EPSILON
from examples.franka.utils import IMPEDANCE_STARTUP_DELAY_S
from examples.franka.utils import THREAD_JOIN_TIMEOUT_S
from examples.franka.utils import align_quaternion_sign
from examples.franka.utils import get_nested
from examples.franka.utils import load_yaml_config
from examples.franka.utils import normalize_quaternion

logger = logging.getLogger(__name__)

# Default config file path
_DEFAULT_CONFIG_FILE = Path(__file__).parent / "real_env_config.yaml"


def _load_real_env_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load real_env configuration from YAML file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_FILE
    return load_yaml_config(path)


@dataclass
class RealEnvConfig:
    """Configuration for FrankaRealEnv.

    All parameters can be loaded from real_env_config.yaml.
    """

    # Robot connection
    robot_ip: str = "172.16.0.2"

    # Control
    control_mode: str = "impedance"
    control_fps: float = 30.0

    # Workspace bounds
    workspace_bounds_min: list[float] = field(default_factory=lambda: [0.2, -0.5, 0.0])
    workspace_bounds_max: list[float] = field(default_factory=lambda: [0.8, 0.5, 0.6])

    # Motion limits
    max_pos_speed: float = 0.5

    # Impedance control
    impedance_translational_stiffness: float = 1400.0
    impedance_rotational_stiffness: float = 80.0
    impedance_exponential_decay: float = 0.005

    # Cartesian control
    cartesian_velocity_factor: float = 0.05

    # Action smoothing
    action_smoothing_alpha: float = 0.1

    # Gripper
    gripper_interpolation_duration: float = 1.4
    gripper_command_interval: float = 1.5
    gripper_open_width: float = 0.078
    gripper_grasp_width: float = 0.015
    gripper_velocity: float = 0.1
    gripper_force: float = 30.0
    gripper_close_threshold: float = 0.7
    gripper_open_threshold: float = 0.3

    # Reset
    auto_reset_on_disconnect: bool = True
    default_joint_position: list[float] = field(
        default_factory=lambda: [-0.26134401, 0.46399827, -0.02856101, -2.23260865, -0.00302741, 2.67803179, 0.5054156]
    )
    move_speed_factor: float = 0.3

    # End-effector transform (4x4 row-major)
    ee_transform: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    # Evaluation parameters
    max_episode_time: float = 30.0
    num_episodes: int = 10
    default_prompt: str = "open the can with the screwdriver"

    # Policy inference defaults
    policy_default_mode: str = "service"
    policy_remote_host: str = "localhost"
    policy_remote_port: int = 8000

    # Teaching mode (per-axis stiffness: [X, Y, Z])
    teaching_translational_stiffness: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    teaching_rotational_stiffness: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    teaching_load_mass: float = 0.3
    teaching_load_com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    teaching_load_inertia: list[float] = field(default_factory=lambda: [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001])

    # Motion scale for impedance control (amplify small movements to overcome stiction)
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    # Canonical execution mode selection
    execution_mode: str | None = None

    # Real-Time Chunking (RTC)
    rtc_enabled: bool = False
    rtc_inference_delay: int = 3
    rtc_execute_horizon: int = 5

    # CR-Dagger baseline execution
    cr_dagger_execute_horizon: int = 10
    cr_dagger_max_skip_steps: int = 2

    @property
    def workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return workspace bounds as numpy arrays."""
        return (
            np.array(self.workspace_bounds_min, dtype=np.float32),
            np.array(self.workspace_bounds_max, dtype=np.float32),
        )

    @property
    def default_joint_position_array(self) -> np.ndarray:
        """Return default joint position as numpy array."""
        return np.array(self.default_joint_position, dtype=np.float64)

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> RealEnvConfig:
        """Load configuration from YAML file."""
        cfg = _load_real_env_config(config_path)

        # Default values
        _default_ws_min = [0.2, -0.5, 0.0]
        _default_ws_max = [0.8, 0.5, 0.6]
        _default_joint_pos = [-0.26134401, 0.46399827, -0.02856101, -2.23260865, -0.00302741, 2.67803179, 0.5054156]
        _default_ee_transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        _default_load_com = [0.0, 0.0, 0.0]
        _default_load_inertia = [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]

        return cls(
            robot_ip=get_nested(cfg, ["robot", "ip"], "172.16.0.2"),
            control_mode=get_nested(cfg, ["control", "mode"], "impedance"),
            control_fps=get_nested(cfg, ["control", "fps"], 30.0),
            workspace_bounds_min=get_nested(cfg, ["workspace_bounds", "min"], _default_ws_min),
            workspace_bounds_max=get_nested(cfg, ["workspace_bounds", "max"], _default_ws_max),
            max_pos_speed=get_nested(cfg, ["motion", "max_pos_speed"], 0.5),
            impedance_translational_stiffness=get_nested(cfg, ["impedance", "translational_stiffness"], 1400.0),
            impedance_rotational_stiffness=get_nested(cfg, ["impedance", "rotational_stiffness"], 80.0),
            impedance_exponential_decay=get_nested(cfg, ["impedance", "exponential_decay"], 0.005),
            cartesian_velocity_factor=get_nested(cfg, ["cartesian", "velocity_factor"], 0.05),
            action_smoothing_alpha=get_nested(cfg, ["smoothing", "alpha"], 0.1),
            gripper_interpolation_duration=get_nested(cfg, ["gripper", "interpolation_duration"], 1.4),
            gripper_command_interval=get_nested(cfg, ["gripper", "command_interval"], 1.5),
            gripper_open_width=get_nested(cfg, ["gripper", "open_width"], 0.078),
            gripper_grasp_width=get_nested(cfg, ["gripper", "grasp_width"], 0.015),
            gripper_velocity=get_nested(cfg, ["gripper", "velocity"], 0.1),
            gripper_force=get_nested(cfg, ["gripper", "force"], 30.0),
            gripper_close_threshold=get_nested(cfg, ["gripper", "close_threshold"], 0.7),
            gripper_open_threshold=get_nested(cfg, ["gripper", "open_threshold"], 0.3),
            auto_reset_on_disconnect=get_nested(cfg, ["reset", "auto_on_disconnect"], True),
            default_joint_position=get_nested(cfg, ["reset", "default_joint_position"], _default_joint_pos),
            move_speed_factor=get_nested(cfg, ["reset", "move_speed_factor"], 0.3),
            ee_transform=get_nested(cfg, ["ee_transform"], _default_ee_transform),
            # Evaluation parameters
            max_episode_time=get_nested(cfg, ["evaluation", "max_episode_time"], 30.0),
            num_episodes=get_nested(cfg, ["evaluation", "num_episodes"], 10),
            default_prompt=get_nested(cfg, ["evaluation", "default_prompt"], "open the can with the screwdriver"),
            policy_default_mode=get_nested(cfg, ["policy", "default_mode"], "service"),
            policy_remote_host=get_nested(cfg, ["policy", "remote_host"], "localhost"),
            policy_remote_port=get_nested(cfg, ["policy", "remote_port"], 8000),
            # Teaching mode (per-axis stiffness)
            teaching_translational_stiffness=get_nested(cfg, ["teaching", "translational_stiffness"], [0.0, 0.0, 0.0]),
            teaching_rotational_stiffness=get_nested(cfg, ["teaching", "rotational_stiffness"], [0.0, 0.0, 0.0]),
            teaching_load_mass=get_nested(cfg, ["teaching", "load_mass"], 0.3),
            teaching_load_com=get_nested(cfg, ["teaching", "load_com"], _default_load_com),
            teaching_load_inertia=get_nested(cfg, ["teaching", "load_inertia"], _default_load_inertia),
            execution_mode=get_nested(cfg, ["execution", "mode"], None),
            # Real-Time Chunking (RTC)
            rtc_enabled=get_nested(cfg, ["rtc", "enabled"], False),
            rtc_inference_delay=get_nested(cfg, ["rtc", "inference_delay"], 3),
            rtc_execute_horizon=get_nested(cfg, ["rtc", "execute_horizon"], 5),
            cr_dagger_execute_horizon=get_nested(cfg, ["cr_dagger", "execute_horizon"], 10),
            cr_dagger_max_skip_steps=get_nested(cfg, ["cr_dagger", "max_skip_steps"], 2),
            # Motion scale
            translation_scale=get_nested(cfg, ["motion", "translation_scale"], 1.0),
            rotation_scale=get_nested(cfg, ["motion", "rotation_scale"], 1.0),
        )


_CONTROL_MODES = ("impedance", "cartesian")


class FrankaRealEnv:
    """Low-level Franka robot environment (robot state + actions).

    Handles:
    - Robot communication via frankx Robot/Gripper
    - Robot state collection
    - Action execution with safety checks (workspace clipping, velocity limiting)

    State format: 14D [TCP pose (7D: x, y, z, qw, qx, qy, qz)
    + gripper state (1D: 0=open, 1=closed)
    + TCP wrench (6D: fx, fy, fz, tx, ty, tz)]
    Action format: 8D [position (3D: x, y, z) + quaternion (4D: qw, qx, qy, qz) + gripper (1D)]
    """

    def __init__(
        self,
        config: RealEnvConfig | None = None,
        config_path: str | Path | None = None,
        *,
        # Override parameters (take precedence over config file)
        robot_ip: str | None = None,
        control_mode: str | None = None,
        control_fps: float | None = None,
        workspace_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        max_pos_speed: float | None = None,
        impedance_translational_stiffness: float | None = None,
        impedance_rotational_stiffness: float | None = None,
        gripper_interpolation_duration: float | None = None,
        gripper_command_interval_s: float | None = None,
        auto_reset_on_disconnect: bool | None = None,
        action_smoothing_alpha: float | None = None,
        cartesian_velocity_factor: float | None = None,
        translation_scale: float | None = None,
        rotation_scale: float | None = None,
    ) -> None:
        # Load config from file or use provided config
        if config is not None:
            self._config = config
        else:
            self._config = RealEnvConfig.from_yaml(config_path)

        # Apply parameter overrides (if provided)
        self._robot_ip = robot_ip if robot_ip is not None else self._config.robot_ip

        _control_mode = control_mode if control_mode is not None else self._config.control_mode
        if _control_mode not in _CONTROL_MODES:
            raise ValueError(f"Unsupported control_mode={_control_mode}. Choose from {_CONTROL_MODES}.")
        self._control_mode = _control_mode

        self._control_fps = control_fps if control_fps is not None else self._config.control_fps
        self._workspace_bounds = workspace_bounds if workspace_bounds is not None else self._config.workspace_bounds
        self._max_pos_speed = max_pos_speed if max_pos_speed is not None else self._config.max_pos_speed
        self._dt = 1.0 / self._control_fps
        self._impedance_translational_stiffness = (
            impedance_translational_stiffness
            if impedance_translational_stiffness is not None
            else self._config.impedance_translational_stiffness
        )
        self._impedance_rotational_stiffness = (
            impedance_rotational_stiffness
            if impedance_rotational_stiffness is not None
            else self._config.impedance_rotational_stiffness
        )
        self._impedance_exponential_decay = self._config.impedance_exponential_decay
        self._gripper_command_interval_s = (
            gripper_command_interval_s
            if gripper_command_interval_s is not None
            else self._config.gripper_command_interval
        )
        self._auto_reset_on_disconnect = (
            auto_reset_on_disconnect if auto_reset_on_disconnect is not None else self._config.auto_reset_on_disconnect
        )
        self._action_smoothing_alpha = (
            action_smoothing_alpha if action_smoothing_alpha is not None else self._config.action_smoothing_alpha
        )
        self._cartesian_velocity_factor = (
            cartesian_velocity_factor
            if cartesian_velocity_factor is not None
            else self._config.cartesian_velocity_factor
        )
        self._translation_scale = translation_scale if translation_scale is not None else self._config.translation_scale
        self._rotation_scale = rotation_scale if rotation_scale is not None else self._config.rotation_scale

        self._robot: Robot | None = None
        self._gripper = None
        self._impedance_motion: ImpedanceMotion | None = None
        self._impedance_thread: Thread | None = None
        self._waypoint_motion: WaypointMotion | None = None
        self._waypoint_thread: Thread | None = None
        self._gripper_thread: Thread | None = None
        self._last_state: np.ndarray | None = None
        self._last_action: np.ndarray | None = None  # For action smoothing
        self._last_action_time: float = 0.0
        self._last_gripper_command_time: float | None = None
        self._last_gripper_target: float | None = None  # 0.0=open, 1.0=closed
        self._last_sent_quaternion: np.ndarray | None = None
        self._last_target_affine: Affine | None = None
        self._last_target_action: np.ndarray | None = None

        _gripper_interp_duration = (
            gripper_interpolation_duration
            if gripper_interpolation_duration is not None
            else self._config.gripper_interpolation_duration
        )
        self._gripper_interpolator = GripperStateInterpolator(interpolation_duration=_gripper_interp_duration)
        self._teaching_mode: bool = False

    def connect(self) -> None:
        """Connect to the Franka robot controller."""
        if self._robot is not None:
            logger.warning("Already connected to robot")
            return

        logger.info("Connecting to Franka robot at %s", self._robot_ip)
        self._robot = Robot(self._robot_ip)
        self._robot.set_default_behavior()
        self._robot.recover_from_errors()
        self._robot.set_EE(self._config.ee_transform)
        try:
            state = self._robot.get_state()
            affine = Affine(state.O_T_EE)
            quat = np.asarray(affine.quaternion(), dtype=np.float32)
            logger.info(
                "Current EE quaternion (w,x,y,z): %s",
                np.array2string(quat, precision=6, floatmode="fixed"),
            )
        except Exception as exc:
            logger.warning("Failed to read EE quaternion after set_EE: %s", exc)
        self._gripper = self._robot.get_gripper()
        logger.info("Connected to Franka robot")

    def disconnect(self) -> None:
        """Disconnect from the Franka robot controller."""
        if self._robot is not None:
            if self._auto_reset_on_disconnect:
                try:
                    # Ensure the robot returns to a safe reset state before disabling impedance control.
                    self.reset(grasp=False, start_control=False)
                except Exception as e:
                    logger.warning("Error resetting robot before disconnect: %s", e)
            try:
                self._stop_control_motion()
                self._robot.stop()
            except Exception as e:
                logger.warning("Error stopping control during disconnect: %s", e)
            finally:
                self._robot = None
                self._gripper = None
                self._gripper_thread = None
        logger.info("Disconnected from Franka robot")

    def get_state(self) -> np.ndarray:
        """Get current robot state.

        Returns:
            14D state: [TCP pose (7D: x, y, z, qw, qx, qy, qz)
            + gripper state (1D: 0=open, 1=closed)
            + TCP wrench (6D: fx, fy, fz, tx, ty, tz)]
        """
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        state = self._get_robot_state()
        affine = Affine(state.O_T_EE)
        position = np.asarray(affine.translation(), dtype=np.float32)
        quaternion = np.asarray(affine.quaternion(), dtype=np.float32)
        wrench = np.asarray(state.O_F_ext_hat_K, dtype=np.float32)
        gripper_state = self._get_interpolated_gripper_state()

        state_vec = np.concatenate([position, quaternion, [gripper_state], wrench]).astype(np.float32)
        self._last_state = state_vec
        return self._last_state

    def get_tcp_velocity(self) -> np.ndarray:
        """Get TCP 6D velocity from cached control-loop robot state if available."""
        state = self._get_robot_state()
        velocity = getattr(state, "O_dP_EE_c", None)
        if velocity is None:
            return np.zeros((6,), dtype=np.float32)
        return np.asarray(velocity, dtype=np.float32)

    def get_last_target_action(self) -> np.ndarray | None:
        if self._last_target_action is None:
            return None
        return self._last_target_action.copy()

    def execute_action(self, action: np.ndarray) -> np.ndarray:
        """Execute action on robot with safety checks.

        Args:
            action: 8D action [x, y, z, qw, qx, qy, qz, gripper]

        Returns:
            8D executed action after clipping/limiting/normalization.
        """
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (constants.ACTION_DIM,):
            raise ValueError(f"Expected action shape ({constants.ACTION_DIM},), got {action.shape}")

        # Apply action smoothing (EMA filter) if enabled
        if self._action_smoothing_alpha > 0.0 and self._last_action is not None:
            # EMA: smoothed = alpha * new + (1 - alpha) * old
            # For pose (position + quaternion), use Affine.slerp (translation lerp + quaternion slerp).
            alpha = float(np.clip(self._action_smoothing_alpha, 0.0, 1.0))
            prev_pos = self._last_action[:3]
            prev_q = normalize_quaternion(self._last_action[3:7])
            curr_pos = action[:3]
            curr_q = normalize_quaternion(action[3:7])
            curr_q = align_quaternion_sign(curr_q, prev_q)
            prev_affine = Affine(
                float(prev_pos[0]),
                float(prev_pos[1]),
                float(prev_pos[2]),
                float(prev_q[0]),
                float(prev_q[1]),
                float(prev_q[2]),
                float(prev_q[3]),
            )
            curr_affine = Affine(
                float(curr_pos[0]),
                float(curr_pos[1]),
                float(curr_pos[2]),
                float(curr_q[0]),
                float(curr_q[1]),
                float(curr_q[2]),
                float(curr_q[3]),
            )
            smooth_affine = prev_affine.slerp(curr_affine, alpha)
            action[:3] = np.asarray(smooth_affine.translation(), dtype=np.float32)
            action[3:7] = np.asarray(smooth_affine.quaternion(), dtype=np.float32)
            # Gripper: simple threshold-based, don't smooth
            action[7] = action[7]  # Keep original gripper command

        # Store for next iteration
        self._last_action = action.copy()

        # Apply motion scale (amplify movements to overcome stiction in impedance control)
        if self._last_state is not None:
            # Translation scale: linear extrapolation
            if self._translation_scale != 1.0:
                current_pos = self._last_state[:3]
                target_pos = action[:3]
                delta_xyz = target_pos - current_pos
                action[:3] = current_pos + delta_xyz * self._translation_scale

            # Rotation scale: slerp extrapolation
            if self._rotation_scale != 1.0:
                current_affine = Affine(0.0, 0.0, 0.0, *self._last_state[3:7].astype(float))
                target_affine = Affine(0.0, 0.0, 0.0, *action[3:7].astype(float))
                scaled_affine = current_affine.slerp(target_affine, self._rotation_scale)
                action[3:7] = np.asarray(scaled_affine.quaternion(), dtype=np.float32)

        # Extract position and gripper
        position = np.asarray(action[:3], dtype=np.float32)
        quaternion = np.asarray(action[3:7], dtype=np.float32)
        gripper = float(action[7])

        # Apply workspace clipping
        position_clipped = self._clip_to_workspace(position)
        if not np.allclose(position, position_clipped, atol=1e-4):
            logger.debug("Action clipped: %s -> %s", position, position_clipped)
        position = position_clipped

        # Apply velocity limiting
        if self._last_state is not None:
            current_pos = self._last_state[:3]
            position = self._limit_velocity(current_pos, position)

        reference_quat = self._last_sent_quaternion
        if reference_quat is None and self._last_state is not None:
            reference_quat = self._last_state[3:7]
        quaternion = align_quaternion_sign(quaternion, reference_quat)
        quaternion = normalize_quaternion(quaternion)
        self._last_sent_quaternion = quaternion.copy()

        executed_action = np.concatenate(
            [position, quaternion, np.asarray([gripper], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

        target_affine = Affine(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )
        # Ensure motion loop is running before updating targets.
        self._ensure_control_motion()
        if self._control_mode == "impedance":
            if self._impedance_motion is None:
                raise RuntimeError("Impedance motion not initialized")
            self._impedance_motion.target = target_affine
        else:
            if self._waypoint_motion is None:
                raise RuntimeError("Waypoint motion not initialized")
            self._waypoint_motion.set_next_waypoint(Waypoint(target_affine))
        self._last_target_affine = target_affine
        self._last_target_action = executed_action.copy()

        # Send gripper command separately (rate-limited + busy-aware)
        # self._maybe_send_gripper_command(gripper)
        self._last_action_time = time.time()
        return executed_action

    def _get_interpolated_gripper_state(self) -> float:
        """Return interpolated gripper state without reading width/force."""
        current_time = time.time()

        # Avoid querying gripper hardware state; rely on time-based interpolation.
        if self._gripper_thread is not None:
            try:
                if (not self._gripper_thread.is_alive()) and self._gripper_interpolator.is_interpolating:
                    self._gripper_interpolator.mark_early_termination()
            except (RuntimeError, AttributeError):
                pass

        return float(self._gripper_interpolator.get_state(current_time))

    def _get_gripper_is_moving(self) -> bool | None:
        if self._gripper is None:
            return None
        try:
            state = None
            if hasattr(self._gripper, "get_state"):
                state = self._gripper.get_state()
            elif hasattr(self._gripper, "read_once"):
                state = self._gripper.read_once()
            if state is not None and hasattr(state, "is_moving"):
                return bool(state.is_moving)
        except (RuntimeError, OSError):
            return None
        if self._gripper_thread is not None:
            return self._gripper_thread.is_alive()
        return None

    def _grasp_async(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float,
        epsilon_outer: float,
    ) -> Thread:
        """Call frankx grasp_async with compatibility for custom signatures."""
        if self._gripper is None:
            raise RuntimeError("Gripper not initialized.")
        try:
            return self._gripper.grasp_async(width, speed, force, epsilon_inner, epsilon_outer)
        except TypeError:
            # Fallback for custom frankx builds that remove epsilon args.
            return self._gripper.grasp_async(width, speed, force)

    def _compute_gripper_target(self, gripper_cmd: float) -> float:
        """Map action gripper scalar to target state (0=open, 1=closed)."""
        close_threshold = self._config.gripper_close_threshold
        open_threshold = self._config.gripper_open_threshold
        if gripper_cmd >= close_threshold:
            return 1.0
        if gripper_cmd <= open_threshold:
            return 0.0
        if self._last_gripper_target is not None:
            return self._last_gripper_target
        return 0.0

    def _maybe_send_gripper_command(self, gripper_cmd: float) -> None:
        """Send gripper open/close command if allowed by busy/interval checks."""
        if self._gripper is None:
            return

        target_state = self._compute_gripper_target(gripper_cmd)
        if self._last_gripper_target is not None and target_state == self._last_gripper_target:
            return

        current_time = time.time()
        if self._last_gripper_command_time is not None:
            if current_time - self._last_gripper_command_time < self._gripper_command_interval_s:
                return

        if self._get_gripper_is_moving() is True:
            return

        if target_state == 0.0:
            self._gripper_thread = self._gripper.move_async(
                self._config.gripper_open_width,
                self._config.gripper_velocity,
            )
        else:
            self._gripper_thread = self._grasp_async(
                self._config.gripper_grasp_width,
                self._config.gripper_velocity,
                self._config.gripper_force,
                GRIPPER_GRASP_EPSILON,
                GRIPPER_GRASP_EPSILON,
            )

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

    def _get_robot_state(self):
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        if self._control_mode == "impedance" and self._impedance_motion is not None:
            if self._impedance_thread is not None and self._impedance_thread.is_alive():
                state = getattr(self._impedance_motion, "robot_state", None)
                if state is not None:
                    return state
        if self._control_mode == "cartesian" and self._waypoint_motion is not None:
            if self._waypoint_thread is not None and self._waypoint_thread.is_alive():
                state = getattr(self._waypoint_motion, "robot_state", None)
                if state is not None:
                    return state
        return self._robot.get_state()

    def _get_current_affine(self, *, force_robot_state: bool = False) -> Affine:
        if force_robot_state:
            if self._robot is None:
                raise RuntimeError("Robot not connected. Call connect() first.")
            state = self._robot.get_state()
        else:
            state = self._get_robot_state()
        return Affine(state.O_T_EE)

    def _log_state_cache(self, label: str, cached_state) -> None:
        if self._robot is None or cached_state is None:
            return

        control_running = False
        if self._control_mode == "impedance":
            control_running = self._impedance_thread is not None and self._impedance_thread.is_alive()
        elif self._control_mode == "cartesian":
            control_running = self._waypoint_thread is not None and self._waypoint_thread.is_alive()

        if control_running:
            try:
                cached_pos = np.asarray(Affine(cached_state.O_T_EE).translation(), dtype=np.float32)
                logger.info(
                    "%s state cache: cached_pos=%s",
                    label,
                    np.array2string(cached_pos, precision=4, floatmode="fixed"),
                )
            except Exception as exc:
                logger.warning("%s state cache check failed: %s", label, exc)
            return

        try:
            robot_state = self._robot.get_state()
            robot_pos = np.asarray(Affine(robot_state.O_T_EE).translation(), dtype=np.float32)
            cached_pos = np.asarray(Affine(cached_state.O_T_EE).translation(), dtype=np.float32)
            pos_delta = float(np.linalg.norm(robot_pos - cached_pos))
            logger.info(
                "%s state cache check: robot_pos=%s cache_pos=%s delta=%.4f m",
                label,
                np.array2string(robot_pos, precision=4, floatmode="fixed"),
                np.array2string(cached_pos, precision=4, floatmode="fixed"),
                pos_delta,
            )
        except Exception as exc:
            logger.warning("%s state cache check failed: %s", label, exc)

    def _start_impedance_motion(self) -> None:
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        self._stop_waypoint_motion()
        try:
            self._robot.set_load(
                self._config.teaching_load_mass,
                [0.0, 0.0, 0.0],
                [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001],
            )
            logger.info("Set load mass: %.3f kg", self._config.teaching_load_mass)
        except Exception as e:
            logger.warning("set_load failed: %s", e)
        if self._impedance_translational_stiffness is None and self._impedance_rotational_stiffness is None:
            self._impedance_motion = ImpedanceMotion()
        else:
            if self._impedance_translational_stiffness is None or self._impedance_rotational_stiffness is None:
                raise ValueError("Impedance stiffness values must be set together.")
            self._impedance_motion = ImpedanceMotion(
                float(self._impedance_translational_stiffness),
                float(self._impedance_rotational_stiffness),
            )
        self._impedance_motion.exponential_decay = self._impedance_exponential_decay
        logger.debug(
            "Impedance stiffness: translational=%.1f, rotational=%.1f, exponential_decay=%.4f",
            float(self._impedance_translational_stiffness),
            float(self._impedance_rotational_stiffness),
            self._impedance_exponential_decay,
        )
        # Use direct robot state to avoid stale impedance state during startup.
        current_affine = self._get_current_affine(force_robot_state=True)
        try:
            self._impedance_motion.target = current_affine
        except (RuntimeError, AttributeError):
            pass
        self._impedance_thread = self._robot.move_async(self._impedance_motion)
        try:
            time.sleep(IMPEDANCE_STARTUP_DELAY_S)
            cached_state = getattr(self._impedance_motion, "robot_state", None)
            if cached_state is not None:
                self._impedance_motion.target = Affine(cached_state.O_T_EE)
        except (RuntimeError, AttributeError):
            pass
        try:
            self._log_state_cache("Impedance", getattr(self._impedance_motion, "robot_state", None))
        except (RuntimeError, AttributeError):
            pass

    def _start_waypoint_motion(self) -> None:
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        self._stop_impedance_motion()

        # Set dynamic velocity/acceleration scaling for smoother cartesian motion
        try:
            self._robot.set_dynamic_rel(self._cartesian_velocity_factor)
            logger.info("Set cartesian dynamic velocity factor to %.3f", self._cartesian_velocity_factor)
        except Exception as e:
            logger.warning("Failed to set dynamic velocity factor: %s", e)

        current_affine = self._get_current_affine()
        initial_waypoint = Waypoint(current_affine)
        self._last_target_affine = current_affine
        self._waypoint_motion = WaypointMotion([initial_waypoint], return_when_finished=False)
        self._waypoint_thread = self._robot.move_async(self._waypoint_motion)
        logger.info("Started cartesian control with velocity_factor=%.2f", self._cartesian_velocity_factor)
        try:
            self._log_state_cache("Waypoint", self._waypoint_motion.robot_state)
        except (RuntimeError, AttributeError):
            pass

    def _ensure_control_motion(self) -> None:
        if self._control_mode == "impedance":
            if (
                self._impedance_motion is None
                or self._impedance_thread is None
                or not self._impedance_thread.is_alive()
            ):
                self._start_impedance_motion()
        elif self._waypoint_motion is None or self._waypoint_thread is None or not self._waypoint_thread.is_alive():
            self._start_waypoint_motion()

    def _stop_impedance_motion(self) -> None:
        if self._impedance_motion is not None:
            try:
                self._impedance_motion.finish()
            except (RuntimeError, AttributeError):
                pass
        if self._impedance_thread is not None:
            self._impedance_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        self._impedance_motion = None
        self._impedance_thread = None

    def _stop_waypoint_motion(self) -> None:
        if self._waypoint_motion is not None:
            try:
                self._waypoint_motion.finish()
            except (RuntimeError, AttributeError):
                pass
        if self._waypoint_thread is not None:
            self._waypoint_thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        self._waypoint_motion = None
        self._waypoint_thread = None
        self._last_target_affine = None

    def _stop_control_motion(self) -> None:
        self._stop_impedance_motion()
        self._stop_waypoint_motion()

    @property
    def is_teaching_mode(self) -> bool:
        return self._teaching_mode

    def enable_teaching_mode(self) -> None:
        """Switch to zero-stiffness teaching mode."""
        if self._control_mode != "impedance":
            logger.warning("Teaching mode only available in impedance control mode")
            return
        if self._teaching_mode:
            return
        if self._robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        logger.info("Enabling teaching mode...")
        self._stop_impedance_motion()

        try:
            self._robot.set_load(
                self._config.teaching_load_mass,
                self._config.teaching_load_com,
                self._config.teaching_load_inertia,
            )
        except Exception as e:
            logger.warning("set_load failed (robot may sag): %s", e)

        trans_stiffness = self._config.teaching_translational_stiffness
        rot_stiffness = self._config.teaching_rotational_stiffness
        self._impedance_motion = ImpedanceMotion(trans_stiffness, rot_stiffness)
        logger.info(
            "Teaching stiffness: translational=%s, rotational=%s",
            trans_stiffness,
            rot_stiffness,
        )
        current_affine = self._get_current_affine(force_robot_state=True)
        self._impedance_motion.target = current_affine

        try:
            self._impedance_thread = self._robot.move_async(self._impedance_motion)
        except Exception as e:
            raise RuntimeError(f"Failed to start teaching motion: {e}") from e

        self._teaching_mode = True
        logger.info("Teaching mode enabled - robot can be guided by hand")

    def reset(self, grasp: bool = True, *, start_control: bool = True) -> None:
        """Reset robot to default position and optionally grasp the screwdriver."""
        if self._robot is None or self._gripper is None:
            raise RuntimeError("Robot not connected. Call connect() first.")

        self._teaching_mode = False
        logger.info("Resetting robot...")

        logger.info("Stopping active control motions...")
        try:
            self._stop_control_motion()
            self._robot.stop()
        except Exception as e:
            logger.warning("Error stopping control motions: %s", e)

        # Restore dynamic speed scaling to default before reset motion.
        try:
            self._robot.set_dynamic_rel(1.0)
            logger.info("Reset dynamic velocity factor to 1.0 for reset motion")
        except Exception as e:
            logger.warning("Failed to reset dynamic velocity factor: %s", e)

        # Open gripper before moving joints
        logger.info("Opening gripper before moving joints...")
        try:
            self._gripper_thread = self._gripper.move_async(
                self._config.gripper_open_width,
                self._config.gripper_velocity,
            )
            self._gripper_thread.join(timeout=2.0)
            current_time = time.time()
            self._last_gripper_command_time = current_time
            self._last_gripper_target = 0.0
            self._gripper_interpolator.set_target(0.0, current_time)
            self._gripper_interpolator.mark_early_termination()
        except Exception as e:
            logger.warning("Error opening gripper: %s", e)

        # Move to default joint position
        logger.info("Moving robot to default joint position...")
        try:
            joint_motion = JointMotion(self._config.default_joint_position)
            motion_data = MotionData(self._config.move_speed_factor)
            success = self._robot.move(joint_motion, motion_data)
            if not success:
                logger.warning("Failed to move robot to default position")
        except Exception as e:
            logger.warning("Error moving to default position: %s", e)

        # Close gripper if requested
        if grasp:
            logger.info("Closing gripper to grasp screwdriver...")
            try:
                self._gripper_thread = self._grasp_async(
                    self._config.gripper_grasp_width,
                    self._config.gripper_velocity,
                    self._config.gripper_force,
                    0.01,
                    0.01,
                )
                self._gripper_thread.join(timeout=2.0)
                current_time = time.time()
                self._last_gripper_command_time = current_time
                self._last_gripper_target = 1.0
                self._gripper_interpolator.set_target(1.0, current_time)
                self._gripper_interpolator.mark_early_termination()
            except Exception as e:
                logger.warning("Error closing gripper: %s", e)

        if start_control:
            logger.info("Starting %s control mode...", self._control_mode)
            try:
                self._ensure_control_motion()
            except Exception as e:
                logger.warning("Error starting control motion: %s", e)

        # Clear last state to avoid velocity limiting issues
        self._last_state = None
        self._last_action = None  # Clear action smoothing buffer

        logger.info("Robot reset complete")

    def __enter__(self) -> FrankaRealEnv:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
