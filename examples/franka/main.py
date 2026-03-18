"""Main entry point for Franka robot evaluation.

Usage:
    # Local inference with impedance control (default)
    uv run examples/franka/main.py --args.checkpoint-dir ./checkpoints/11999 --args.config pi05_franka_screwdriver_lora

    # Local inference with position control (using shifted-state-to-action config)
    uv run examples/franka/main.py \\
        --args.checkpoint-dir ./checkpoints/pi05_franka_position_control_lora/11999 \\
        --args.config pi05_franka_position_control_lora \\
        --args.control-mode cartesian \\
        --args.cartesian-velocity-factor 0.05

    # Remote inference (policy server mode)
    uv run examples/franka/main.py --args.remote-host 0.0.0.0 --args.remote-port 8000

Control modes:
    - impedance: Default mode, uses impedance control with delta actions
    - cartesian: Cartesian position control, suitable for position control configs
                 (e.g., pi05_franka_position_control_lora)

Safety tips for position control:
    - Start with low velocity factor (0.01-0.03) for initial testing
    - Increase gradually to 0.05-0.1 once behavior is verified
    - Keep emergency stop button ready
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np
from openpi.serving import policy_loading as _policy_loading
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.cr_dagger_chunk_broker import CrDaggerChunkBroker, CrDaggerChunkBrokerConfig
from openpi_client.realtime_chunk_broker import RealTimeChunkBroker, RTCConfig
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import env as _env
from examples.franka import real_env as _real_env
from examples.franka import pkl_recorder as _pkl_recorder
from examples.franka.keyboard_utils import cbreak_terminal

logger = logging.getLogger(__name__)

# Load default config for evaluation parameter defaults
_default_config = _real_env.RealEnvConfig.from_yaml()
_EXECUTION_MODES = ("sync", "rtc", "cr_dagger_baseline")
_POLICY_MODES = ("service", "local")


@dataclasses.dataclass(frozen=True)
class ResolvedExecutionSettings:
    mode: str
    control_hz: float
    rtc_inference_delay: int
    rtc_execute_horizon: int
    cr_dagger_execute_horizon: int
    cr_dagger_max_skip_steps: int


@dataclasses.dataclass(frozen=True)
class ResolvedPolicySettings:
    mode: str
    checkpoint_dir: str | None
    config_name: str | None
    remote_host: str | None
    remote_port: int | None


@dataclasses.dataclass(frozen=True)
class ResolvedResidualSettings:
    checkpoint_dir: str | None
    device: str
    scale: float
    translation_cap_m: float | None
    rotation_cap_rad: float | None
    gripper_cap: float | None

    @property
    def enabled(self) -> bool:
        return self.checkpoint_dir is not None


@dataclasses.dataclass
class Args:
    """Command-line arguments for Franka evaluation."""

    # Checkpoint (for local inference)
    checkpoint_dir: str | None = None
    config: str | None = None

    # Real environment config file
    real_env_config: str | None = None  # Path to real_env_config.yaml (None = use default)

    # Robot connection (can override config file)
    robot_ip: str | None = None

    # Control parameters (can override config file)
    control_mode: str | None = None  # "impedance" or "cartesian"
    control_fps: float | None = None
    open_loop_horizon: int | None = None  # None = use model action_horizon
    max_episode_time: float = _default_config.max_episode_time
    num_episodes: int = _default_config.num_episodes
    action_smoothing_alpha: float | None = None  # None=use config, 0.0=no smoothing, 0.9=heavy
    cartesian_velocity_factor: float | None = None  # Velocity factor for cartesian mode
    translation_scale: float | None = None  # Scale factor for xyz delta (>1.0 amplifies translation)
    rotation_scale: float | None = None     # Scale factor for rotation delta via slerp (>1.0 amplifies rotation)

    # Task
    prompt: str = _default_config.default_prompt

    # Safety (can override config file)
    max_pos_speed: float | None = None

    # Camera service (Python 3.9)
    camera_host: str = constants.CAMERA_HOST
    camera_port: int = constants.CAMERA_PORT
    camera_timeout_s: float = constants.CAMERA_TIMEOUT_S

    # Remote policy (optional, mutually exclusive with checkpoint_dir)
    remote_host: str | None = None
    remote_port: int | None = None

    # Residual policy (optional, applied after broker per control step)
    residual_checkpoint_dir: str | None = None
    residual_device: str = "auto"
    residual_scale: float | None = None  # None = use config file
    residual_translation_cap_m: float | None = None
    residual_rotation_cap_rad: float | None = None
    residual_gripper_cap: float | None = None

    # Recording (optional)
    record_pkl: bool = False
    record_dir: str | None = None  # None = use config file value

    # Canonical execution mode
    execution_mode: str | None = None  # "sync" | "rtc" | "cr_dagger_baseline"
    cr_dagger_execute_horizon: int | None = None
    cr_dagger_max_skip_steps: int | None = None

    # Real-Time Chunking (RTC) parameters; execution is selected by execution_mode
    rtc: bool = False  # Legacy shorthand for execution_mode="rtc"
    rtc_inference_delay: int | None = None  # None = use config file value
    rtc_execute_horizon: int | None = None  # None = use config file value


def _create_local_policy(checkpoint_dir: str, config_name: str) -> tuple[object, object]:
    """Load a local policy from checkpoint.

    Returns:
        Tuple of (policy, train_config)
    """
    logger.info("Loading policy from checkpoint: %s (config: %s)", checkpoint_dir, config_name)
    policy, cfg = _policy_loading.load_checkpoint_policy(checkpoint_dir, config_name)
    logger.info("Policy loaded successfully")
    return policy, cfg


def _create_remote_policy(host: str, port: int) -> _websocket_client_policy.WebsocketClientPolicy:
    """Create a remote policy client."""
    logger.info("Connecting to remote policy server at %s:%s", host, port)
    client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    logger.info("Connected to remote policy server")
    return client


def _resolve_policy_settings(args: Args, env_config: _real_env.RealEnvConfig) -> ResolvedPolicySettings:
    policy_mode = env_config.policy_default_mode
    if policy_mode not in _POLICY_MODES:
        raise ValueError(f"Unsupported policy.default_mode={policy_mode!r}. Choose from {', '.join(_POLICY_MODES)}.")

    explicit_remote = args.remote_host is not None or args.remote_port is not None
    explicit_local = args.checkpoint_dir is not None or args.config is not None

    if explicit_remote and explicit_local:
        raise ValueError("Cannot specify both local checkpoint arguments and remote service arguments")

    if args.checkpoint_dir is not None:
        if args.config is None:
            raise ValueError("Must specify --args.config when using --args.checkpoint-dir")
        return ResolvedPolicySettings(
            mode="local",
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.config,
            remote_host=None,
            remote_port=None,
        )

    if args.config is not None:
        raise ValueError("--args.config requires --args.checkpoint-dir")

    if explicit_remote:
        remote_host = args.remote_host if args.remote_host is not None else env_config.policy_remote_host
        remote_port = args.remote_port if args.remote_port is not None else env_config.policy_remote_port
        return ResolvedPolicySettings(
            mode="service",
            checkpoint_dir=None,
            config_name=None,
            remote_host=remote_host,
            remote_port=remote_port,
        )

    if policy_mode == "service":
        return ResolvedPolicySettings(
            mode="service",
            checkpoint_dir=None,
            config_name=None,
            remote_host=env_config.policy_remote_host,
            remote_port=env_config.policy_remote_port,
        )

    raise ValueError(
        "No inference source configured. Provide --args.checkpoint-dir/--args.config, provide --args.remote-host, "
        "or set policy.default_mode=service in real_env_config.yaml."
    )


def _resolve_execution_settings(
    args: Args,
    env_config: _real_env.RealEnvConfig,
    *,
    action_horizon: int,
) -> ResolvedExecutionSettings:
    explicit_mode = args.execution_mode if args.execution_mode is not None else env_config.execution_mode
    if explicit_mode is not None and explicit_mode not in _EXECUTION_MODES:
        raise ValueError(
            f"Unsupported execution_mode={explicit_mode!r}. Choose from {', '.join(_EXECUTION_MODES)}."
        )

    if explicit_mode is not None:
        if args.rtc and explicit_mode != "rtc":
            raise ValueError(
                "execution_mode conflicts with legacy `--args.rtc` shorthand. "
                "Disable `--args.rtc`, or select `execution_mode='rtc'`."
            )
        mode = explicit_mode
    else:
        legacy_rtc_enabled = bool(args.rtc or env_config.rtc_enabled)
        mode = "rtc" if legacy_rtc_enabled else "sync"

    control_hz = args.control_fps if args.control_fps is not None else env_config.control_fps
    rtc_inference_delay = (
        args.rtc_inference_delay if args.rtc_inference_delay is not None else env_config.rtc_inference_delay
    )
    rtc_execute_horizon = (
        args.rtc_execute_horizon if args.rtc_execute_horizon is not None else env_config.rtc_execute_horizon
    )
    cr_dagger_execute_horizon = (
        args.cr_dagger_execute_horizon
        if args.cr_dagger_execute_horizon is not None
        else env_config.cr_dagger_execute_horizon
    )
    cr_dagger_max_skip_steps = (
        args.cr_dagger_max_skip_steps
        if args.cr_dagger_max_skip_steps is not None
        else env_config.cr_dagger_max_skip_steps
    )

    if mode == "cr_dagger_baseline" and cr_dagger_execute_horizon > action_horizon:
        raise ValueError(
            "CR-Dagger execute_horizon exceeds the known model action horizon "
            f"({cr_dagger_execute_horizon} > {action_horizon})."
        )

    return ResolvedExecutionSettings(
        mode=mode,
        control_hz=control_hz,
        rtc_inference_delay=rtc_inference_delay,
        rtc_execute_horizon=rtc_execute_horizon,
        cr_dagger_execute_horizon=cr_dagger_execute_horizon,
        cr_dagger_max_skip_steps=cr_dagger_max_skip_steps,
    )


def _resolve_residual_settings(args: Args, env_config: _real_env.RealEnvConfig) -> ResolvedResidualSettings:
    residual_checkpoint_dir = (
        args.residual_checkpoint_dir
        if args.residual_checkpoint_dir is not None
        else env_config.residual_checkpoint_dir
    )
    residual_scale = args.residual_scale if args.residual_scale is not None else env_config.residual_scale
    residual_translation_cap_m = (
        args.residual_translation_cap_m
        if args.residual_translation_cap_m is not None
        else env_config.residual_translation_cap_m
    )
    residual_rotation_cap_rad = (
        args.residual_rotation_cap_rad
        if args.residual_rotation_cap_rad is not None
        else env_config.residual_rotation_cap_rad
    )
    if residual_scale < 0:
        raise ValueError(f"residual_scale must be >= 0, got {residual_scale}")
    if residual_translation_cap_m is not None and residual_translation_cap_m <= 0:
        raise ValueError(f"residual_translation_cap_m must be > 0, got {residual_translation_cap_m}")
    if residual_rotation_cap_rad is not None and residual_rotation_cap_rad <= 0:
        raise ValueError(f"residual_rotation_cap_rad must be > 0, got {residual_rotation_cap_rad}")
    if args.residual_gripper_cap is not None and args.residual_gripper_cap <= 0:
        raise ValueError(f"residual_gripper_cap must be > 0, got {args.residual_gripper_cap}")
    if residual_checkpoint_dir is not None:
        checkpoint_path = Path(residual_checkpoint_dir).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Residual checkpoint_dir not found: {residual_checkpoint_dir}")
        if not checkpoint_path.is_dir():
            raise NotADirectoryError(f"Residual checkpoint_dir is not a directory: {residual_checkpoint_dir}")
        residual_checkpoint_dir = str(checkpoint_path.resolve())
    return ResolvedResidualSettings(
        checkpoint_dir=residual_checkpoint_dir,
        device=args.residual_device,
        scale=residual_scale,
        translation_cap_m=residual_translation_cap_m,
        rotation_cap_rad=residual_rotation_cap_rad,
        gripper_cap=args.residual_gripper_cap,
    )


def _resolve_record_dir(args: Args, env_config: _real_env.RealEnvConfig) -> Path:
    record_dir = args.record_dir if args.record_dir is not None else env_config.record_dir
    if not record_dir:
        raise ValueError("record_dir must not be empty")
    return Path(record_dir).expanduser()


def _maybe_wrap_with_residual(policy: object, residual_settings: ResolvedResidualSettings) -> object:
    if not residual_settings.enabled:
        return policy

    from residual_policy.inference import FrankaResidualStepPolicy
    from residual_policy.inference import ResidualInferenceConfig

    logger.info(
        "Enabling residual policy: checkpoint=%s scale=%.3f translation_cap=%s rotation_cap=%s gripper_cap=%s device=%s apply_gripper_delta=%s",
        residual_settings.checkpoint_dir,
        residual_settings.scale,
        residual_settings.translation_cap_m,
        residual_settings.rotation_cap_rad,
        residual_settings.gripper_cap,
        residual_settings.device,
        False,
    )
    return FrankaResidualStepPolicy(
        policy=policy,
        config=ResidualInferenceConfig(
            checkpoint_dir=residual_settings.checkpoint_dir,
            device=residual_settings.device,
            scale=residual_settings.scale,
            translation_cap_m=residual_settings.translation_cap_m,
            rotation_cap_rad=residual_settings.rotation_cap_rad,
            gripper_cap=residual_settings.gripper_cap,
            apply_gripper_delta=False,
        ),
    )


def _maybe_wrap_policy_with_pose10(policy: object, residual_settings: ResolvedResidualSettings) -> object:
    if not residual_settings.enabled:
        return policy

    from residual_policy.inference import FrankaPolicyPose10Wrapper

    logger.info("Exposing Franka base policy actions in pose10 [xyz,r6d,gripper] space before execution.")
    return FrankaPolicyPose10Wrapper(policy)


def _run_episode(
    runtime: _runtime.Runtime,
    environment: _env.FrankaEnvironment,
    episode_idx: int,
) -> dict:
    """Run a single evaluation episode.

    Returns:
        Episode result dict with keys: episode, steps, elapsed_time, success
    """
    logger.info("=" * 50)
    logger.info("Episode %d", episode_idx + 1)
    logger.info("=" * 50)

    # Wait for user confirmation before starting
    input("Press Enter to start episode (Ctrl+C to abort)...")

    start_time = time.time()
    # Runtime.run() handles the episode loop internally
    with cbreak_terminal():
        runtime.run()
    elapsed_time = time.time() - start_time

    result = {
        "episode": episode_idx + 1,
        "steps": environment.step_count,
        "elapsed_time": elapsed_time,
        "success": False,  # Manual annotation required
    }

    logger.info("Episode %d complete: %d steps in %.1fs", episode_idx + 1, result["steps"], result["elapsed_time"])
    return result


def main(args: Args) -> None:
    """Main evaluation function."""
    resolved_env_config = _real_env.RealEnvConfig.from_yaml(args.real_env_config)
    policy_settings = _resolve_policy_settings(args, resolved_env_config)
    residual_settings = _resolve_residual_settings(args, resolved_env_config)
    logger.info(
        "Policy mode resolved to %s (config default: %s)",
        policy_settings.mode,
        resolved_env_config.policy_default_mode,
    )

    # Create policy
    if policy_settings.mode == "service":
        policy = _create_remote_policy(policy_settings.remote_host or "localhost", policy_settings.remote_port or 8000)
        action_horizon = args.open_loop_horizon or 30  # Default for remote
    else:
        if policy_settings.checkpoint_dir is None or policy_settings.config_name is None:
            raise RuntimeError("Local policy settings are incomplete")
        policy, cfg = _create_local_policy(policy_settings.checkpoint_dir, policy_settings.config_name)
        action_horizon = args.open_loop_horizon or cfg.model.action_horizon
    policy = _maybe_wrap_policy_with_pose10(policy, residual_settings)

    execution = _resolve_execution_settings(args, resolved_env_config, action_horizon=action_horizon)
    logger.info(
        "Execution mode resolved to %s (override with --args.execution-mode {%s})",
        execution.mode,
        ",".join(_EXECUTION_MODES),
    )

    if execution.mode == "rtc":
        rtc_config = RTCConfig(
            action_horizon=action_horizon,
            inference_delay=execution.rtc_inference_delay,
            execute_horizon=execution.rtc_execute_horizon,
            control_hz=execution.control_hz,
            use_action_prefix=False,
        )
        chunked_policy = RealTimeChunkBroker(policy=policy, config=rtc_config)
        logger.info(
            "RTC parameters: action_horizon=%d, inference_delay=%d, execute_horizon=%d, control_hz=%.1f, action_prefix=%s",
            action_horizon,
            execution.rtc_inference_delay,
            execution.rtc_execute_horizon,
            execution.control_hz,
            "off",
        )
    elif execution.mode == "cr_dagger_baseline":
        cr_dagger_config = CrDaggerChunkBrokerConfig(
            action_horizon=action_horizon,
            execute_horizon=execution.cr_dagger_execute_horizon,
            max_skip_steps=execution.cr_dagger_max_skip_steps,
            control_hz=execution.control_hz,
        )
        chunked_policy = CrDaggerChunkBroker(policy=policy, config=cr_dagger_config)
        logger.info(
            "CR-Dagger baseline parameters: action_horizon=%d, execute_horizon=%d, max_skip_steps=%d, control_hz=%.1f",
            action_horizon,
            execution.cr_dagger_execute_horizon,
            execution.cr_dagger_max_skip_steps,
            execution.control_hz,
        )
    else:
        chunked_policy = action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=action_horizon,
        )
        logger.info("Sync execution parameters: action_horizon=%d, control_hz=%.1f", action_horizon, execution.control_hz)
    chunked_policy = _maybe_wrap_with_residual(chunked_policy, residual_settings)

    # Create robot environment (loads from real_env_config.yaml by default)
    real_env = _real_env.FrankaRealEnv(
        config_path=args.real_env_config,
        # Command-line overrides (None means use config file value)
        robot_ip=args.robot_ip,
        control_mode=args.control_mode,
        control_fps=args.control_fps,
        max_pos_speed=args.max_pos_speed,
        action_smoothing_alpha=args.action_smoothing_alpha,
        cartesian_velocity_factor=args.cartesian_velocity_factor,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
    )

    # Create camera client
    camera = _camera_client.CameraClient(
        host=args.camera_host,
        port=args.camera_port,
        timeout_s=args.camera_timeout_s,
    )

    # Create environment wrapper
    environment = _env.FrankaEnvironment(
        real_env=real_env,
        camera=camera,
        prompt=args.prompt,
        max_episode_time=args.max_episode_time,
    )

    # Get effective control fps (from args or config)
    effective_control_fps = execution.control_hz

    subscribers: list = []
    if args.record_pkl:
        record_dir = _resolve_record_dir(args, resolved_env_config)
        logger.info("Recording PKL episodes to %s", record_dir)
        recorder_config = _pkl_recorder.RecorderConfig(
            record_dir=record_dir,
            control_hz=effective_control_fps,
            prompt=args.prompt,
        )
        subscribers.append(_pkl_recorder.EpisodePklRecorder(environment, recorder_config))

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=chunked_policy),
        subscribers=subscribers,
        max_hz=effective_control_fps,
        num_episodes=1,  # We control episodes manually
        max_episode_steps=0,  # Disable step limit; timeout handled by FrankaEnvironment
    )

    # Connect to robot
    real_env.connect()

    try:
        try:
            # Verify camera connection
            if camera.ping():
                logger.info("Camera service connected")
            else:
                reason = "Camera service not responding before evaluation start"
                real_env.safety_stop_control(reason)
                raise _env.CameraSafetyStop(reason)

            # Run evaluation episodes
            results = []
            for episode_idx in range(args.num_episodes):
                try:
                    result = _run_episode(runtime, environment, episode_idx)
                    results.append(result)
                except _env.CameraSafetyStop as exc:
                    logger.error("Evaluation stopped by camera safety stop: %s", exc)
                    break
                except KeyboardInterrupt:
                    logger.info("Evaluation interrupted by user")
                    break

            # Print summary
            if results:
                avg_steps = np.mean([r["steps"] for r in results])
                avg_time = np.mean([r["elapsed_time"] for r in results])
                logger.info("=" * 50)
                logger.info("Evaluation complete: %d episodes", len(results))
                logger.info("Average steps: %.1f", avg_steps)
                logger.info("Average time: %.1fs", avg_time)
                logger.info("=" * 50)
        except _env.CameraSafetyStop as exc:
            logger.error("Evaluation aborted by camera safety stop: %s", exc)

    finally:
        real_env.disconnect()
        camera.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
