"""Main entry point for Franka robot evaluation.

Usage:
    # Local inference with impedance control (default)
    uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora

    # Local inference with position control (using shifted-state-to-action config)
    uv run examples/franka/main.py \\
        --checkpoint-dir ./checkpoints/pi05_franka_position_control_lora/11999 \\
        --config pi05_franka_position_control_lora \\
        --control-mode cartesian \\
        --cartesian-velocity-factor 0.05

    # Remote inference (policy server mode)
    uv run examples/franka/main.py --remote-host 0.0.0.0 --remote-port 8000

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
import time

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import env as _env
from examples.franka import real_env as _real_env
from examples.franka import pkl_recorder as _pkl_recorder

logger = logging.getLogger(__name__)

# Load default config for evaluation parameter defaults
_default_config = _real_env.RealEnvConfig.from_yaml()


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

    # Recording (optional)
    record_pkl: bool = False
    record_dir: str = "eval_records"
    record_fps: float = 30.0
    record_queue_size: int = 256
    record_config_name: str | None = None


def _create_local_policy(checkpoint_dir: str, config_name: str) -> tuple[object, object]:
    """Load a local policy from checkpoint.

    Returns:
        Tuple of (policy, train_config)
    """
    from openpi.policies import policy_config
    from openpi.training import config as _config

    logger.info("Loading policy from checkpoint: %s (config: %s)", checkpoint_dir, config_name)
    cfg = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(
        train_config=cfg,
        checkpoint_dir=checkpoint_dir,
    )
    logger.info("Policy loaded successfully")
    return policy, cfg


def _create_remote_policy(host: str, port: int) -> _websocket_client_policy.WebsocketClientPolicy:
    """Create a remote policy client."""
    logger.info("Connecting to remote policy server at %s:%s", host, port)
    client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    logger.info("Connected to remote policy server")
    return client


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
    # Validate args
    if args.checkpoint_dir is not None and args.remote_host is not None:
        raise ValueError("Cannot specify both checkpoint_dir and remote_host")
    if args.checkpoint_dir is None and args.remote_host is None:
        raise ValueError("Must specify either checkpoint_dir (local) or remote_host (remote)")
    if args.checkpoint_dir is not None and args.config is None:
        raise ValueError("Must specify --config when using --checkpoint-dir")

    # Create policy
    if args.remote_host is not None:
        policy = _create_remote_policy(args.remote_host, args.remote_port or 8000)
        action_horizon = args.open_loop_horizon or 30  # Default for remote
    else:
        policy, cfg = _create_local_policy(args.checkpoint_dir, args.config)  # type: ignore[arg-type]
        action_horizon = args.open_loop_horizon or cfg.model.action_horizon

    # Create action chunk broker
    chunked_policy = action_chunk_broker.ActionChunkBroker(
        policy=policy,
        action_horizon=action_horizon,
    )

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
    effective_control_fps = args.control_fps if args.control_fps is not None else _default_config.control_fps

    subscribers: list = []
    if args.record_pkl:
        config_name = args.record_config_name or args.config or "unknown"
        recorder_config = _pkl_recorder.RecorderConfig(
            record_dir=Path(args.record_dir),
            record_fps=args.record_fps,
            queue_size=args.record_queue_size,
            config_name=config_name,
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
        max_episode_steps=int(args.max_episode_time * effective_control_fps),
    )

    # Connect to robot
    real_env.connect()

    try:
        # Verify camera connection
        if camera.ping():
            logger.info("Camera service connected")
        else:
            logger.warning("Camera service not responding, continuing without camera")

        # Run evaluation episodes
        results = []
        for episode_idx in range(args.num_episodes):
            try:
                result = _run_episode(runtime, environment, episode_idx)
                results.append(result)
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

    finally:
        real_env.disconnect()
        camera.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
