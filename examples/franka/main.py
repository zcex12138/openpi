"""Main entry point for Franka robot evaluation.

Usage:
    # Local inference (default)
    uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora

    # Remote inference (policy server mode)
    uv run examples/franka/main.py --remote-host 0.0.0.0 --remote-port 8000
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import time
from typing import Any

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

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command-line arguments for Franka evaluation."""

    # Checkpoint (for local inference)
    checkpoint_dir: str | None = None
    config: str | None = None

    # Robot connection
    robot_ip: str = constants.ROBOT_IP
    robot_port: int = constants.ROBOT_PORT

    # Control parameters
    control_fps: float = constants.CONTROL_FPS
    open_loop_horizon: int | None = None  # None = use model action_horizon
    max_episode_time: float = constants.MAX_EPISODE_TIME
    num_episodes: int = constants.NUM_EPISODES

    # Task
    prompt: str = constants.DEFAULT_PROMPT

    # Safety
    max_pos_speed: float = constants.MAX_POS_SPEED

    # Output
    save_video: bool = False  # Video saving not implemented yet
    output_dir: str = "./eval_results"
    save_summary: bool = True

    # Camera service (Python 3.9)
    camera_host: str = constants.CAMERA_HOST
    camera_port: int = constants.CAMERA_PORT
    camera_timeout_s: float = constants.CAMERA_TIMEOUT_S

    # Remote policy (optional, mutually exclusive with checkpoint_dir)
    remote_host: str | None = None
    remote_port: int | None = None


def _create_local_policy(checkpoint_dir: str, config_name: str) -> tuple[Any, Any]:
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
    metadata = client.get_server_metadata()
    logger.info("Connected to remote policy server. Metadata: %s", metadata)
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


def _save_results(results: list[dict], output_dir: pathlib.Path) -> None:
    """Save evaluation results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "elapsed_time", "success"])
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved to %s", output_path)


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

    # Create robot environment
    real_env = _real_env.FrankaRealEnv(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        control_fps=args.control_fps,
        workspace_bounds=constants.WORKSPACE_BOUNDS,
        max_pos_speed=args.max_pos_speed,
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

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=chunked_policy),
        subscribers=[],
        max_hz=args.control_fps,
        num_episodes=1,  # We control episodes manually
        max_episode_steps=int(args.max_episode_time * args.control_fps),
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

        # Save results
        if args.save_summary and results:
            _save_results(results, pathlib.Path(args.output_dir))

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
