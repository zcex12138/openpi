"""Standalone PKL recording script for Franka evaluation data."""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import tyro

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import env as _env
from examples.franka import pkl_recorder as _pkl_recorder
from examples.franka import real_env as _real_env

logger = logging.getLogger(__name__)

_default_config = _real_env.RealEnvConfig.from_yaml()


@dataclasses.dataclass
class Args:
    """Command-line arguments for standalone PKL recording."""

    real_env_config: str | None = None
    robot_ip: str | None = None

    # Recording
    record_dir: str = "eval_records"
    record_fps: float = 30.0
    record_queue_size: int = 256
    config_name: str = "unknown"

    # Episode control
    max_episode_time: float = _default_config.max_episode_time
    num_episodes: int = _default_config.num_episodes
    prompt: str = _default_config.default_prompt

    # Camera service
    camera_host: str = constants.CAMERA_HOST
    camera_port: int = constants.CAMERA_PORT
    camera_timeout_s: float = constants.CAMERA_TIMEOUT_S


def main(args: Args) -> None:
    real_env = _real_env.FrankaRealEnv(
        config_path=args.real_env_config,
        robot_ip=args.robot_ip,
    )

    camera = _camera_client.CameraClient(
        host=args.camera_host,
        port=args.camera_port,
        timeout_s=args.camera_timeout_s,
    )

    environment = _env.FrankaEnvironment(
        real_env=real_env,
        camera=camera,
        prompt=args.prompt,
        max_episode_time=args.max_episode_time,
    )

    recorder_config = _pkl_recorder.RecorderConfig(
        record_dir=Path(args.record_dir),
        record_fps=args.record_fps,
        queue_size=args.record_queue_size,
        config_name=args.config_name,
        prompt=args.prompt,
    )
    recorder = _pkl_recorder.EpisodePklRecorder(environment, recorder_config)

    real_env.connect()
    try:
        if camera.ping():
            logger.info("Camera service connected")
        else:
            logger.warning("Camera service not responding, continuing without camera")

        for episode_idx in range(args.num_episodes):
            logger.info("=" * 50)
            logger.info("Recording episode %d", episode_idx + 1)
            logger.info("=" * 50)
            input("Press Enter to start episode (Ctrl+C to abort)...")

            recorder.on_episode_start()
            start_time = time.time()
            try:
                while time.time() - start_time < args.max_episode_time:
                    recorder.on_step({}, {})
                    if args.record_fps > 0:
                        time.sleep(1.0 / args.record_fps)
            except KeyboardInterrupt:
                logger.info("Recording interrupted by user")
                break
            finally:
                recorder.on_episode_end()

    finally:
        real_env.disconnect()
        camera.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
