"""Zero-gravity impedance data collection for Franka robot."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import time

import tyro

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import env as _env
from examples.franka import pkl_recorder as _pkl_recorder
from examples.franka import real_env as _real_env
from examples.franka.keyboard_utils import check_key_pressed, cbreak_terminal

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    real_env_config: str | None = None
    robot_ip: str | None = None
    camera_host: str = constants.CAMERA_HOST
    camera_port: int = constants.CAMERA_PORT
    camera_timeout_s: float = constants.CAMERA_TIMEOUT_S
    record_dir: str = "demo_records"
    record_fps: float = 30.0
    prompt: str = ""
    num_episodes: int = 10


def main(args: Args) -> None:
    real_env = _real_env.FrankaRealEnv(
        config_path=args.real_env_config,
        robot_ip=args.robot_ip,
        control_mode="impedance",
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
    )

    recorder_config = _pkl_recorder.RecorderConfig(
        record_dir=Path(args.record_dir),
        record_fps=args.record_fps,
        prompt=args.prompt,
    )
    recorder = _pkl_recorder.EpisodePklRecorder(environment, recorder_config)

    real_env.connect()

    try:
        if camera.ping():
            logger.info("Camera service connected")
        else:
            logger.warning("Camera service not responding")

        logger.info("=" * 50)
        logger.info("Zero-Gravity Data Collection")
        logger.info("=" * 50)
        logger.info("Controls:")
        logger.info("  Enter  - Start/End episode")
        logger.info("  'g'    - Toggle gripper")
        logger.info("  Ctrl+C - Quit")
        logger.info("=" * 50)

        completed_episodes = 0

        try:
            for episode_idx in range(args.num_episodes):
                real_env.reset(grasp=False, start_control=False)

                print(f"\nEpisode {episode_idx + 1}/{args.num_episodes}: Press Enter to start recording...")
                input()

                real_env.enable_teaching_mode()
                logger.info("Teaching mode enabled - guide the robot by hand")

                recorder.on_episode_start()

                gripper_open = True
                print("Recording... ('g'=toggle gripper, Enter=end episode)")

                with cbreak_terminal():
                    while True:
                        key = check_key_pressed()

                        if key == "g":
                            gripper_open = not gripper_open
                            real_env._maybe_send_gripper_command(0.0 if gripper_open else 1.0)
                            state_str = "OPEN" if gripper_open else "CLOSED"
                            print(f"Gripper: {state_str}")

                        elif key in ("\n", "\r"):
                            break

                        time.sleep(0.02)

                recorder.on_episode_end()
                completed_episodes += 1
                logger.info("Episode %d complete", episode_idx + 1)

        except KeyboardInterrupt:
            logger.info("\nData collection interrupted by user")

        logger.info("=" * 50)
        logger.info("Collection complete: %d/%d episodes", completed_episodes, args.num_episodes)
        logger.info("Recordings saved to: %s", args.record_dir)
        logger.info("=" * 50)

    finally:
        real_env.disconnect()
        camera.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
