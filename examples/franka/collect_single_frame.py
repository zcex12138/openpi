"""Single-frame data collection for Franka robot in zero-impedance teaching mode."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import pickle
import re
import time
from typing import Any

import numpy as np
import tyro

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import env as _env
from examples.franka import real_env as _real_env
from examples.franka.keyboard_utils import check_key_pressed, cbreak_terminal
from examples.franka.pkl_recorder import _downsample_half, _empty_marker3d

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    real_env_config: str | None = None
    robot_ip: str | None = None
    camera_host: str = constants.CAMERA_HOST
    camera_port: int = constants.CAMERA_PORT
    camera_timeout_s: float = constants.CAMERA_TIMEOUT_S
    record_dir: str = "demo_records"


def _resolve_start_index(record_dir: Path) -> int:
    if not record_dir.exists():
        return -1
    max_idx = -1
    for path in record_dir.glob("episode_*.pkl"):
        match = re.match(r"episode_(\d+)$", path.stem)
        if match is None:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx


def _build_record(
    sample: dict[str, Any],
    frame_index: int,
    episode_start_ns: int | None,
) -> tuple[dict[str, Any], int]:
    frames = sample.get("frames", {})
    marker3d_data = sample.get("marker3d", {})
    images = {
        "l500": _downsample_half(frames.get("l500_rgb")),
        "d400": _downsample_half(frames.get("d400_rgb")),
        "xense_1": _downsample_half(frames.get("xense_1_rgb")),
    }

    markers = {
        "xense_1": marker3d_data.get("xense_1_marker3d", _empty_marker3d()),
    }

    timestamp_ns = int(sample.get("timestamp_ns", 0))
    if timestamp_ns > 0:
        if episode_start_ns is None:
            episode_start_ns = timestamp_ns
        timestamp = (timestamp_ns - episode_start_ns) / 1e9
    else:
        timestamp = 0.0
        if episode_start_ns is None:
            episode_start_ns = 0

    tcp_pose = np.asarray(sample.get("tcp_pose", np.zeros(7, dtype=np.float32)), dtype=np.float32)
    gripper = np.asarray(sample.get("gripper", np.zeros(1, dtype=np.float32)), dtype=np.float32)
    action = np.concatenate([tcp_pose, gripper], axis=0).astype(np.float32)

    record = {
        "timestamp": timestamp,
        "timestamp_ns": timestamp_ns,
        "seq": int(sample.get("seq", -1)),
        "frame_index": frame_index,
        "images": images,
        "marker3d": markers,
        "tcp_pose": tcp_pose,
        "tcp_velocity": np.asarray(sample.get("tcp_velocity", np.zeros(6, dtype=np.float32)), dtype=np.float32),
        "wrench": np.asarray(sample.get("wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32),
        "gripper": gripper,
        "action": action,
        "is_human_teaching": True,
    }
    return record, episode_start_ns


def _save_episode_pkl(
    record_dir: Path,
    episode_index: int,
    frames: list[dict[str, Any]],
) -> Path:
    record_dir.mkdir(parents=True, exist_ok=True)
    output_path = record_dir / f"episode_{episode_index:03d}.pkl"

    payload = {
        "version": 1,
        "episode_index": episode_index,
        "prompt": "",
        "fps": 0,
        "frames": frames,
    }

    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


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
        prompt="",
    )

    record_dir = Path(args.record_dir)
    episode_index = _resolve_start_index(record_dir)

    real_env.connect()

    try:
        if camera.ping():
            logger.info("Camera service connected")
        else:
            logger.warning("Camera service not responding")

        print("=" * 50)
        print("Single-Frame Data Collection")
        print("=" * 50)
        print("Controls:")
        print("  Enter  - Capture frame")
        print("  'g'    - Toggle gripper")
        print("  'q'    - End episode")
        print("  Ctrl+C - Quit")
        print("=" * 50)

        completed_episodes = 0
        frames: list[dict[str, Any]] = []
        frame_index = 0
        episode_start_ns: int | None = None

        try:
            while True:
                print(f"\nEpisode {completed_episodes + 1}: Press Enter to reset...")
                input()

                print("Opening gripper...")
                real_env._maybe_send_gripper_command(0.0)
                time.sleep(0.5)

                real_env.reset(grasp=True, start_control=False)

                print("Press Enter to start teaching mode...")
                input()

                real_env.enable_teaching_mode()
                print("Teaching mode enabled - guide robot by hand")
                print("Capturing... (Enter=capture, g=gripper, q=end)")

                episode_index += 1
                frames = []
                frame_index = 0
                episode_start_ns = None
                gripper_open = False

                with cbreak_terminal():
                    while True:
                        key = check_key_pressed()

                        if key == "g":
                            gripper_open = not gripper_open
                            real_env._maybe_send_gripper_command(0.0 if gripper_open else 1.0)
                            state_str = "OPEN" if gripper_open else "CLOSED"
                            print(f"Gripper: {state_str}")

                        elif key in ("\n", "\r"):
                            try:
                                sample = environment.get_recording_frame()
                                record, episode_start_ns = _build_record(sample, frame_index, episode_start_ns)
                                frames.append(record)
                                seq = record["seq"]
                                print(f"Frame {frame_index} captured (seq={seq})")
                                frame_index += 1
                            except Exception as exc:
                                logger.warning("Failed to capture frame: %s", exc)

                        elif key == "q":
                            break

                        time.sleep(0.02)

                output_path = _save_episode_pkl(record_dir, episode_index, frames)
                print(f"Saved: {output_path} ({len(frames)} frames)")
                completed_episodes += 1

        except KeyboardInterrupt:
            if frames:
                output_path = _save_episode_pkl(record_dir, episode_index, frames)
                print(f"\nSaved: {output_path} ({len(frames)} frames)")
                completed_episodes += 1
            logger.info("Data collection interrupted by user")

        print("=" * 50)
        print(f"Collection complete: {completed_episodes} episodes")
        print(f"Recordings saved to: {args.record_dir}")
        print("=" * 50)

    finally:
        real_env.disconnect()
        camera.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
