"""FrankaEnvironment wrapper for openpi runtime."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import real_env as _real_env

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class FrankaEnvironment(_environment.Environment):
    """Environment wrapper for Franka robot evaluation.

    Implements openpi_client.runtime.environment.Environment interface.
    Wraps FrankaRealEnv for robot control and CameraClient for image capture.

    Observation format (matching FrankaInputs expectations):
    - observation/image: Base camera image (L500), resized to 224x224
    - observation/wrist_image: Wrist camera image (D400), resized to 224x224
    - observation/state: Robot state (14D)
    - prompt: Task instruction
    """

    def __init__(
        self,
        real_env: _real_env.FrankaRealEnv,
        camera: _camera_client.CameraClient,
        *,
        prompt: str = constants.DEFAULT_PROMPT,
        render_height: int = constants.IMAGE_HEIGHT,
        render_width: int = constants.IMAGE_WIDTH,
        max_episode_time: float = constants.MAX_EPISODE_TIME,
    ) -> None:
        self._real_env = real_env
        self._camera = camera
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._max_episode_time = max_episode_time

        self._episode_start_time: float = 0.0
        self._step_count: int = 0
        self._episode_complete: bool = False
        self._last_frame_seq: int | None = None
        self._stale_frame_count: int = 0

    @override
    def reset(self) -> None:
        """Reset environment for a new episode."""
        logger.info("Resetting environment...")
        self._real_env.reset()
        self._episode_start_time = time.time()
        self._step_count = 0
        self._episode_complete = False
        logger.info("Environment reset complete")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete (timeout or user signal)."""
        if self._episode_complete:
            return True
        elapsed = time.time() - self._episode_start_time
        if elapsed > self._max_episode_time:
            logger.info("Episode timeout after %.1fs", elapsed)
            self._episode_complete = True
            return True
        return False

    def mark_episode_complete(self) -> None:
        """Mark the current episode as complete."""
        self._episode_complete = True

    @override
    def get_observation(self) -> dict:
        """Get current observation.

        Returns:
            Dict with keys matching FrankaInputs expectations:
            - observation/image: Base camera image (L500), uint8 (H, W, C)
            - observation/wrist_image: Wrist camera image (D400), uint8 (H, W, C)
            - observation/state: Robot state (14D)
            - prompt: Task instruction
        """
        # Get robot state
        state = self._real_env.get_state()

        # Get camera frames
        try:
            frames, _timestamp_ns, seq = self._camera.get_frames()
            l500_image = frames["l500_rgb"]
            d400_image = frames["d400_rgb"]
            if self._last_frame_seq is not None and seq == self._last_frame_seq:
                self._stale_frame_count += 1
                if self._stale_frame_count == 1 or self._stale_frame_count % 30 == 0:
                    logger.warning("Camera frames not advancing (seq=%d, stale_count=%d)", seq, self._stale_frame_count)
            else:
                self._stale_frame_count = 0
            self._last_frame_seq = seq
        except Exception as e:
            logger.warning("Camera frame retrieval failed: %s, using zero images", e)
            l500_image = np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
            d400_image = np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)

        # Resize images to model input size
        l500_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(l500_image, self._render_height, self._render_width)
        )
        d400_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(d400_image, self._render_height, self._render_width)
        )

        return {
            "observation/image": l500_image,
            "observation/wrist_image": d400_image,
            "observation/state": state,
            "prompt": self._prompt,
        }

    def get_recording_frame(self) -> dict:
        """Get raw frames + robot state for recording."""
        state = self._real_env.get_state()
        tcp_pose = np.asarray(state[:7], dtype=np.float32)
        gripper = np.asarray(state[7:8], dtype=np.float32)
        wrench = np.asarray(state[8:14], dtype=np.float32)
        tcp_velocity = self._real_env.get_tcp_velocity()

        try:
            frames, marker3d, timestamp_ns, seq = self._camera.get_frames_with_markers()
        except Exception as e:
            logger.warning("Camera frame retrieval failed for recording: %s", e)
            frames = {}
            marker3d = {}
            timestamp_ns = 0
            seq = -1

        return {
            "frames": frames,
            "marker3d": marker3d,
            "timestamp_ns": int(timestamp_ns),
            "seq": int(seq),
            "tcp_pose": tcp_pose,
            "tcp_velocity": tcp_velocity,
            "wrench": wrench,
            "gripper": gripper,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply a single-step action.

        Args:
            action: Dict with "actions" key containing the action array.
                   Shape: (action_horizon, action_dim) or (action_dim,)
        """
        actions = np.asarray(action["actions"])

        # Handle action chunk (take first action if chunked)
        if actions.ndim == 2:
            actions = actions[0]

        elapsed = time.time() - self._episode_start_time
        print(f"[openpi] step={self._step_count} t={elapsed:.3f}s", flush=True)

        executed_action = self._real_env.execute_action(actions)
        action["executed_action"] = executed_action
        self._step_count += 1

    @property
    def step_count(self) -> int:
        """Number of steps executed in current episode."""
        return self._step_count

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since episode start."""
        return time.time() - self._episode_start_time
