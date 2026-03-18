"""FrankaEnvironment wrapper for openpi runtime."""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from residual_policy.action_repr import pose10_to_pose8
from typing_extensions import override

from examples.franka import camera_client as _camera_client
from examples.franka import constants
from examples.franka import real_env as _real_env

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class CameraSafetyStop(RuntimeError):
    """Raised when camera loss triggers a Franka evaluation safety stop."""


def _normalize_policy_action(action: object) -> tuple[np.ndarray, np.ndarray]:
    """Convert a canonical pose10 policy action to executable pose8."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]

    if arr.shape == (10,):
        pose10 = arr.copy()
        return pose10_to_pose8(pose10), pose10

    raise ValueError(f"Expected policy action shape (10,), got {arr.shape}")


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

        self._episode_start_time: float | None = None
        self._step_count: int = 0
        self._episode_complete: bool = False
        self._last_frame_seq: int | None = None
        self._stale_frame_count: int = 0
        self._last_control_step_time_s: float | None = None
        self._control_hz_ema: float | None = None
        self._latest_control_timestamp: float | None = None
        self._teaching_mode_active: bool = False
        self._keyboard_enabled: bool = sys.stdin.isatty()
        self._teaching_segment_id: int = -1
        self._active_teaching_segment_id: int | None = None
        self._active_teaching_step: int = 0
        self._camera_failure_reason: str | None = None
        if not self._keyboard_enabled:
            logger.warning("stdin is not a TTY, keyboard teaching disabled")

    @override
    def reset(self) -> None:
        """Reset environment for a new episode."""
        logger.info("Resetting environment...")
        self._real_env.reset()
        # Defer start time to first apply_action call
        self._episode_start_time = None
        self._step_count = 0
        self._episode_complete = False
        self._last_control_step_time_s = None
        self._control_hz_ema = None
        self._latest_control_timestamp = None
        self._teaching_mode_active = False
        self._teaching_segment_id = -1
        self._active_teaching_segment_id = None
        self._active_teaching_step = 0
        self._camera_failure_reason = None
        logger.info("Environment reset complete")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete (timeout or user signal)."""
        if self._episode_complete:
            return True
        if self._teaching_mode_active:
            return False
        if self._episode_start_time is None:
            return False
        elapsed = time.time() - self._episode_start_time
        if elapsed > self._max_episode_time:
            logger.info("Episode timeout after %.1fs", elapsed)
            self._episode_complete = True
            return True
        return False

    def mark_episode_complete(self) -> None:
        """Mark the current episode as complete."""
        self._episode_complete = True

    @property
    def camera_failure_reason(self) -> str | None:
        return self._camera_failure_reason

    @property
    def is_teaching_mode(self) -> bool:
        return self._real_env.is_teaching_mode

    def enable_teaching_mode(self) -> None:
        self._real_env.enable_teaching_mode()
        self._teaching_mode_active = True

    def disable_teaching_mode(self) -> None:
        self._real_env.disable_teaching_mode()
        self._teaching_mode_active = False
        self._active_teaching_segment_id = None
        self._active_teaching_step = 0

    def _poll_teaching_controls(self) -> None:
        """Handle runtime keyboard controls for human correction."""
        if not self._keyboard_enabled:
            return
        from examples.franka.keyboard_utils import check_key_pressed

        key = check_key_pressed()
        if key is None:
            return
        if not self._teaching_mode_active and key == " ":
            self.enable_teaching_mode()
            self._teaching_segment_id += 1
            self._active_teaching_segment_id = self._teaching_segment_id
            self._active_teaching_step = 0
            logger.info("Shared control enabled - guide robot by hand, press Space again to exit")
            return
        if self._teaching_mode_active and key == " ":
            self.disable_teaching_mode()
            logger.info("Shared control disabled - restoring normal stiffness while policy keeps running")

    def _trigger_camera_safety_stop(self, reason: str) -> None:
        if self._camera_failure_reason is None:
            self._camera_failure_reason = reason
            self._episode_complete = True
            self._teaching_mode_active = False
            self._active_teaching_segment_id = None
            self._active_teaching_step = 0
            try:
                self._real_env.safety_stop_control(reason)
            except Exception as exc:
                logger.warning("Failed to stop robot control after camera failure: %s", exc)
            logger.error("Camera safety stop: %s", reason)
        raise CameraSafetyStop(self._camera_failure_reason)

    @override
    def get_observation(self) -> dict:
        """Get current observation.

        Returns:
            Dict with keys matching FrankaInputs expectations:
            - observation/image: Base camera image (L500), uint8 (H, W, C)
            - observation/wrist_image: Wrist camera image (D400), uint8 (H, W, C)
            - observation/state: Robot state (14D)
            - observation/tactile: Tactile marker3d (optional, 26x14x3)
            - prompt: Task instruction
        """
        if self._camera_failure_reason is not None:
            raise CameraSafetyStop(self._camera_failure_reason)

        self._poll_teaching_controls()

        # Get robot state
        state = self._real_env.get_state()
        tcp_velocity = self._real_env.get_tcp_velocity()

        # Get camera frames with markers
        timestamp_ns = 0
        try:
            frames, marker3d, timestamp_ns, seq = self._camera.get_frames_with_markers()
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
            self._trigger_camera_safety_stop(f"Camera frame retrieval failed: {e}")

        # Resize images to model input size
        l500_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(l500_image, self._render_height, self._render_width)
        )
        d400_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(d400_image, self._render_height, self._render_width)
        )

        obs = {
            "observation/image": l500_image,
            "observation/wrist_image": d400_image,
            "observation/state": state,
            "prompt": self._prompt,
        }

        # Add tactile data if available (key: xense_1_marker3d -> observation/tactile)
        tactile_data = marker3d.get("xense_1_marker3d")
        if tactile_data is not None and tactile_data.size > 0:
            obs["observation/tactile"] = tactile_data.astype(np.float32)

        meta: dict[str, object] = {
            "recording_snapshot": {
                "frames": dict(frames),
                "marker3d": dict(marker3d),
                "timestamp_ns": int(timestamp_ns),
                "seq": int(seq),
                "state": state.copy(),
                "tcp_velocity": tcp_velocity.copy(),
            },
            "is_human_teaching": bool(self._teaching_mode_active),
        }
        if self._teaching_mode_active:
            segment_id = self._active_teaching_segment_id if self._active_teaching_segment_id is not None else 0
            teaching_step = self._active_teaching_step
            meta["teaching_segment_id"] = int(segment_id)
            meta["teaching_step"] = int(teaching_step)
            self._active_teaching_step += 1
        obs["__openpi"] = meta

        return obs

    def get_recording_frame(self) -> dict:
        """Get raw frames + robot state for recording."""
        state = self._real_env.get_state()
        tcp_pose = np.asarray(state[:7], dtype=np.float32)
        gripper = np.asarray(state[7:8], dtype=np.float32)
        wrench = np.asarray(state[8:14], dtype=np.float32)
        tcp_velocity = self._real_env.get_tcp_velocity()
        action = self._real_env.get_last_target_action()
        if action is None:
            action = np.concatenate([tcp_pose, gripper], axis=0).astype(np.float32)

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
            "control_timestamp": self._latest_control_timestamp,
            "seq": int(seq),
            "tcp_pose": tcp_pose,
            "tcp_velocity": tcp_velocity,
            "wrench": wrench,
            "gripper": gripper,
            "action": action,
            "teaching_segment_id": self._active_teaching_segment_id if self._teaching_mode_active else None,
            "teaching_step": max(self._active_teaching_step - 1, 0) if self._teaching_mode_active else None,
            "is_human_teaching": self.is_teaching_mode,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply a single-step action.

        Args:
            action: Dict with "actions" key containing the action array.
                   Shape: (action_horizon, action_dim) or (action_dim,).
                   Franka public actions use 10D rotate6d pose10.
        """
        actions, pose10 = _normalize_policy_action(action["actions"])
        action["actions"] = pose10.copy()
        action["actions_pose10"] = pose10.copy()

        action_meta = action.get("__openpi")
        control_timestamp = time.time()
        if isinstance(action_meta, dict) and "control_timestamp" in action_meta:
            control_timestamp = float(action_meta["control_timestamp"])
        self._latest_control_timestamp = control_timestamp

        if self._episode_start_time is None:
            self._episode_start_time = control_timestamp

        elapsed = control_timestamp - self._episode_start_time

        now_s = time.perf_counter()
        if self._last_control_step_time_s is not None:
            dt_s = now_s - self._last_control_step_time_s
            if dt_s > 0:
                inst_hz = 1.0 / dt_s
                ema_alpha = 0.2
                if self._control_hz_ema is None:
                    self._control_hz_ema = inst_hz
                else:
                    self._control_hz_ema = (ema_alpha * inst_hz) + ((1.0 - ema_alpha) * self._control_hz_ema)
        self._last_control_step_time_s = now_s

        hz_str = "--" if self._control_hz_ema is None else f"{self._control_hz_ema:.1f}"

        chunk_meta = action.get("__chunk_meta")
        if chunk_meta:
            chunk_idx = chunk_meta.get("chunk_idx", "?")
            chunk_size = chunk_meta.get("chunk_size", "?")
            new_chunk = chunk_meta.get("new_chunk", False) or chunk_meta.get("new_horizon", False)
            infer_started = chunk_meta.get("inference_started", False)
            skipped_steps = chunk_meta.get("skipped_steps", 0)
            flags = []
            if new_chunk:
                flags.append("NEW")
            if infer_started:
                flags.append("INFER")
            if self._teaching_mode_active:
                flags.append("TEACHING")
            flag_str = f" [{','.join(flags)}]" if flags else ""

            infer_str = ""
            if "infer_ms" in chunk_meta:
                infer_ms = chunk_meta["infer_ms"]
                stats = chunk_meta.get("infer_stats", {})
                mean_ms = stats.get("mean_ms", 0)
                infer_str = f" infer={infer_ms:.0f}ms(avg={mean_ms:.0f}ms)"
            skip_str = f" skip={skipped_steps}" if skipped_steps else ""
            mode_str = f" mode={chunk_meta.get('mode')}" if chunk_meta.get("mode") else ""

            print(
                "[openpi] "
                f"step={self._step_count} t={elapsed:.3f}s hz={hz_str}{mode_str} "
                f"chunk={chunk_idx}/{chunk_size}{skip_str}{flag_str}{infer_str}",
                flush=True,
            )
        else:
            teaching_str = " [TEACHING]" if self._teaching_mode_active else ""
            print(f"[openpi] step={self._step_count} t={elapsed:.3f}s hz={hz_str}{teaching_str}", flush=True)

        executed_action = self._real_env.execute_action(actions)
        action["executed_action"] = executed_action.copy()
        self._step_count += 1

    @property
    def step_count(self) -> int:
        """Number of steps executed in current episode."""
        return self._step_count

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since episode start."""
        if self._episode_start_time is None:
            return 0.0
        return time.time() - self._episode_start_time
