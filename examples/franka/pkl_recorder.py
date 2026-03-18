"""Episode-level PKL recorder for Franka evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from residual_policy.action_repr import pose8_to_pose10
from residual_policy.action_repr import pose10_to_pose8

from examples.franka import env as _env

logger = logging.getLogger(__name__)
_ACTION_DIM = 10
_EXECUTED_ACTION_DIM = 8
_POSE10_DIM = 10
_CORRECTED_ACTION_SHIFT = 10


@dataclass(frozen=True)
class RecorderConfig:
    record_dir: Path
    control_hz: float
    prompt: str = ""


def _downsample_half(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        return np.asarray(image)
    return np.ascontiguousarray(image[::2, ::2, :])


def _empty_marker3d() -> np.ndarray:
    return np.zeros((0, 0, 3), dtype=np.float32)


def _as_action_vector(value: object, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            return np.zeros(_ACTION_DIM, dtype=np.float32)
        return np.asarray(default, dtype=np.float32).reshape(-1).copy()

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape == (_ACTION_DIM,):
        return arr.copy()
    if arr.shape == (_EXECUTED_ACTION_DIM,):
        return pose8_to_pose10(arr)
    arr = arr.reshape(-1)
    if arr.shape == (_ACTION_DIM,):
        return arr.copy()
    if arr.shape == (_EXECUTED_ACTION_DIM,):
        return pose8_to_pose10(arr)
    raise ValueError(f"Expected canonical action shape (10,) or executable action shape (8,), got {arr.shape}")


def _as_executed_action_vector(value: object, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            return np.zeros(_EXECUTED_ACTION_DIM, dtype=np.float32)
        return np.asarray(default, dtype=np.float32).reshape(-1).copy()

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape == (_EXECUTED_ACTION_DIM,):
        return arr.copy()
    if arr.shape == (_ACTION_DIM,):
        return pose10_to_pose8(arr)
    arr = arr.reshape(-1)
    if arr.shape == (_EXECUTED_ACTION_DIM,):
        return arr.copy()
    if arr.shape == (_ACTION_DIM,):
        return pose10_to_pose8(arr)
    raise ValueError(f"Expected executed action shape (8,) or canonical action shape (10,), got {arr.shape}")


def _as_pose10_vector(value: object, *, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            return np.zeros(_POSE10_DIM, dtype=np.float32)
        arr = np.asarray(default, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape == (_POSE10_DIM,):
        return arr.copy()
    if arr.shape == (_ACTION_DIM,):
        return pose8_to_pose10(arr)
    arr = arr.reshape(-1)
    if arr.shape == (_POSE10_DIM,):
        return arr.copy()
    if arr.shape == (_ACTION_DIM,):
        return pose8_to_pose10(arr)
    raise ValueError(f"Expected action shape (8,) or (10,), got {arr.shape}")


def _quaternion_distance_rad(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_q = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_q = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = float(np.linalg.norm(lhs_q))
    rhs_norm = float(np.linalg.norm(rhs_q))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    dot = float(np.dot(lhs_q / lhs_norm, rhs_q / rhs_norm))
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _build_action_deltas(
    *,
    raw_action: np.ndarray,
    base_action: np.ndarray,
    executed_action: np.ndarray,
) -> dict[str, float]:
    raw_action_pose8 = pose10_to_pose8(raw_action)
    base_action_pose8 = pose10_to_pose8(base_action)
    return {
        "raw_minus_base_translation_m": float(np.linalg.norm(raw_action_pose8[:3] - base_action_pose8[:3])),
        "raw_minus_base_rotation_rad": _quaternion_distance_rad(raw_action_pose8[3:7], base_action_pose8[3:7]),
        "executed_minus_raw_translation_m": float(np.linalg.norm(executed_action[:3] - raw_action_pose8[:3])),
        "executed_minus_raw_rotation_rad": _quaternion_distance_rad(executed_action[3:7], raw_action_pose8[3:7]),
        "executed_minus_base_translation_m": float(np.linalg.norm(executed_action[:3] - base_action_pose8[:3])),
        "executed_minus_base_rotation_rad": _quaternion_distance_rad(executed_action[3:7], base_action_pose8[3:7]),
    }


def _extract_policy_timing(action: dict[str, Any]) -> dict[str, int | float]:
    timing = action.get("policy_timing")
    if not isinstance(timing, dict):
        return {}
    extracted: dict[str, int | float] = {}
    for key, value in timing.items():
        if isinstance(value, (np.integer, int)):
            extracted[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            extracted[key] = float(value)
    return extracted


def _extract_teaching_metadata(obs_meta: dict[str, Any]) -> tuple[bool, int | None, int | None]:
    is_human_teaching = bool(obs_meta.get("is_human_teaching", False))
    if not is_human_teaching:
        return False, None, None

    segment_id = obs_meta.get("teaching_segment_id")
    teaching_step = obs_meta.get("teaching_step")
    return (
        True,
        None if segment_id is None else int(segment_id),
        None if teaching_step is None else int(teaching_step),
    )


class EpisodePklRecorder(_subscriber.Subscriber):
    """Runtime subscriber that records episode frames to a PKL file."""

    def __init__(self, environment: _env.FrankaEnvironment, config: RecorderConfig) -> None:
        self._env = environment
        self._config = config

        self._episode_index = self._resolve_start_index()
        self._frame_index = 0
        self._frames: list[dict[str, Any]] = []
        self._policy_steps: list[dict[str, Any]] = []
        self._policy_horizons: list[dict[str, Any]] = []
        self._human_teaching_steps: list[dict[str, Any]] = []
        self._episode_start_control_timestamp: float | None = None

    def on_episode_start(self) -> None:
        self._episode_index += 1
        self._frame_index = 0
        self._frames = []
        self._policy_steps = []
        self._policy_horizons = []
        self._human_teaching_steps = []
        self._episode_start_control_timestamp = None

    def _resolve_start_index(self) -> int:
        output_dir = Path(self._config.record_dir)
        if not output_dir.exists():
            return -1
        max_idx = -1
        for path in output_dir.glob("episode_*.pkl"):
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

    def on_step(self, observation: dict, action: dict) -> None:
        chunk_meta = action.get("__chunk_meta")
        obs_meta = observation.get("__openpi")
        if not isinstance(obs_meta, dict):
            return

        control_timestamp = float(obs_meta["control_timestamp"])
        snapshot = obs_meta.get("recording_snapshot")
        if not isinstance(snapshot, dict):
            logger.warning("Missing recording snapshot in observation metadata; skipping recorder frame")
            return
        raw_action = _as_action_vector(action.get("actions"))
        base_action = _as_action_vector(action.get("base_action"), default=raw_action)
        executed_action = _as_executed_action_vector(
            action.get("executed_action"),
            default=pose10_to_pose8(raw_action),
        )
        raw_action_pose10 = _as_pose10_vector(action.get("actions_pose10"), default=raw_action)
        base_action_pose10 = _as_pose10_vector(action.get("base_action_pose10"), default=base_action)
        residual_action_pose10 = _as_pose10_vector(action.get("residual_action_pose10"))
        action_deltas = _build_action_deltas(
            raw_action=raw_action,
            base_action=base_action,
            executed_action=executed_action,
        )
        policy_timing = _extract_policy_timing(action)
        is_human_teaching, teaching_segment_id, teaching_step = _extract_teaching_metadata(obs_meta)
        self._frames.append(
            self._build_record(
                control_timestamp=control_timestamp,
                snapshot=snapshot,
                raw_action=raw_action,
                base_action=base_action,
                executed_action=executed_action,
                is_human_teaching=is_human_teaching,
                teaching_segment_id=teaching_segment_id,
                teaching_step=teaching_step,
            )
        )
        frame_index = int(self._frames[-1]["frame_index"])
        if is_human_teaching:
            self._human_teaching_steps.append(
                {
                    "frame_index": frame_index,
                    "control_timestamp": control_timestamp,
                    "teaching_segment_id": -1 if teaching_segment_id is None else teaching_segment_id,
                    "teaching_step": -1 if teaching_step is None else teaching_step,
                }
            )

        mode = chunk_meta.get("mode") if isinstance(chunk_meta, dict) else None
        if not isinstance(chunk_meta, dict) or mode not in {"cr_dagger_baseline", "rtc"}:
            return

        skipped_steps = int(chunk_meta.get("skipped_steps", chunk_meta.get("skip_count", 0)))
        self._policy_steps.append(
            {
                "frame_index": frame_index,
                "control_timestamp": control_timestamp,
                "episode_step": int(obs_meta["episode_step"]),
                "horizon_id": int(chunk_meta["horizon_id"]),
                "chunk_idx": int(chunk_meta["chunk_idx"]),
                "skipped_steps": skipped_steps,
                "base_action": base_action.copy(),
                "raw_action": raw_action.copy(),
                "base_action_pose10": base_action_pose10.copy(),
                "raw_action_pose10": raw_action_pose10.copy(),
                "residual_action_pose10": residual_action_pose10.copy(),
                "executed_action": executed_action.copy(),
                "action_deltas": dict(action_deltas),
                "policy_timing": dict(policy_timing),
                "chunk_meta": {
                    "mode": mode,
                    "horizon_id": int(chunk_meta["horizon_id"]),
                    "chunk_idx": int(chunk_meta["chunk_idx"]),
                    "chunk_size": int(chunk_meta["chunk_size"]),
                    "requested_execute_horizon": int(chunk_meta["requested_execute_horizon"]),
                    "effective_horizon": int(chunk_meta["effective_horizon"]),
                    "new_horizon": bool(chunk_meta.get("new_horizon", chunk_meta.get("new_chunk", False))),
                    "skipped_steps": skipped_steps,
                    "infer_count": int(chunk_meta.get("infer_count", 0)),
                    "infer_ms": float(chunk_meta.get("infer_ms", 0.0)),
                    "horizon_start_timestamp": float(chunk_meta["horizon_start_timestamp"]),
                    "planned_timestamp": float(chunk_meta["planned_timestamp"]),
                    "time_base": chunk_meta.get("time_base", "control_timestamp"),
                    **(
                        {
                            "source_chunk_idx": int(chunk_meta.get("source_chunk_idx", chunk_meta["chunk_idx"])),
                            "trigger_chunk_index": int(chunk_meta.get("trigger_chunk_index", 0)),
                            "frames_elapsed": int(chunk_meta.get("frames_elapsed", 0)),
                            "skip_count": int(chunk_meta.get("skip_count", 0)),
                            "used_action_prefix": bool(chunk_meta.get("used_action_prefix", False)),
                            "action_prefix_len": int(chunk_meta.get("action_prefix_len", 0)),
                        }
                        if mode == "rtc"
                        else {}
                    ),
                },
            }
        )

        horizon_meta = action.get("__horizon_meta")
        base_chunk = action.get("__base_chunk")
        horizon_mode = horizon_meta.get("mode") if isinstance(horizon_meta, dict) else None
        if isinstance(horizon_meta, dict) and base_chunk is not None and horizon_mode in {"cr_dagger_baseline", "rtc"}:
            self._policy_horizons.append(
                {
                    "horizon_id": int(horizon_meta["horizon_id"]),
                    "horizon_start_timestamp": float(horizon_meta["horizon_start_timestamp"]),
                    "planned_timestamps": np.asarray(horizon_meta["planned_timestamps"], dtype=np.float64).copy(),
                    "time_base": horizon_meta.get("time_base", "control_timestamp"),
                    "base_chunk": np.asarray(base_chunk, dtype=np.float32).copy(),
                    "requested_execute_horizon": int(horizon_meta["requested_execute_horizon"]),
                    "effective_horizon": int(horizon_meta["effective_horizon"]),
                    "policy_timing": {
                        "infer_ms": float(horizon_meta.get("policy_timing", {}).get("infer_ms", 0.0)),
                        "infer_count": int(horizon_meta.get("policy_timing", {}).get("infer_count", 0)),
                    },
                    **(
                        {
                            "mode": horizon_mode,
                            "trigger_chunk_index": int(horizon_meta.get("trigger_chunk_index", 0)),
                            "frames_elapsed": int(horizon_meta.get("frames_elapsed", 0)),
                            "skip_count": int(horizon_meta.get("skip_count", 0)),
                            "used_action_prefix": bool(horizon_meta.get("used_action_prefix", False)),
                            "action_prefix_len": int(horizon_meta.get("action_prefix_len", 0)),
                        }
                        if horizon_mode == "rtc"
                        else {"mode": horizon_mode}
                    ),
                }
            )

    def on_episode_end(self) -> None:
        self._finalize_human_corrections()

        output_dir = Path(self._config.record_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"episode_{self._episode_index:03d}.pkl"

        payload = {
            "version": 1,
            "episode_index": self._episode_index,
            "prompt": self._config.prompt,
            "fps": float(self._config.control_hz),
            "frames": self._frames,
        }
        if self._policy_steps:
            payload["policy_steps"] = self._policy_steps
        if self._policy_horizons:
            payload["policy_horizons"] = self._policy_horizons
        if self._human_teaching_steps:
            payload["human_teaching_steps"] = self._human_teaching_steps

        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved episode PKL: %s (frames=%d)", output_path, len(self._frames))

    def _build_record(
        self,
        *,
        control_timestamp: float,
        snapshot: dict[str, Any],
        raw_action: np.ndarray,
        base_action: np.ndarray,
        executed_action: np.ndarray,
        is_human_teaching: bool,
        teaching_segment_id: int | None,
        teaching_step: int | None,
    ) -> dict[str, Any]:
        frames = snapshot.get("frames", {})
        marker3d = snapshot.get("marker3d", {})
        state = np.asarray(snapshot.get("state", np.zeros(14, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if state.size < 14:
            padded_state = np.zeros(14, dtype=np.float32)
            padded_state[: state.size] = state
            state = padded_state
        tcp_pose = state[:7].copy()
        gripper = state[7:8].copy()
        wrench = state[8:14].copy()
        tcp_velocity = np.asarray(snapshot.get("tcp_velocity", np.zeros(6, dtype=np.float32)), dtype=np.float32)

        images = {
            "l500": _downsample_half(frames.get("l500_rgb")),
            "d400": _downsample_half(frames.get("d400_rgb")),
            "xense_1": _downsample_half(frames.get("xense_1_rgb")),
        }

        markers = {
            "xense_1": marker3d.get("xense_1_marker3d", _empty_marker3d()),
        }

        timestamp_ns = int(snapshot.get("timestamp_ns", 0))
        if self._episode_start_control_timestamp is None:
            self._episode_start_control_timestamp = control_timestamp
        timestamp = float(control_timestamp - self._episode_start_control_timestamp)
        corrected_action = base_action.copy()
        corrected_action_valid = not is_human_teaching
        record = {
            "timestamp": timestamp,
            "timestamp_ns": timestamp_ns,
            "control_timestamp": control_timestamp,
            "seq": int(snapshot.get("seq", -1)),
            "frame_index": self._frame_index,
            "images": images,
            "marker3d": markers,
            "tcp_pose": tcp_pose,
            "tcp_velocity": tcp_velocity,
            "wrench": wrench,
            "gripper": gripper,
            "action": np.asarray(raw_action, dtype=np.float32).copy(),
            "executed_action": np.asarray(executed_action, dtype=np.float32).copy(),
            "base_action": np.asarray(base_action, dtype=np.float32).copy(),
            "corrected_action": corrected_action,
            "corrected_action_valid": corrected_action_valid,
            "teaching_segment_id": teaching_segment_id,
            "teaching_step": teaching_step,
            "is_human_teaching": is_human_teaching,
        }
        self._frame_index += 1
        return record

    def _project_frame_state_to_action(self, frame: dict[str, Any]) -> np.ndarray:
        tcp_pose = np.asarray(frame.get("tcp_pose", np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
        gripper = np.asarray(frame.get("gripper", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)
        pose8 = np.concatenate([tcp_pose[:7], gripper[:1]], axis=0).astype(np.float32)
        return pose8_to_pose10(pose8)

    def _finalize_human_corrections(self) -> None:
        if not self._human_teaching_steps:
            return

        steps_by_segment: dict[int, list[dict[str, Any]]] = {}
        for step in self._human_teaching_steps:
            segment_id = int(step.get("teaching_segment_id", -1))
            steps_by_segment.setdefault(segment_id, []).append(step)

        for segment_steps in steps_by_segment.values():
            segment_steps.sort(key=lambda item: int(item["frame_index"]))
            for idx, step in enumerate(segment_steps):
                frame_index = int(step["frame_index"])
                frame = self._frames[frame_index]
                future_idx = idx + _CORRECTED_ACTION_SHIFT
                corrected_valid = future_idx < len(segment_steps)
                if corrected_valid:
                    future_frame_index = int(segment_steps[future_idx]["frame_index"])
                    corrected_action = self._project_frame_state_to_action(self._frames[future_frame_index])
                else:
                    corrected_action = np.asarray(frame["corrected_action"], dtype=np.float32)

                frame["corrected_action"] = np.asarray(corrected_action, dtype=np.float32)
                frame["corrected_action_valid"] = corrected_valid
