"""Real-Time Chunking broker for overlapped inference and execution."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy


@dataclass
class RTCConfig:
    """Configuration for Real-Time Chunking."""

    action_horizon: int = 30
    inference_delay: int = 3  # Actions executed during inference
    execute_horizon: int = 5  # Total actions per iteration
    control_hz: float = 10.0  # Control frequency
    use_action_prefix: bool = True  # Condition realtime inference on executed prefix


@dataclass
class InferenceStats:
    """Statistics for inference timing."""

    last_infer_ms: float = 0.0
    mean_infer_ms: float = 0.0
    max_infer_ms: float = 0.0
    min_infer_ms: float = float("inf")
    total_infer_count: int = 0
    _sum_ms: float = 0.0

    def update(self, infer_ms: float) -> None:
        self.last_infer_ms = infer_ms
        self.total_infer_count += 1
        self._sum_ms += infer_ms
        self.mean_infer_ms = self._sum_ms / self.total_infer_count
        self.max_infer_ms = max(self.max_infer_ms, infer_ms)
        self.min_infer_ms = min(self.min_infer_ms, infer_ms)

    def reset(self) -> None:
        self.last_infer_ms = 0.0
        self.mean_infer_ms = 0.0
        self.max_infer_ms = 0.0
        self.min_infer_ms = float("inf")
        self.total_infer_count = 0
        self._sum_ms = 0.0


class RealTimeChunkBroker(_base_policy.BasePolicy):
    """Broker that enables overlapped inference and execution.

    Unlike ActionChunkBroker which blocks during inference, this broker
    continues executing previous actions while generating new ones.
    Uses prefix-conditioned sampling for temporal consistency.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        config: RTCConfig,
        *,
        infer_fn: Callable | None = None,
    ):
        """Initialize the RealTimeChunkBroker.

        Args:
            policy: The underlying policy (must support infer_realtime).
            config: RTC configuration parameters.
            infer_fn: Optional custom inference function. If None, uses
                policy.infer_realtime if available, else policy.infer.
        """
        self._policy = policy
        self._config = config
        self._infer_fn = infer_fn or self._default_infer_fn

        # Action queue and state
        self._action_queue: deque[np.ndarray] = deque()
        self._current_chunk: np.ndarray | None = None
        self._current_step_meta: list[dict[str, object]] = []
        self._chunk_index: int = 0
        self._last_action: np.ndarray | None = None
        self._infer_count: int = 0
        self._last_new_chunk: bool = False
        self._inference_started_this_step: bool = False
        self._next_horizon_id: int = 0
        self._pending_horizon_meta: dict | None = None
        self._pending_base_chunk: np.ndarray | None = None

        # Threading
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._running = False
        self._pending_obs: dict | None = None
        self._new_chunk_ready = threading.Event()

        # Inference timing
        self._infer_stats = InferenceStats()
        self._last_infer_ms: float = 0.0
        self._trigger_chunk_index: int = 0  # chunk_index when inference was triggered

    def _default_infer_fn(self, obs: dict, action_prefix: np.ndarray | None = None) -> dict:
        """Default inference function."""
        if self._config.use_action_prefix and hasattr(self._policy, "infer_realtime"):
            return self._policy.infer_realtime(obs, action_prefix=action_prefix)
        return self._policy.infer(obs)

    @override
    def infer(self, obs: dict) -> dict:
        """Get next action, triggering inference when needed.

        This is the synchronous interface matching ActionChunkBroker.
        For async operation, use start_async/get_action pattern.
        """
        self._inference_started_this_step = False

        with self._lock:
            # Check if we have actions available (limited by execute_horizon)
            effective_len = self._get_effective_chunk_len()
            if self._current_chunk is not None and self._chunk_index < effective_len:
                chunk_idx = self._chunk_index
                step_meta = dict(self._current_step_meta[chunk_idx])
                action = self._current_chunk[self._chunk_index]
                new_chunk, pending_horizon_meta, pending_base_chunk = self._pop_pending_horizon_payload()
                self._chunk_index += 1
                self._last_action = action.copy()

                # Trigger new inference when remaining <= inference_delay
                remaining = effective_len - self._chunk_index
                if remaining <= self._config.inference_delay:
                    self._trigger_inference(obs)

                output = {
                    "actions": action,
                    "__chunk_meta": self._build_chunk_meta(
                        chunk_idx=chunk_idx,
                        chunk_size=effective_len,
                        new_chunk=new_chunk,
                        inference_started=self._inference_started_this_step,
                        step_meta=step_meta,
                    ),
                }
                if new_chunk and pending_horizon_meta is not None and pending_base_chunk is not None:
                    output["__horizon_meta"] = pending_horizon_meta
                    output["__base_chunk"] = pending_base_chunk
                return output

        # No actions available, must wait for inference
        return self._blocking_infer(obs)

    def _get_effective_chunk_len(self) -> int:
        """Get effective chunk length, limited by execute_horizon."""
        if self._current_chunk is None:
            return 0
        return len(self._current_chunk)

    def _blocking_infer(self, obs: dict) -> dict:
        """Perform blocking inference when no cached actions available."""
        prefix = self._get_executed_prefix() if self._config.use_action_prefix else None
        infer_count = self._register_inference()
        control_timestamp = self._extract_control_timestamp(obs)
        start_time = time.perf_counter()
        result = self._infer_fn(obs, action_prefix=prefix)
        infer_ms = (time.perf_counter() - start_time) * 1000
        self._infer_stats.update(infer_ms)
        self._last_infer_ms = infer_ms

        actions = result.get("actions", result.get("action"))

        if actions is None:
            raise ValueError("Inference result missing 'actions' key")

        base_chunk = np.asarray(actions)
        horizon_id = self._allocate_horizon_id()
        horizon_meta, step_meta = self._build_horizon_payload(
            horizon_id=horizon_id,
            base_chunk=base_chunk,
            control_timestamp=control_timestamp,
            infer_ms=infer_ms,
            infer_count=infer_count,
            trigger_chunk_index=0,
            frames_elapsed=0,
            skip_count=0,
            action_prefix_len=0 if prefix is None else len(prefix),
        )

        with self._lock:
            self._current_chunk = base_chunk[: self._config.execute_horizon].copy()
            self._current_step_meta = step_meta[: self._config.execute_horizon]
            self._chunk_index = 0
            effective_len = self._get_effective_chunk_len()

            # Return first action
            action = self._current_chunk[0]
            self._chunk_index = 1
            self._last_action = action.copy()
            self._last_new_chunk = False

        return {
            "actions": action,
            "__chunk_meta": self._build_chunk_meta(
                chunk_idx=0,
                chunk_size=effective_len,
                new_chunk=True,
                inference_started=True,
                step_meta=self._current_step_meta[0],
            ),
            "__horizon_meta": horizon_meta,
            "__base_chunk": base_chunk.copy(),
            **{k: v for k, v in result.items() if k != "actions"},
        }

    def _get_executed_prefix(self) -> np.ndarray | None:
        """Get recently executed actions as prefix for conditioning."""
        with self._lock:
            if self._current_chunk is None or self._chunk_index == 0:
                return None

            # Return executed portion of current chunk
            prefix_len = min(self._chunk_index, self._config.inference_delay)
            start_idx = self._chunk_index - prefix_len
            return self._current_chunk[start_idx : self._chunk_index].copy()

    def _trigger_inference(self, obs: dict) -> None:
        """Trigger async inference in background."""
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return  # Already running

        self._pending_obs = obs
        self._trigger_chunk_index = self._chunk_index
        self._inference_started_this_step = True
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

    def _inference_worker(self) -> None:
        """Background inference worker."""
        obs = self._pending_obs
        if obs is None:
            return

        prefix = self._get_executed_prefix() if self._config.use_action_prefix else None
        infer_count = self._register_inference()
        control_timestamp = self._extract_control_timestamp(obs)
        start_time = time.perf_counter()
        result = self._infer_fn(obs, action_prefix=prefix)
        infer_ms = (time.perf_counter() - start_time) * 1000
        self._infer_stats.update(infer_ms)
        self._last_infer_ms = infer_ms

        actions = result.get("actions", result.get("action"))

        if actions is not None:
            self._merge_chunk(
                np.asarray(actions),
                control_timestamp=control_timestamp,
                infer_ms=infer_ms,
                infer_count=infer_count,
                action_prefix_len=0 if prefix is None else len(prefix),
            )
            self._new_chunk_ready.set()

    def _merge_chunk(
        self,
        new_chunk: np.ndarray,
        *,
        control_timestamp: float | None,
        infer_ms: float,
        infer_count: int,
        action_prefix_len: int,
    ) -> None:
        """Merge new chunk with existing queue, respecting execute_horizon."""
        base_chunk = np.asarray(new_chunk)
        horizon_id = self._allocate_horizon_id()
        with self._lock:
            frames_elapsed = 0
            skip_count = 0
            if self._current_chunk is not None:
                remaining = self._current_chunk[self._chunk_index :]
                remaining_step_meta = self._current_step_meta[self._chunk_index :]
                frames_elapsed = self._chunk_index - self._trigger_chunk_index
                skip_count = max(0, min(frames_elapsed, len(base_chunk) - 1))
            else:
                remaining = np.empty((0,) + base_chunk.shape[1:], dtype=base_chunk.dtype)
                remaining_step_meta = []

            horizon_meta, step_meta = self._build_horizon_payload(
                horizon_id=horizon_id,
                base_chunk=base_chunk,
                control_timestamp=control_timestamp,
                infer_ms=infer_ms,
                infer_count=infer_count,
                trigger_chunk_index=self._trigger_chunk_index,
                frames_elapsed=frames_elapsed,
                skip_count=skip_count,
                action_prefix_len=action_prefix_len,
            )
            new_suffix = base_chunk[skip_count:]
            new_suffix_meta = step_meta[skip_count:]
            if len(remaining) > 0:
                merged_chunk = np.concatenate([remaining, new_suffix], axis=0)
            else:
                merged_chunk = new_suffix
            merged_step_meta = remaining_step_meta + new_suffix_meta

            self._current_chunk = merged_chunk[: self._config.execute_horizon].copy()
            self._current_step_meta = merged_step_meta[: self._config.execute_horizon]
            self._chunk_index = 0
            self._last_new_chunk = True
            self._pending_horizon_meta = horizon_meta
            self._pending_base_chunk = base_chunk.copy()

    def _pop_pending_horizon_payload(self) -> tuple[bool, dict | None, np.ndarray | None]:
        new_chunk = self._last_new_chunk
        pending_horizon_meta = self._pending_horizon_meta if new_chunk else None
        pending_base_chunk = self._pending_base_chunk.copy() if new_chunk and self._pending_base_chunk is not None else None
        if new_chunk:
            self._last_new_chunk = False
            self._pending_horizon_meta = None
            self._pending_base_chunk = None
        return new_chunk, pending_horizon_meta, pending_base_chunk

    def get_action(self, obs: dict, *, block: bool = True, timeout: float | None = None) -> np.ndarray | None:
        """Get next action for execution.

        Args:
            obs: Current observation.
            block: Whether to block waiting for action.
            timeout: Max wait time in seconds.

        Returns:
            Action array, or None if non-blocking and unavailable.
        """
        with self._lock:
            effective_len = self._get_effective_chunk_len()
            if self._current_chunk is not None and self._chunk_index < effective_len:
                action = self._current_chunk[self._chunk_index]
                self._chunk_index += 1
                self._last_action = action.copy()

                # Trigger async inference when nearing end
                remaining = effective_len - self._chunk_index
                if remaining <= self._config.inference_delay:
                    self._trigger_inference(obs)

                return action

        if not block:
            # Return last action as fallback (position hold)
            return self._last_action.copy() if self._last_action is not None else None

        # Block waiting for inference
        result = self._blocking_infer(obs)
        return result["actions"]

    @override
    def reset(self) -> None:
        """Reset state for new episode."""
        with self._lock:
            self._action_queue.clear()
            self._current_chunk = None
            self._current_step_meta = []
            self._chunk_index = 0
            self._last_action = None
            self._pending_obs = None
            self._new_chunk_ready.clear()
            self._infer_count = 0
            self._last_new_chunk = False
            self._inference_started_this_step = False
            self._infer_stats.reset()
            self._last_infer_ms = 0.0
            self._trigger_chunk_index = 0
            self._next_horizon_id = 0
            self._pending_horizon_meta = None
            self._pending_base_chunk = None

        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def has_pending_actions(self) -> bool:
        """Check if there are actions available without inference."""
        with self._lock:
            effective_len = self._get_effective_chunk_len()
            return self._current_chunk is not None and self._chunk_index < effective_len

    @property
    def remaining_actions(self) -> int:
        """Number of actions remaining in current chunk."""
        with self._lock:
            effective_len = self._get_effective_chunk_len()
            return max(0, effective_len - self._chunk_index)

    @property
    def infer_stats(self) -> InferenceStats:
        return self._infer_stats

    def _register_inference(self) -> int:
        with self._lock:
            self._infer_count += 1
            return self._infer_count

    def _extract_control_timestamp(self, obs: dict) -> float | None:
        meta = obs.get("__openpi")
        if isinstance(meta, dict) and "control_timestamp" in meta:
            return float(meta["control_timestamp"])
        return None

    def _build_horizon_payload(
        self,
        *,
        horizon_id: int,
        base_chunk: np.ndarray,
        control_timestamp: float | None,
        infer_ms: float,
        infer_count: int,
        trigger_chunk_index: int,
        frames_elapsed: int,
        skip_count: int,
        action_prefix_len: int,
    ) -> tuple[dict, list[dict[str, object]]]:
        effective_horizon = min(len(base_chunk), self._config.execute_horizon)
        if control_timestamp is None:
            horizon_start_timestamp = float("nan")
            planned_timestamps = np.full((len(base_chunk),), np.nan, dtype=np.float64)
        else:
            horizon_start_timestamp = float(control_timestamp)
            dt = 1.0 / self._config.control_hz
            planned_timestamps = control_timestamp + (np.arange(len(base_chunk), dtype=np.float64) * dt)

        horizon_meta = {
            "mode": "rtc",
            "horizon_id": int(horizon_id),
            "horizon_start_timestamp": float(horizon_start_timestamp),
            "planned_timestamps": planned_timestamps.copy(),
            "time_base": "control_timestamp",
            "base_chunk": base_chunk.copy(),
            "requested_execute_horizon": int(self._config.execute_horizon),
            "effective_horizon": int(effective_horizon),
            "trigger_chunk_index": int(trigger_chunk_index),
            "frames_elapsed": int(frames_elapsed),
            "skip_count": int(skip_count),
            "used_action_prefix": bool(self._config.use_action_prefix),
            "action_prefix_len": int(action_prefix_len),
            "policy_timing": {
                "infer_ms": float(infer_ms),
                "infer_count": int(infer_count),
            },
        }
        step_meta = [
            {
                "horizon_id": int(horizon_id),
                "source_chunk_idx": int(idx),
                "horizon_start_timestamp": float(horizon_start_timestamp),
                "planned_timestamp": float(planned_timestamps[idx]) if idx < len(planned_timestamps) else float("nan"),
                "time_base": "control_timestamp",
                "infer_count": int(infer_count),
                "infer_ms": float(infer_ms),
                "trigger_chunk_index": int(trigger_chunk_index),
                "frames_elapsed": int(frames_elapsed),
                "skip_count": int(skip_count),
                "used_action_prefix": bool(self._config.use_action_prefix),
                "action_prefix_len": int(action_prefix_len),
            }
            for idx in range(len(base_chunk))
        ]
        return horizon_meta, step_meta

    def _build_chunk_meta(
        self,
        *,
        chunk_idx: int,
        chunk_size: int,
        new_chunk: bool,
        inference_started: bool,
        step_meta: dict[str, object],
    ) -> dict[str, object]:
        return {
            "mode": "rtc",
            "horizon_id": int(step_meta["horizon_id"]),
            "chunk_idx": int(chunk_idx),
            "source_chunk_idx": int(step_meta["source_chunk_idx"]),
            "chunk_size": int(chunk_size),
            "requested_execute_horizon": int(self._config.execute_horizon),
            "effective_horizon": int(chunk_size),
            "new_chunk": bool(new_chunk),
            "infer_count": int(step_meta["infer_count"]),
            "inference_started": bool(inference_started),
            "infer_ms": float(step_meta["infer_ms"]),
            "infer_stats": {
                "mean_ms": self._infer_stats.mean_infer_ms,
                "max_ms": self._infer_stats.max_infer_ms,
                "min_ms": self._infer_stats.min_infer_ms if self._infer_stats.min_infer_ms != float("inf") else 0.0,
            },
            "horizon_start_timestamp": float(step_meta["horizon_start_timestamp"]),
            "planned_timestamp": float(step_meta["planned_timestamp"]),
            "time_base": str(step_meta["time_base"]),
            "trigger_chunk_index": int(step_meta["trigger_chunk_index"]),
            "frames_elapsed": int(step_meta["frames_elapsed"]),
            "skip_count": int(step_meta["skip_count"]),
            "used_action_prefix": bool(step_meta["used_action_prefix"]),
            "action_prefix_len": int(step_meta["action_prefix_len"]),
        }

    def _allocate_horizon_id(self) -> int:
        with self._lock:
            horizon_id = self._next_horizon_id
            self._next_horizon_id += 1
            return horizon_id
