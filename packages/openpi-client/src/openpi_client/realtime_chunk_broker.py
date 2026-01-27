"""Real-Time Chunking broker for overlapped inference and execution."""

from __future__ import annotations

import threading
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
        self._chunk_index: int = 0
        self._last_action: np.ndarray | None = None

        # Threading
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._running = False
        self._pending_obs: dict | None = None
        self._new_chunk_ready = threading.Event()

    def _default_infer_fn(
        self, obs: dict, action_prefix: np.ndarray | None = None
    ) -> dict:
        """Default inference function."""
        if hasattr(self._policy, "infer_realtime"):
            return self._policy.infer_realtime(obs, action_prefix=action_prefix)
        return self._policy.infer(obs)

    @override
    def infer(self, obs: dict) -> dict:
        """Get next action, triggering inference when needed.

        This is the synchronous interface matching ActionChunkBroker.
        For async operation, use start_async/get_action pattern.
        """
        with self._lock:
            # Check if we have actions available
            if self._current_chunk is not None and self._chunk_index < len(self._current_chunk):
                action = self._current_chunk[self._chunk_index]
                self._chunk_index += 1
                self._last_action = action.copy()

                # Trigger new inference when approaching end
                remaining = len(self._current_chunk) - self._chunk_index
                if remaining <= self._config.inference_delay:
                    self._trigger_inference(obs)

                return {"actions": action}

        # No actions available, must wait for inference
        return self._blocking_infer(obs)

    def _blocking_infer(self, obs: dict) -> dict:
        """Perform blocking inference when no cached actions available."""
        # Get prefix from recently executed actions
        prefix = self._get_executed_prefix()

        result = self._infer_fn(obs, action_prefix=prefix)
        actions = result.get("actions", result.get("action"))

        if actions is None:
            raise ValueError("Inference result missing 'actions' key")

        with self._lock:
            self._current_chunk = np.asarray(actions)
            self._chunk_index = 0

            # Return first action
            action = self._current_chunk[0]
            self._chunk_index = 1
            self._last_action = action.copy()

        return {"actions": action, **{k: v for k, v in result.items() if k != "actions"}}

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
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

    def _inference_worker(self) -> None:
        """Background inference worker."""
        obs = self._pending_obs
        if obs is None:
            return

        prefix = self._get_executed_prefix()
        result = self._infer_fn(obs, action_prefix=prefix)
        actions = result.get("actions", result.get("action"))

        if actions is not None:
            self._merge_chunk(np.asarray(actions))
            self._new_chunk_ready.set()

    def _merge_chunk(self, new_chunk: np.ndarray) -> None:
        """Merge new chunk with existing queue."""
        with self._lock:
            # Discard executed portion, keep remainder + new chunk suffix
            if self._current_chunk is not None:
                remaining = self._current_chunk[self._chunk_index :]
                # Use new chunk from inference_delay onwards
                new_suffix = new_chunk[self._config.inference_delay :]
                self._current_chunk = np.concatenate([remaining, new_suffix], axis=0)
                self._chunk_index = 0
            else:
                self._current_chunk = new_chunk
                self._chunk_index = 0

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
            if self._current_chunk is not None and self._chunk_index < len(self._current_chunk):
                action = self._current_chunk[self._chunk_index]
                self._chunk_index += 1
                self._last_action = action.copy()

                # Trigger async inference when nearing end
                remaining = len(self._current_chunk) - self._chunk_index
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
            self._chunk_index = 0
            self._last_action = None
            self._pending_obs = None
            self._new_chunk_ready.clear()

        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def has_pending_actions(self) -> bool:
        """Check if there are actions available without inference."""
        with self._lock:
            return self._current_chunk is not None and self._chunk_index < len(self._current_chunk)

    @property
    def remaining_actions(self) -> int:
        """Number of actions remaining in current chunk."""
        with self._lock:
            if self._current_chunk is None:
                return 0
            return len(self._current_chunk) - self._chunk_index
