"""Local gripper state interpolation utilities."""

from __future__ import annotations

import time
from typing import Optional


class GripperStateInterpolator:
    """Interpolate gripper state between 0.0 (open) and 1.0 (closed)."""

    def __init__(self, interpolation_duration: float = 1.4) -> None:
        self.interpolation_duration = interpolation_duration
        self._target_state: float = 0.0
        self._state_before_change: float = 0.0
        self._change_time: Optional[float] = None
        self._early_terminated: bool = False

    def set_target(self, target: float, current_time: float) -> None:
        if self._change_time is not None:
            self._state_before_change = self._compute_interpolated_state(current_time)
        else:
            self._state_before_change = self._target_state

        self._target_state = target
        self._change_time = current_time
        self._early_terminated = False

    def mark_early_termination(self) -> None:
        self._early_terminated = True

    def get_state(self, current_time: float) -> float:
        if self._change_time is None:
            return self._target_state
        if self._early_terminated:
            return self._target_state
        return self._compute_interpolated_state(current_time)

    def _compute_interpolated_state(self, current_time: float) -> float:
        if self._change_time is None:
            return self._target_state

        elapsed = current_time - self._change_time
        if elapsed >= self.interpolation_duration:
            return self._target_state

        progress = elapsed / self.interpolation_duration
        return self._state_before_change + (self._target_state - self._state_before_change) * progress

    @property
    def is_interpolating(self) -> bool:
        if self._change_time is None:
            return False
        if self._early_terminated:
            return False
        return (time.time() - self._change_time) < self.interpolation_duration
