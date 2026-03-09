"""CR-Dagger-style chunk broker for observation-anchored horizon execution."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy


_MODE = "cr_dagger_baseline"
_TIME_BASE = "control_timestamp"
_STEP_EPSILON = 1e-6


@dataclass(frozen=True)
class CrDaggerChunkBrokerConfig:
    """Configuration for CR-Dagger-style cached horizon execution."""

    action_horizon: int
    execute_horizon: int = 10
    max_skip_steps: int = 2
    control_hz: float = 10.0

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}")
        if self.execute_horizon > self.action_horizon:
            raise ValueError(
                "CR-Dagger execute_horizon exceeds the known model action horizon "
                f"({self.execute_horizon} > {self.action_horizon})"
            )
        if self.max_skip_steps < 0:
            raise ValueError(f"max_skip_steps must be non-negative, got {self.max_skip_steps}")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {self.control_hz}")


class CrDaggerLagExceeded(RuntimeError):
    """Raised when control lag exceeds the configured safety threshold."""

    def __init__(
        self,
        *,
        horizon_id: int,
        logical_step: int,
        next_expected_step: int,
        skip_count: int,
        max_skip_steps: int,
    ) -> None:
        self.horizon_id = horizon_id
        self.logical_step = logical_step
        self.next_expected_step = next_expected_step
        self.skip_count = skip_count
        self.max_skip_steps = max_skip_steps
        super().__init__(
            "CR-Dagger control lag exceeded safety threshold "
            f"(horizon_id={horizon_id}, logical_step={logical_step}, "
            f"next_expected_step={next_expected_step}, skip_count={skip_count}, "
            f"max_skip_steps={max_skip_steps})"
        )


class CrDaggerChunkBroker(_base_policy.BasePolicy):
    """Broker that caches one base-policy chunk per execution horizon."""

    def __init__(self, policy: _base_policy.BasePolicy, config: CrDaggerChunkBrokerConfig) -> None:
        self._policy = policy
        self._config = config
        self._dt = 1.0 / self._config.control_hz
        self._clear_state()

    @override
    def infer(self, obs: dict) -> dict:
        control_timestamp = self._get_control_timestamp(obs)
        if self._base_chunk is None:
            return self._start_new_horizon(obs, control_timestamp)

        logical_step = self._logical_step(control_timestamp)
        next_expected_step = self._next_expected_step()
        skip_count = max(0, logical_step - next_expected_step)
        if skip_count > self._config.max_skip_steps:
            raise CrDaggerLagExceeded(
                horizon_id=self._horizon_id,
                logical_step=logical_step,
                next_expected_step=next_expected_step,
                skip_count=skip_count,
                max_skip_steps=self._config.max_skip_steps,
            )

        if logical_step >= self._effective_horizon:
            return self._start_new_horizon(obs, control_timestamp)

        step_idx = min(max(logical_step, 0), self._effective_horizon - 1)
        self._chunk_idx = step_idx
        return self._build_action_output(step_idx=step_idx, skipped_steps=skip_count, new_horizon=False)

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._clear_state()

    def _clear_state(self) -> None:
        self._base_chunk: np.ndarray | None = None
        self._planned_timestamps: np.ndarray | None = None
        self._horizon_start_timestamp: float | None = None
        self._effective_horizon = 0
        self._chunk_idx = -1
        self._horizon_id = -1
        self._infer_count = 0
        self._last_infer_ms = 0.0

    def _get_control_timestamp(self, obs: dict) -> float:
        meta = obs.get("__openpi")
        if not isinstance(meta, dict) or "control_timestamp" not in meta:
            raise ValueError("CrDaggerChunkBroker requires observation['__openpi']['control_timestamp']")
        return float(meta["control_timestamp"])

    def _logical_step(self, control_timestamp: float) -> int:
        if self._horizon_start_timestamp is None:
            return 0
        elapsed = max(0.0, control_timestamp - self._horizon_start_timestamp)
        return int(np.floor((elapsed / self._dt) + _STEP_EPSILON))

    def _next_expected_step(self) -> int:
        return max(self._chunk_idx + 1, 0)

    def _start_new_horizon(self, obs: dict, control_timestamp: float) -> dict:
        self._infer_count += 1
        infer_start = time.perf_counter()
        result = self._policy.infer(obs)
        self._last_infer_ms = (time.perf_counter() - infer_start) * 1000.0

        actions = result.get("actions", result.get("action"))
        if actions is None:
            raise ValueError("Inference result missing 'actions' key")

        base_chunk = np.asarray(actions)
        if base_chunk.ndim == 1:
            base_chunk = base_chunk[None, :]
        if base_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with ndim 1 or 2, got shape {base_chunk.shape}")
        if len(base_chunk) == 0:
            raise ValueError("Inference returned an empty action chunk")

        self._horizon_id += 1
        self._base_chunk = base_chunk
        self._effective_horizon = min(self._config.execute_horizon, len(base_chunk))
        self._horizon_start_timestamp = control_timestamp
        self._planned_timestamps = control_timestamp + (np.arange(self._effective_horizon, dtype=np.float64) * self._dt)
        self._chunk_idx = 0

        output = self._build_action_output(step_idx=0, skipped_steps=0, new_horizon=True)
        output["__base_chunk"] = self._base_chunk.copy()
        output["__horizon_meta"] = {
            "mode": _MODE,
            "horizon_id": self._horizon_id,
            "horizon_start_timestamp": float(self._horizon_start_timestamp),
            "planned_timestamps": self._planned_timestamps.copy(),
            "time_base": _TIME_BASE,
            "base_chunk": self._base_chunk.copy(),
            "requested_execute_horizon": self._config.execute_horizon,
            "effective_horizon": self._effective_horizon,
            "policy_timing": {
                "infer_ms": self._last_infer_ms,
                "infer_count": self._infer_count,
            },
        }
        return output

    def _build_action_output(self, *, step_idx: int, skipped_steps: int, new_horizon: bool) -> dict:
        if self._base_chunk is None or self._planned_timestamps is None or self._horizon_start_timestamp is None:
            raise RuntimeError("CR-Dagger horizon state is not initialized")
        return {
            "actions": self._base_chunk[step_idx].copy(),
            "__chunk_meta": {
                "mode": _MODE,
                "horizon_id": self._horizon_id,
                "chunk_idx": step_idx,
                "chunk_size": self._effective_horizon,
                "requested_execute_horizon": self._config.execute_horizon,
                "effective_horizon": self._effective_horizon,
                "new_horizon": new_horizon,
                "skipped_steps": skipped_steps,
                "infer_count": self._infer_count,
                "infer_ms": self._last_infer_ms,
                "horizon_start_timestamp": float(self._horizon_start_timestamp),
                "planned_timestamp": float(self._planned_timestamps[step_idx]),
                "time_base": _TIME_BASE,
            },
        }
