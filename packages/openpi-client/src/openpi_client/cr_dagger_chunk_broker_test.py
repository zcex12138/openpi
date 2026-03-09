import numpy as np
import pytest

from openpi_client import base_policy as _base_policy
from openpi_client.cr_dagger_chunk_broker import CrDaggerChunkBroker
from openpi_client.cr_dagger_chunk_broker import CrDaggerChunkBrokerConfig
from openpi_client.cr_dagger_chunk_broker import CrDaggerLagExceeded


def _make_obs(control_timestamp: float) -> dict:
    return {"__openpi": {"control_timestamp": control_timestamp}}


class _DummyPolicy(_base_policy.BasePolicy):
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.calls: list[float] = []
        self.reset_calls = 0

    def infer(self, obs: dict) -> dict:
        self.calls.append(float(obs["__openpi"]["control_timestamp"]))
        idx = min(len(self.calls) - 1, len(self._chunks) - 1)
        return {"actions": self._chunks[idx].copy()}

    def reset(self) -> None:
        self.reset_calls += 1


def test_first_horizon_inference_is_observation_anchored() -> None:
    chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    policy = _DummyPolicy([chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=4, execute_horizon=3, max_skip_steps=2, control_hz=10.0),
    )

    first = broker.infer(_make_obs(100.0))
    second = broker.infer(_make_obs(100.1))

    assert len(policy.calls) == 1
    np.testing.assert_array_equal(first["actions"], chunk[0])
    np.testing.assert_array_equal(second["actions"], chunk[1])
    assert first["__chunk_meta"]["new_horizon"] is True
    assert second["__chunk_meta"]["new_horizon"] is False
    assert first["__chunk_meta"]["horizon_start_timestamp"] == pytest.approx(100.0)
    np.testing.assert_allclose(first["__horizon_meta"]["planned_timestamps"], np.array([100.0, 100.1, 100.2]))


def test_effective_horizon_truncates_short_runtime_chunk() -> None:
    short_chunk = np.arange(6, dtype=np.float32).reshape(2, 3)
    policy = _DummyPolicy([short_chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=4, execute_horizon=4, max_skip_steps=2, control_hz=10.0),
    )

    result = broker.infer(_make_obs(5.0))

    assert result["__chunk_meta"]["requested_execute_horizon"] == 4
    assert result["__chunk_meta"]["effective_horizon"] == 2
    np.testing.assert_array_equal(result["__base_chunk"], short_chunk)
    np.testing.assert_allclose(result["__horizon_meta"]["planned_timestamps"], np.array([5.0, 5.1]))


def test_execute_horizon_validation_fails_fast() -> None:
    with pytest.raises(ValueError, match="execute_horizon exceeds the known model action horizon"):
        CrDaggerChunkBrokerConfig(action_horizon=4, execute_horizon=5, max_skip_steps=2, control_hz=10.0)


def test_small_lag_skips_stale_steps_but_keeps_horizon() -> None:
    chunk = np.arange(15, dtype=np.float32).reshape(5, 3)
    policy = _DummyPolicy([chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=5, execute_horizon=5, max_skip_steps=2, control_hz=10.0),
    )

    broker.infer(_make_obs(0.0))
    result = broker.infer(_make_obs(0.25))

    assert len(policy.calls) == 1
    assert result["__chunk_meta"]["chunk_idx"] == 2
    assert result["__chunk_meta"]["skipped_steps"] == 1
    np.testing.assert_array_equal(result["actions"], chunk[2])


def test_excessive_lag_raises_safety_stop() -> None:
    chunk = np.arange(15, dtype=np.float32).reshape(5, 3)
    policy = _DummyPolicy([chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=5, execute_horizon=5, max_skip_steps=1, control_hz=10.0),
    )

    broker.infer(_make_obs(0.0))

    with pytest.raises(CrDaggerLagExceeded, match="skip_count=2"):
        broker.infer(_make_obs(0.3))


def test_horizon_rollover_reinfers_after_execution_window() -> None:
    first_chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    second_chunk = np.arange(100, 112, dtype=np.float32).reshape(4, 3)
    policy = _DummyPolicy([first_chunk, second_chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=4, execute_horizon=3, max_skip_steps=2, control_hz=10.0),
    )

    broker.infer(_make_obs(0.0))
    broker.infer(_make_obs(0.1))
    broker.infer(_make_obs(0.2))
    rollover = broker.infer(_make_obs(0.3))

    assert len(policy.calls) == 2
    assert rollover["__chunk_meta"]["horizon_id"] == 1
    assert rollover["__chunk_meta"]["new_horizon"] is True
    np.testing.assert_array_equal(rollover["actions"], second_chunk[0])


def test_reset_clears_cached_horizon_state() -> None:
    chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    policy = _DummyPolicy([chunk, chunk])
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=4, execute_horizon=3, max_skip_steps=2, control_hz=10.0),
    )

    broker.infer(_make_obs(1.0))
    broker.reset()
    result = broker.infer(_make_obs(2.0))

    assert policy.reset_calls == 1
    assert result["__chunk_meta"]["horizon_id"] == 0
    assert result["__chunk_meta"]["infer_count"] == 1
