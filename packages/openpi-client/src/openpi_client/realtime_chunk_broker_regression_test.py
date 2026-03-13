import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client.realtime_chunk_broker import RTCConfig
from openpi_client.realtime_chunk_broker import RealTimeChunkBroker


class _DummyPolicy(_base_policy.BasePolicy):
    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self.infer_calls = 0

    def infer(self, obs: dict) -> dict:
        self.infer_calls += 1
        return {"actions": self._chunk.copy()}


class _PrefixAwarePolicy(_base_policy.BasePolicy):
    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self.infer_calls = 0
        self.infer_realtime_calls = 0
        self.last_action_prefix = None

    def infer(self, obs: dict) -> dict:
        self.infer_calls += 1
        return {"actions": self._chunk.copy()}

    def infer_realtime(self, obs: dict, *, action_prefix=None) -> dict:
        self.infer_realtime_calls += 1
        self.last_action_prefix = action_prefix
        return {"actions": self._chunk.copy()}


def test_rtc_mode_still_reuses_cached_chunk_when_enabled() -> None:
    chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    policy = _DummyPolicy(chunk)
    broker = RealTimeChunkBroker(
        policy=policy,
        config=RTCConfig(action_horizon=4, inference_delay=0, execute_horizon=4, control_hz=10.0),
    )

    first = broker.infer({"observation/state": np.zeros(1, dtype=np.float32)})
    second = broker.infer({"observation/state": np.zeros(1, dtype=np.float32)})

    assert policy.infer_calls == 1
    assert first["__chunk_meta"]["new_chunk"] is True
    assert second["__chunk_meta"]["new_chunk"] is False
    assert first["__chunk_meta"]["chunk_idx"] == 0
    assert second["__chunk_meta"]["chunk_idx"] == 1
    np.testing.assert_array_equal(first["actions"], chunk[0])
    np.testing.assert_array_equal(second["actions"], chunk[1])


def test_rtc_can_disable_action_prefix_and_emit_rtc_metadata() -> None:
    chunk = np.arange(16, dtype=np.float32).reshape(4, 4)
    policy = _PrefixAwarePolicy(chunk)
    broker = RealTimeChunkBroker(
        policy=policy,
        config=RTCConfig(
            action_horizon=4,
            inference_delay=1,
            execute_horizon=3,
            control_hz=20.0,
            use_action_prefix=False,
        ),
    )

    result = broker.infer({"__openpi": {"control_timestamp": 12.5}})

    assert policy.infer_calls == 1
    assert policy.infer_realtime_calls == 0
    assert result["__chunk_meta"]["mode"] == "rtc"
    assert result["__chunk_meta"]["new_chunk"] is True
    assert result["__chunk_meta"]["used_action_prefix"] is False
    assert result["__horizon_meta"]["mode"] == "rtc"
    assert result["__horizon_meta"]["used_action_prefix"] is False
    assert result["__horizon_meta"]["effective_horizon"] == 3
    np.testing.assert_allclose(result["__horizon_meta"]["planned_timestamps"], np.array([12.5, 12.55, 12.6, 12.65]))


def test_rtc_merge_keeps_step_metadata_aligned_after_large_skip() -> None:
    initial_chunk = np.arange(80, dtype=np.float32).reshape(10, 8)
    next_chunk = np.arange(240, dtype=np.float32).reshape(30, 8)
    policy = _DummyPolicy(initial_chunk)
    broker = RealTimeChunkBroker(
        policy=policy,
        config=RTCConfig(
            action_horizon=30,
            inference_delay=4,
            execute_horizon=10,
            control_hz=30.0,
            use_action_prefix=False,
        ),
    )

    first = broker.infer({"__openpi": {"control_timestamp": 1.0}})
    assert first["__chunk_meta"]["new_chunk"] is True

    with broker._lock:
        broker._current_chunk = initial_chunk.copy()
        broker._current_step_meta = [
            {
                "horizon_id": 0,
                "source_chunk_idx": idx,
                "horizon_start_timestamp": 1.0,
                "planned_timestamp": 1.0 + (idx / 30.0),
                "time_base": "control_timestamp",
                "infer_count": 1,
                "infer_ms": 10.0,
                "trigger_chunk_index": 6,
                "frames_elapsed": 0,
                "skip_count": 0,
                "used_action_prefix": False,
                "action_prefix_len": 0,
            }
            for idx in range(10)
        ]
        broker._chunk_index = 9
        broker._trigger_chunk_index = 6

    broker._merge_chunk(
        next_chunk,
        control_timestamp=1.3,
        infer_ms=12.0,
        infer_count=2,
        action_prefix_len=0,
    )

    assert len(broker._current_chunk) == 10
    assert len(broker._current_step_meta) == 10

    result = broker.infer({"__openpi": {"control_timestamp": 1.31}})
    assert result["__chunk_meta"]["mode"] == "rtc"
    assert result["__chunk_meta"]["chunk_idx"] == 0


def test_rtc_async_merge_emits_horizon_payload_once_on_next_step() -> None:
    initial_chunk = np.arange(32, dtype=np.float32).reshape(4, 8)
    next_chunk = np.arange(32, 64, dtype=np.float32).reshape(4, 8)
    broker = RealTimeChunkBroker(
        policy=_DummyPolicy(initial_chunk),
        config=RTCConfig(
            action_horizon=4,
            inference_delay=1,
            execute_horizon=4,
            control_hz=20.0,
            use_action_prefix=False,
        ),
    )

    first = broker.infer({"__openpi": {"control_timestamp": 5.0}})
    second = broker.infer({"__openpi": {"control_timestamp": 5.05}})

    assert first["__chunk_meta"]["new_chunk"] is True
    assert second["__chunk_meta"]["new_chunk"] is False

    broker._merge_chunk(
        next_chunk,
        control_timestamp=5.1,
        infer_ms=8.0,
        infer_count=2,
        action_prefix_len=0,
    )

    merged = broker.infer({"__openpi": {"control_timestamp": 5.1}})
    follow_up = broker.infer({"__openpi": {"control_timestamp": 5.15}})

    assert merged["__chunk_meta"]["new_chunk"] is True
    assert merged["__horizon_meta"]["mode"] == "rtc"
    np.testing.assert_array_equal(merged["__base_chunk"], next_chunk)
    assert follow_up["__chunk_meta"]["new_chunk"] is False
    assert "__horizon_meta" not in follow_up
    assert "__base_chunk" not in follow_up
