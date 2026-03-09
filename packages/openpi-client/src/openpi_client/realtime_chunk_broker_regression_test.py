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
