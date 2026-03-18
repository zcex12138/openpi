import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client.cr_dagger_chunk_broker import CrDaggerChunkBroker
from openpi_client.cr_dagger_chunk_broker import CrDaggerChunkBrokerConfig
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents.policy_agent import PolicyAgent


class _DummyEnvironment:
    def __init__(self) -> None:
        self.applied_actions: list[dict] = []
        self.marked_complete = False
        self.include_recording_snapshot = False

    def reset(self) -> None:
        return

    def get_observation(self) -> dict:
        observation = {"sensor": np.array([1.0], dtype=np.float32)}
        if self.include_recording_snapshot:
            observation["__openpi"] = {"recording_snapshot": {"seq": 7, "payload": np.ones((2, 2), dtype=np.uint8)}}
        return observation

    def apply_action(self, action: dict) -> None:
        self.applied_actions.append(action)

    def is_episode_complete(self) -> bool:
        return False

    def mark_episode_complete(self) -> None:
        self.marked_complete = True


class _CaptureAgent:
    def __init__(self) -> None:
        self.observations: list[dict] = []

    def get_action(self, observation: dict) -> dict:
        self.observations.append(observation)
        return {"actions": np.ones(8, dtype=np.float32)}

    def reset(self) -> None:
        return


class _CaptureSubscriber:
    def __init__(self) -> None:
        self.steps: list[tuple[dict, dict]] = []

    def on_episode_start(self) -> None:
        return

    def on_step(self, observation: dict, action: dict) -> None:
        self.steps.append((observation, action))

    def on_episode_end(self) -> None:
        return


class _DummyPolicy(_base_policy.BasePolicy):
    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self.infer_calls = 0

    def infer(self, obs: dict) -> dict:
        self.infer_calls += 1
        return {"actions": self._chunk.copy()}


def test_runtime_propagates_canonical_control_timestamp(monkeypatch) -> None:
    environment = _DummyEnvironment()
    agent = _CaptureAgent()
    subscriber = _CaptureSubscriber()
    runtime = _runtime.Runtime(environment=environment, agent=agent, subscribers=[subscriber])

    monkeypatch.setattr(_runtime.time, "time", lambda: 123.456)

    assert runtime._step() is True
    assert agent.observations[0]["__openpi"]["control_timestamp"] == 123.456
    assert environment.applied_actions[0]["__openpi"]["control_timestamp"] == 123.456
    assert subscriber.steps[0][0]["__openpi"]["control_timestamp"] == 123.456


def test_runtime_handles_cr_dagger_lag_stop_without_dispatch(monkeypatch) -> None:
    environment = _DummyEnvironment()
    policy = _DummyPolicy(np.arange(15, dtype=np.float32).reshape(5, 3))
    broker = CrDaggerChunkBroker(
        policy=policy,
        config=CrDaggerChunkBrokerConfig(action_horizon=5, execute_horizon=5, max_skip_steps=1, control_hz=10.0),
    )
    runtime = _runtime.Runtime(environment=environment, agent=PolicyAgent(broker), subscribers=[])
    runtime._in_episode = True

    time_values = iter([0.0, 0.3, 0.3, 0.3])
    monkeypatch.setattr(_runtime.time, "time", lambda: next(time_values))

    assert runtime._step() is True
    assert runtime._step() is False
    assert len(environment.applied_actions) == 1
    assert environment.marked_complete is True
    assert runtime._in_episode is False
    assert policy.infer_calls == 1


def test_runtime_hides_recording_snapshot_from_agent_but_keeps_it_for_subscribers(monkeypatch) -> None:
    environment = _DummyEnvironment()
    environment.include_recording_snapshot = True
    agent = _CaptureAgent()
    subscriber = _CaptureSubscriber()
    runtime = _runtime.Runtime(environment=environment, agent=agent, subscribers=[subscriber])

    monkeypatch.setattr(_runtime.time, "time", lambda: 456.789)

    assert runtime._step() is True
    assert "recording_snapshot" not in agent.observations[0]["__openpi"]
    assert "recording_snapshot" not in environment.applied_actions[0]["__openpi"]
    assert subscriber.steps[0][0]["__openpi"]["recording_snapshot"]["seq"] == 7
