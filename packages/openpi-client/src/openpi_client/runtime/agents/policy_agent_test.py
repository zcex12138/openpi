import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client.runtime.agents.policy_agent import PolicyAgent


class _DummyPolicy(_base_policy.BasePolicy):
    def __init__(self) -> None:
        self.infer_calls = 0
        self.reset_calls = 0
        self.last_obs = None

    def infer(self, obs: dict) -> dict:
        self.infer_calls += 1
        self.last_obs = obs
        return {"actions": np.full(8, self.infer_calls, dtype=np.float32)}

    def reset(self) -> None:
        self.reset_calls += 1


def test_policy_agent_uses_policy_inference_even_with_teaching_metadata() -> None:
    policy = _DummyPolicy()
    agent = PolicyAgent(policy)

    action = agent.get_action(
        {
            "__openpi": {
                "is_human_teaching": True,
                "teaching_segment_id": 3,
                "teaching_step": 1,
            }
        }
    )

    assert policy.infer_calls == 1
    assert policy.reset_calls == 0
    np.testing.assert_array_equal(action["actions"], np.ones(8, dtype=np.float32))


def test_policy_agent_reset_delegates_to_policy() -> None:
    policy = _DummyPolicy()
    agent = PolicyAgent(policy)

    agent.reset()

    assert policy.reset_calls == 1
    assert policy.infer_calls == 0
