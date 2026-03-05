import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.policies import policy as _policy


class _DummyTorchModel:
    def __init__(self):
        self.last_action_prefix = None

    def to(self, _device: str):
        return self

    def eval(self):
        return self

    def sample_actions(self, *_args, **_kwargs):
        raise AssertionError("sample_actions should not be called in this test")

    def realtime_sample_actions(self, _sample_rng, _observation, **kwargs):
        action_prefix = kwargs["action_prefix"]
        self.last_action_prefix = action_prefix.detach().cpu().numpy()
        batch_size, _, action_dim = action_prefix.shape
        return torch.zeros((batch_size, 4, action_dim), dtype=torch.float32, device=action_prefix.device)


def test_infer_realtime_prefix_transform_avoids_double_state_quat_conversion():
    quat_to_r6d = _transforms.QuatToRotate6d()
    delta_r6d = _transforms.DeltaRotate6dActions(r6d_start=3)
    model = _DummyTorchModel()
    policy = _policy.Policy(model, transforms=[quat_to_r6d, delta_r6d], is_pytorch=True)

    obs_state = np.array([0.1, -0.2, 0.3, 0.70710677, 0.0, 0.70710677, 0.0, 0.5], dtype=np.float32)
    action_prefix = np.array(
        [
            [0.2, -0.1, 0.4, 0.9238795, 0.0, 0.38268343, 0.0, 0.2],
            [0.3, 0.0, 0.5, 0.8660254, 0.0, 0.5, 0.0, 0.3],
        ],
        dtype=np.float32,
    )

    state_r6d = quat_to_r6d({"state": obs_state.copy()})["state"]
    expected_prefix = quat_to_r6d({"actions": action_prefix.copy()})["actions"]
    expected_prefix = delta_r6d({"state": state_r6d.copy(), "actions": expected_prefix})["actions"]

    obs = {
        "image": {"cam": np.zeros((4, 4, 3), dtype=np.uint8)},
        "image_mask": {"cam": np.array(True)},
        "state": obs_state.copy(),
    }
    policy.infer_realtime(obs, action_prefix=action_prefix.copy())

    assert model.last_action_prefix is not None
    np.testing.assert_allclose(model.last_action_prefix[0], expected_prefix, atol=1e-6)
