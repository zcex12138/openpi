import numpy as np
import pytest
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import franka_policy
from openpi.shared import normalize as _normalize


class _DummyTorchModel:
    def __init__(self, *, action_dim: int = 32):
        self.action_dim = action_dim
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
        batch_size = action_prefix.shape[0]
        return torch.zeros((batch_size, 4, self.action_dim), dtype=torch.float32, device=action_prefix.device)


def _make_obs(obs_state: np.ndarray) -> dict:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": obs_state,
    }


def test_infer_realtime_prefix_transform_maps_to_model_space_with_pre_normalize_state():
    franka_inputs = franka_policy.FrankaInputs(
        model_type=_model.ModelType.PI05,
        state_dim=8,
        rotation_representation="r6d",
        normalize_quat_sign=False,
    )
    quat_to_r6d = _transforms.QuatToRotate6d()
    delta_xyz = _transforms.DeltaActions(_transforms.make_bool_mask(3, -7))
    delta_r6d = _transforms.DeltaRotate6dActions(r6d_start=3)
    norm = _transforms.Normalize(
        {
            "state": _normalize.NormStats(mean=np.linspace(0.1, 1.0, 10), std=np.linspace(1.1, 2.0, 10)),
            "actions": _normalize.NormStats(mean=np.linspace(-0.5, 0.4, 10), std=np.linspace(1.5, 2.4, 10)),
        }
    )
    pad = _transforms.PadStatesAndActions(32)

    model = _DummyTorchModel()
    policy = _policy.Policy(
        model,
        transforms=[franka_inputs, quat_to_r6d, delta_xyz, delta_r6d, norm, pad],
        is_pytorch=True,
    )

    obs_state = np.array([0.1, -0.2, 0.3, 0.70710677, 0.0, 0.70710677, 0.0, 0.5], dtype=np.float32)
    action_prefix = np.array(
        [
            [0.2, -0.1, 0.4, 0.9238795, 0.0, 0.38268343, 0.0, 0.2],
            [0.3, 0.0, 0.5, 0.8660254, 0.0, 0.5, 0.0, 0.3],
        ],
        dtype=np.float32,
    )

    state_r6d = franka_inputs(_make_obs(obs_state.copy()))["state"]
    state_r6d = quat_to_r6d({"state": state_r6d})["state"]
    expected_prefix = quat_to_r6d({"actions": action_prefix.copy()})["actions"]
    expected_prefix = delta_xyz({"state": state_r6d.copy(), "actions": expected_prefix})["actions"]
    expected_prefix = delta_r6d({"state": state_r6d.copy(), "actions": expected_prefix})["actions"]
    expected_prefix = norm({"actions": expected_prefix})["actions"]
    expected_prefix = pad({"state": state_r6d.copy(), "actions": expected_prefix})["actions"]

    policy.infer_realtime(_make_obs(obs_state.copy()), action_prefix=action_prefix.copy())

    assert model.last_action_prefix is not None
    np.testing.assert_allclose(model.last_action_prefix[0], expected_prefix, atol=1e-6)
    assert model.last_action_prefix.shape[-1] == 32


def test_infer_realtime_prefix_transform_raises_when_dim_mismatches_model():
    model = _DummyTorchModel(action_dim=32)
    policy = _policy.Policy(model, transforms=[_transforms.QuatToRotate6d()], is_pytorch=True)

    obs = {
        "image": {"cam": np.zeros((4, 4, 3), dtype=np.uint8)},
        "image_mask": {"cam": np.array(True)},
        "state": np.array([0.1, -0.2, 0.3, 0.70710677, 0.0, 0.70710677, 0.0, 0.5], dtype=np.float32),
    }
    action_prefix = np.array([[0.2, -0.1, 0.4, 0.9238795, 0.0, 0.38268343, 0.0, 0.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="action_prefix dim mismatch"):
        policy.infer_realtime(obs, action_prefix=action_prefix)
