import numpy as np

from openpi.models import model as _model
from openpi.policies import franka_policy
from openpi.shared.rotation import quat_to_rotate6d
from openpi.training import config as _config
import openpi.transforms as _transforms


def test_rotate6d_franka_configs_use_pose10_targets_without_quat_conversion(tmp_path) -> None:
    for config_name in (
        "pi05_franka_cola_lora",
        "pi05_franka_cola_relative_lora",
        "pi05_franka_tactile_lora",
    ):
        train_config = _config.get_config(config_name)
        data_config = train_config.data.create(tmp_path, train_config.model)

        shifted_state_to_action = next(
            transform
            for transform in data_config.data_transforms.inputs
            if isinstance(transform, _transforms.ShiftedStateToAction)
        )
        assert shifted_state_to_action.pose_dims.start == 0
        assert shifted_state_to_action.pose_dims.stop == 10
        assert shifted_state_to_action.normalize_quat_sign is False

        franka_inputs_transform = next(
            transform for transform in data_config.data_transforms.inputs if isinstance(transform, franka_policy.FrankaInputs)
        )
        assert franka_inputs_transform.state_dim == 10
        assert franka_inputs_transform.normalize_quat_sign is False

        assert not any(
            isinstance(transform, _transforms.QuatToRotate6d) for transform in data_config.data_transforms.inputs
        )
        assert any(
            isinstance(transform, franka_policy.FrankaOutputs) and transform.action_dim == 10
            for transform in data_config.data_transforms.outputs
        )


def test_legacy_pose8_franka_relative_config_uses_pose8_targets(tmp_path) -> None:
    train_config = _config.get_config("pi05_franka_cola_relative_pose8_lora")
    data_config = train_config.data.create(tmp_path, train_config.model)

    assert isinstance(data_config.data_transforms.inputs[0], _transforms.Rotate6dStateToQuat)

    shifted_state_to_action = next(
        transform
        for transform in data_config.data_transforms.inputs
        if isinstance(transform, _transforms.ShiftedStateToAction)
    )
    assert shifted_state_to_action.pose_dims.start == 0
    assert shifted_state_to_action.pose_dims.stop == 8
    assert shifted_state_to_action.normalize_quat_sign is True

    franka_inputs_transform = next(
        transform for transform in data_config.data_transforms.inputs if isinstance(transform, franka_policy.FrankaInputs)
    )
    assert franka_inputs_transform.state_dim == 8
    assert franka_inputs_transform.normalize_quat_sign is True

    assert not any(
        isinstance(transform, _transforms.QuatToRotate6d) for transform in data_config.data_transforms.inputs
    )
    assert any(
        isinstance(transform, franka_policy.FrankaOutputs) and transform.action_dim == 8
        for transform in data_config.data_transforms.outputs
    )


def test_franka_inputs_promote_legacy_pose8_state_to_pose10() -> None:
    quat = np.array([0.9238795, 0.0, 0.38268343, 0.0], dtype=np.float32)
    legacy_state = np.concatenate(
        [
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            quat,
            np.array([0.4], dtype=np.float32),
            np.arange(6, dtype=np.float32),
        ]
    )

    result = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05, state_dim=10)(
        {
            "observation/image": np.zeros((4, 4, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "observation/state": legacy_state,
        }
    )

    expected_pose10 = np.concatenate(
        [
            legacy_state[:3],
            quat_to_rotate6d(quat),
            legacy_state[7:8],
        ]
    )
    np.testing.assert_allclose(result["state"], expected_pose10, atol=1e-6)


def test_rotate6d_state_to_quat_converts_pose10_state_and_preserves_wrench() -> None:
    quat = np.array([0.9238795, 0.0, 0.38268343, 0.0], dtype=np.float32)
    pose10_state = np.concatenate(
        [
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            quat_to_rotate6d(quat),
            np.array([0.4], dtype=np.float32),
            np.arange(6, dtype=np.float32),
        ]
    )

    result = _transforms.Rotate6dStateToQuat()({"observation/state": pose10_state.copy()})
    expected_state = np.concatenate(
        [
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            quat,
            np.array([0.4], dtype=np.float32),
            np.arange(6, dtype=np.float32),
        ]
    )
    np.testing.assert_allclose(result["observation/state"], expected_state, atol=1e-6)

    pose8_state = expected_state.copy()
    result_pose8 = _transforms.Rotate6dStateToQuat()({"observation/state": pose8_state.copy()})
    np.testing.assert_allclose(result_pose8["observation/state"], pose8_state, atol=1e-6)
