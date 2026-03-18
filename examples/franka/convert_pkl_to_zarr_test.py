from __future__ import annotations

import numpy as np

from examples.franka.convert_pkl_to_zarr import _process_episode
from examples.franka.convert_pkl_to_zarr import _select_action_target
from examples.franka.convert_pkl_to_zarr import _TRAINING_DATA_KEYS
from examples.franka.convert_pkl_to_zarr import _TRAINING_OPTIONAL_DATA_KEYS
from examples.franka.convert_pkl_to_zarr import parse_args
from residual_policy.action_repr import pose8_to_pose10
from residual_policy.action_repr import pose10_to_pose8


def _make_frame(
    *,
    action: np.ndarray | None,
    executed_action: np.ndarray | None = None,
    base_action: np.ndarray | None = None,
    corrected_action: np.ndarray | None = None,
    corrected_action_valid: bool | None = None,
    control_timestamp: float = 1.0,
    xense1_image: np.ndarray | None = None,
) -> dict:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    images = {
        "l500": image,
        "d400": image.copy(),
    }
    if xense1_image is not None:
        images["xense_1"] = np.asarray(xense1_image, dtype=np.uint8)
    frame = {
        "images": images,
        "state": np.array(
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "control_timestamp": control_timestamp,
    }
    if action is not None:
        frame["action"] = np.asarray(action, dtype=np.float32)
    if executed_action is None and action is not None:
        executed_action = action if action.shape[-1] == 8 else pose10_to_pose8(np.asarray(action, dtype=np.float32))
    if executed_action is not None:
        frame["executed_action"] = np.asarray(executed_action, dtype=np.float32)
    if base_action is not None:
        frame["base_action"] = np.asarray(base_action, dtype=np.float32)
    if corrected_action is not None:
        frame["corrected_action"] = np.asarray(corrected_action, dtype=np.float32)
    if corrected_action_valid is not None:
        frame["corrected_action_valid"] = corrected_action_valid
    return frame


def test_process_episode_auto_keeps_data_action_as_canonical_action() -> None:
    executed_action = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    action = pose8_to_pose10(executed_action)
    base_action = pose8_to_pose10(np.array([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32))
    corrected_action = pose8_to_pose10(np.array([0.8, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32))

    result = _process_episode(
        {
            "fps": 30.0,
            "frames": [
                _make_frame(
                    action=action,
                    base_action=base_action,
                    corrected_action=corrected_action,
                    corrected_action_valid=True,
                )
            ],
        },
        action_target="auto",
    )

    assert result is not None
    np.testing.assert_allclose(result["action"][0], action, atol=1e-6)
    np.testing.assert_allclose(result["executed_action"][0], executed_action, atol=1e-6)
    np.testing.assert_allclose(result["base_action"][0], base_action, atol=1e-6)
    np.testing.assert_allclose(result["corrected_action"][0], corrected_action, atol=1e-6)
    assert int(result["corrected_action_valid"][0]) == 1


def test_process_episode_converts_legacy_pose8_fields_to_pose10() -> None:
    action = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    base_action = np.array([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    corrected_action = np.array([0.8, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)

    result = _process_episode(
        {
            "fps": 30.0,
            "frames": [
                _make_frame(
                    action=action,
                    base_action=base_action,
                    corrected_action=corrected_action,
                    corrected_action_valid=True,
                )
            ],
        },
        action_target="auto",
    )

    assert result is not None
    np.testing.assert_allclose(result["action"][0], pose8_to_pose10(action), atol=1e-6)
    np.testing.assert_allclose(result["base_action"][0], pose8_to_pose10(base_action), atol=1e-6)
    np.testing.assert_allclose(result["corrected_action"][0], pose8_to_pose10(corrected_action), atol=1e-6)
    np.testing.assert_allclose(result["executed_action"][0], action, atol=1e-6)


def test_process_episode_executed_action_target_uses_true_executed_action() -> None:
    policy_action_pose8 = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    executed_action_pose8 = np.array([0.6, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    base_action_pose8 = np.array([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = _process_episode(
        {
            "fps": 30.0,
            "frames": [
                _make_frame(
                    action=pose8_to_pose10(policy_action_pose8),
                    executed_action=executed_action_pose8,
                    base_action=pose8_to_pose10(base_action_pose8),
                )
            ],
        },
        action_target="executed",
    )

    assert result is not None
    np.testing.assert_allclose(result["action"][0], pose8_to_pose10(executed_action_pose8), atol=1e-6)
    np.testing.assert_allclose(result["executed_action"][0], executed_action_pose8, atol=1e-6)
    np.testing.assert_allclose(result["base_action"][0], pose8_to_pose10(base_action_pose8), atol=1e-6)


def test_select_action_target_auto_falls_back_from_executed_to_corrected_to_base() -> None:
    base_action = np.arange(10, dtype=np.float32) + 10.0
    corrected_action = np.arange(10, dtype=np.float32) + 20.0

    selected, valid = _select_action_target(
        action_target="auto",
        executed_action=None,
        base_action=base_action,
        corrected_action=corrected_action,
        corrected_action_valid=True,
    )
    assert valid is True
    np.testing.assert_allclose(selected, corrected_action, atol=1e-6)

    selected, valid = _select_action_target(
        action_target="auto",
        executed_action=None,
        base_action=base_action,
        corrected_action=corrected_action,
        corrected_action_valid=False,
    )
    assert valid is True
    np.testing.assert_allclose(selected, base_action, atol=1e-6)


def test_process_episode_preserves_xense1_camera_img_when_present() -> None:
    xense1_image = np.full((2, 2, 3), 7, dtype=np.uint8)

    result = _process_episode(
        {
            "fps": 30.0,
            "frames": [
                _make_frame(
                    action=pose8_to_pose10(np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)),
                    xense1_image=xense1_image,
                )
            ],
        },
        action_target="auto",
    )

    assert result is not None
    assert "xense1_camera_img" in result
    np.testing.assert_array_equal(result["xense1_camera_img"][0], xense1_image)


def test_training_export_keeps_only_training_fields() -> None:
    assert "action" in _TRAINING_DATA_KEYS
    assert "base_action" in _TRAINING_DATA_KEYS
    assert "corrected_action" in _TRAINING_DATA_KEYS
    assert "executed_action" not in _TRAINING_DATA_KEYS
    assert "robot_tcp_velocity" not in _TRAINING_DATA_KEYS
    assert _TRAINING_OPTIONAL_DATA_KEYS == ("xense1_camera_img", "xense1_marker3d")


def test_parse_args_uses_script_defaults_and_cli_overrides() -> None:
    args = parse_args(["--input-dir", "demo_records", "--action-target", "corrected"])

    assert args.input_dir == "demo_records"
    assert args.action_target == "corrected"


def test_parse_args_uses_script_defaults_without_cli_args() -> None:
    args = parse_args([])

    assert args.input_dir == "eval_records"
    assert args.drop_frames_after_human_teaching == 0
    assert args.action_target == "auto"


def test_parse_args_rejects_yaml_config_flag() -> None:
    try:
        parse_args(["--config", "legacy_config.yaml"])
    except ValueError as exc:
        assert "--config is no longer supported" in str(exc)
    else:
        raise AssertionError("Expected parse_args() to reject --config")
