from __future__ import annotations

import numpy as np

from examples.franka.convert_pkl_to_zarr import _process_episode
from examples.franka.convert_pkl_to_zarr import _select_action_target


def _make_frame(
    *,
    action: np.ndarray | None,
    base_action: np.ndarray | None = None,
    corrected_action: np.ndarray | None = None,
    corrected_action_valid: bool | None = None,
    control_timestamp: float = 1.0,
) -> dict:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = {
        "images": {
            "l500": image,
            "d400": image.copy(),
        },
        "state": np.array(
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "control_timestamp": control_timestamp,
    }
    if action is not None:
        frame["action"] = np.asarray(action, dtype=np.float32)
    if base_action is not None:
        frame["base_action"] = np.asarray(base_action, dtype=np.float32)
    if corrected_action is not None:
        frame["corrected_action"] = np.asarray(corrected_action, dtype=np.float32)
    if corrected_action_valid is not None:
        frame["corrected_action_valid"] = corrected_action_valid
    return frame


def test_process_episode_auto_keeps_data_action_as_executed_action() -> None:
    executed_action = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    base_action = np.array([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    corrected_action = np.array([0.8, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)

    result = _process_episode(
        {
            "fps": 30.0,
            "frames": [
                _make_frame(
                    action=executed_action,
                    base_action=base_action,
                    corrected_action=corrected_action,
                    corrected_action_valid=True,
                )
            ],
        },
        action_target="auto",
    )

    assert result is not None
    np.testing.assert_allclose(result["action"][0], executed_action, atol=1e-6)
    np.testing.assert_allclose(result["executed_action"][0], executed_action, atol=1e-6)
    np.testing.assert_allclose(result["base_action"][0], base_action, atol=1e-6)
    np.testing.assert_allclose(result["corrected_action"][0], corrected_action, atol=1e-6)
    assert int(result["corrected_action_valid"][0]) == 1


def test_select_action_target_auto_falls_back_from_executed_to_corrected_to_base() -> None:
    base_action = np.arange(8, dtype=np.float32) + 10.0
    corrected_action = np.arange(8, dtype=np.float32) + 20.0

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
