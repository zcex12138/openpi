import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_shifted_state_to_action_basic():
    """Test basic state→action extraction without additional shift."""
    # Simulate 30 frames of 14D state loaded by LeRobot's delta_timestamps
    state = np.ones((30, 14))
    state[:, :8] = np.arange(30 * 8).reshape(30, 8)  # Put identifiable values in pose dims

    transform = _transforms.ShiftedStateToAction()
    result = transform({"observation/state": state})

    # Should extract first 8 dimensions as actions
    assert result["actions"].shape == (30, 8)
    np.testing.assert_array_equal(result["actions"], state[:, :8])


def test_shifted_state_to_action_with_shift():
    """Test additional frame offset for latency compensation."""
    # Create state with identifiable values in each frame
    state = np.arange(30 * 14).reshape(30, 14)

    transform = _transforms.ShiftedStateToAction(additional_shift=1)
    result = transform({"observation/state": state})

    # First 29 actions should be the next frame's state
    np.testing.assert_array_equal(result["actions"][:-1], state[1:, :8])
    # Last action should be padded with the last valid frame (state[29])
    # since there's no state[30] to shift to
    np.testing.assert_array_equal(result["actions"][-1], state[-1, :8])


def test_shifted_state_to_action_custom_dims():
    """Test custom pose dimension selection."""
    state = np.arange(30 * 14).reshape(30, 14)

    # Extract only first 7 dimensions (pose without gripper)
    transform = _transforms.ShiftedStateToAction(pose_dims=slice(0, 7))
    result = transform({"observation/state": state})

    assert result["actions"].shape == (30, 7)
    np.testing.assert_array_equal(result["actions"], state[:, :7])


def test_shifted_state_to_action_custom_keys():
    """Test custom state/action key names."""
    state = np.ones((30, 14))

    transform = _transforms.ShiftedStateToAction(
        state_key="custom/state", action_key="custom/actions"
    )
    result = transform({"custom/state": state})

    assert "custom/actions" in result
    assert result["custom/actions"].shape == (30, 8)


def test_shifted_state_to_action_1d_inference_skipped():
    """Test that 1D state array (inference case) is skipped entirely."""
    # Simulate inference with a single state frame
    state = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 14 dims

    transform = _transforms.ShiftedStateToAction(
        pose_dims=slice(0, 8),
        additional_shift=1,
    )
    data = {"observation/state": state}
    result = transform(data)

    # Transform should be skipped for 1D state (inference case)
    # The "actions" key should NOT be added since we skip the transform
    assert "actions" not in result
    # Original state should be unchanged
    np.testing.assert_array_equal(result["observation/state"], state)
