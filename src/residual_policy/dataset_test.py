from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from residual_policy.action_repr import pose8_to_pose10
from residual_policy.dataset import build_cr_dagger_like_sample_indices
from residual_policy.dataset import ResidualDataset
from residual_policy.dataset import compute_normalization_stats
from residual_policy.dataset import load_residual_zarr


def _create_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    if hasattr(group, "create_array"):
        arr = group.create_array(name, shape=values.shape, dtype=values.dtype, fill_value=0)
        arr[:] = values
    else:
        group.create_dataset(name, data=values)


def _make_pose(index: int) -> np.ndarray:
    pose8 = np.array([0.1 * index, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(index % 2)], dtype=np.float32)
    return pose8_to_pose10(pose8)


def _write_test_zarr(path: Path, *, include_xense: bool = False) -> None:
    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")

    num_frames = 12
    robot_tcp_pose = np.stack([_make_pose(i) for i in range(num_frames)], axis=0)
    base_action = robot_tcp_pose.copy()
    corrected_action = robot_tcp_pose.copy()
    corrected_action[2:4, 0] += 0.05
    corrected_action[7:10, 0] += 0.05

    corrected_action_valid = np.ones(num_frames, dtype=np.uint8)
    is_human_teaching = np.zeros(num_frames, dtype=np.uint8)
    is_human_teaching[2:4] = 1
    is_human_teaching[7:10] = 1
    xense = np.arange(num_frames * 26 * 14 * 3, dtype=np.float32).reshape(num_frames, 26, 14, 3)
    xense[0, 0, 0, 0] = np.nan

    _create_array(data, "robot_tcp_pose", robot_tcp_pose)
    _create_array(data, "base_action", base_action)
    _create_array(data, "corrected_action", corrected_action)
    _create_array(data, "corrected_action_valid", corrected_action_valid)
    _create_array(data, "is_human_teaching", is_human_teaching)
    if include_xense:
        _create_array(data, "xense1_marker3d", xense)
    _create_array(meta, "episode_ends", np.array([6, 12], dtype=np.int64))


def test_cr_dagger_like_sampling_default_keeps_all_valid_frames_and_upweights_teaching(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path)
    data = load_residual_zarr(zarr_path)

    sample_indices = build_cr_dagger_like_sample_indices(
        data,
        [0, 1],
        weighted_sampling=2,
        correction_horizon=2,
        regular_valid_sampling="all",
        num_initial_episodes=0,
    )
    counts = Counter(sample_indices)

    assert counts[2] == 3
    assert counts[3] == 3
    assert counts[7] == 3
    assert counts[8] == 3
    assert counts[9] == 3
    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[4] == 1
    assert counts[5] == 1
    assert counts[6] == 1
    assert counts[10] == 1
    assert counts[11] == 1


def test_cr_dagger_like_sampling_none_matches_correction_only_behavior(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path)
    data = load_residual_zarr(zarr_path)

    sample_indices = build_cr_dagger_like_sample_indices(
        data,
        [0, 1],
        weighted_sampling=2,
        correction_horizon=2,
        regular_valid_sampling="none",
        num_initial_episodes=0,
    )
    counts = Counter(sample_indices)

    assert counts[2] == 3
    assert counts[3] == 3
    assert counts[7] == 3
    assert counts[8] == 3
    assert counts[9] == 3
    assert 0 not in counts
    assert 1 not in counts
    assert 4 not in counts
    assert 5 not in counts
    assert 6 not in counts
    assert 10 not in counts
    assert 11 not in counts


def test_cr_dagger_like_sampling_initial_episodes_uses_train_subset_order(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path)
    data = load_residual_zarr(zarr_path)

    sample_indices = build_cr_dagger_like_sample_indices(
        data,
        [1],
        weighted_sampling=2,
        correction_horizon=1,
        regular_valid_sampling="initial_episodes",
        num_initial_episodes=1,
    )
    counts = Counter(sample_indices)

    assert counts[6] == 1
    assert counts[7] == 3
    assert counts[8] == 3
    assert counts[9] == 1
    assert counts[10] == 1
    assert counts[11] == 1


def test_load_residual_zarr_requires_xense_when_requested(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path, include_xense=False)

    try:
        load_residual_zarr(zarr_path, require_xense=True)
    except KeyError as exc:
        assert "xense1_marker3d" in str(exc)
    else:
        raise AssertionError("Expected load_residual_zarr to require xense1_marker3d")


def test_residual_dataset_xense_model_returns_marker_tensor(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path, include_xense=True)
    data = load_residual_zarr(zarr_path, require_xense=True)
    sample_indices = [0, 2, 7]
    stats = compute_normalization_stats(data, sample_indices)
    dataset = ResidualDataset(data, sample_indices, model_kind="xense_single_step_mlp", stats=stats)

    sample = dataset[0]

    assert set(sample) == {"low_dim_inputs", "xense", "targets"}
    assert sample["low_dim_inputs"].shape == (20,)
    assert sample["xense"].shape == (26, 14, 3)
    assert sample["targets"].shape == (10,)
    assert np.isfinite(sample["xense"].numpy()).all()
