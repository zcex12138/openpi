from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from residual_policy.dataset import build_cr_dagger_like_sample_indices
from residual_policy.dataset import load_residual_zarr


def _create_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    if hasattr(group, "create_array"):
        arr = group.create_array(name, shape=values.shape, dtype=values.dtype, fill_value=0)
        arr[:] = values
    else:
        group.create_dataset(name, data=values)


def _make_pose(index: int) -> np.ndarray:
    return np.array([0.1 * index, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(index % 2)], dtype=np.float32)


def _write_test_zarr(path: Path) -> None:
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

    _create_array(data, "robot_tcp_pose", robot_tcp_pose)
    _create_array(data, "base_action", base_action)
    _create_array(data, "corrected_action", corrected_action)
    _create_array(data, "corrected_action_valid", corrected_action_valid)
    _create_array(data, "is_human_teaching", is_human_teaching)
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
