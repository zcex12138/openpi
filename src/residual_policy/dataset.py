"""Zarr residual dataset with CR-DAGGER-style sampling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from residual_policy.config import ResidualModelKind
from residual_policy.config import RegularValidSamplingMode
from residual_policy.action_repr import build_input_features
from residual_policy.action_repr import encode_residual_action


@dataclasses.dataclass(frozen=True)
class ResidualNormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


@dataclasses.dataclass(frozen=True)
class ResidualZarrData:
    inputs: np.ndarray
    targets: np.ndarray
    xense_marker3d: np.ndarray | None
    is_human_teaching: np.ndarray
    corrected_action_valid: np.ndarray
    episode_ends: np.ndarray

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    def episode_bounds(self) -> list[tuple[int, int]]:
        starts = np.concatenate([[0], self.episode_ends[:-1]])
        return [(int(start), int(end)) for start, end in zip(starts, self.episode_ends, strict=True)]


def _load_required_array(group: zarr.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Missing required Zarr key: {key}")
    return np.asarray(group[key][:])


def _normalize_xense_marker3d(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
    num_frames: int,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected xense1_marker3d shape (N, H, W, 3), got {arr.shape}")
    if arr.shape[0] != num_frames:
        raise ValueError(
            f"xense1_marker3d frame count mismatch: expected {num_frames}, got {arr.shape[0]}"
        )
    if tuple(arr.shape[1:]) != expected_shape:
        raise ValueError(f"Expected xense1_marker3d shape (N, {expected_shape}), got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_residual_zarr(
    zarr_path: str | Path,
    *,
    require_xense: bool = False,
    xense_shape: tuple[int, int, int] = (26, 14, 3),
) -> ResidualZarrData:
    root = zarr.open(str(zarr_path), mode="r")
    data_group = root["data"]
    meta_group = root["meta"]

    robot_tcp_pose = _load_required_array(data_group, "robot_tcp_pose").astype(np.float32)
    base_action = _load_required_array(data_group, "base_action").astype(np.float32)
    corrected_action = _load_required_array(data_group, "corrected_action").astype(np.float32)
    corrected_action_valid = _load_required_array(data_group, "corrected_action_valid").astype(bool)
    is_human_teaching = _load_required_array(data_group, "is_human_teaching").astype(bool)
    episode_ends = _load_required_array(meta_group, "episode_ends").astype(np.int64)

    if robot_tcp_pose.ndim != 2 or robot_tcp_pose.shape[-1] != 10:
        raise ValueError(f"Expected robot_tcp_pose shape (N, 10), got {robot_tcp_pose.shape}")
    if base_action.shape != robot_tcp_pose.shape or corrected_action.shape != robot_tcp_pose.shape:
        raise ValueError("robot_tcp_pose, base_action, and corrected_action must have identical (N, 10) shapes")
    if corrected_action_valid.shape != (robot_tcp_pose.shape[0],):
        raise ValueError("corrected_action_valid shape mismatch")
    if is_human_teaching.shape != (robot_tcp_pose.shape[0],):
        raise ValueError("is_human_teaching shape mismatch")

    xense_marker3d: np.ndarray | None = None
    if "xense1_marker3d" in data_group:
        xense_marker3d = _normalize_xense_marker3d(
            data_group["xense1_marker3d"][:],
            expected_shape=xense_shape,
            num_frames=robot_tcp_pose.shape[0],
        )
    elif require_xense:
        raise KeyError("Missing required Zarr key: xense1_marker3d")

    inputs = build_input_features(robot_tcp_pose, base_action)
    targets = encode_residual_action(base_action, corrected_action)

    return ResidualZarrData(
        inputs=inputs,
        targets=targets,
        xense_marker3d=xense_marker3d,
        is_human_teaching=is_human_teaching,
        corrected_action_valid=corrected_action_valid,
        episode_ends=episode_ends,
    )


def split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if num_episodes <= 1 or val_ratio <= 0:
        return list(range(num_episodes)), []

    n_val = min(max(1, round(num_episodes * val_ratio)), num_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_indices = sorted(rng.choice(num_episodes, size=n_val, replace=False).tolist())
    train_indices = [idx for idx in range(num_episodes) if idx not in set(val_indices)]
    return train_indices, val_indices


def build_cr_dagger_like_sample_indices(
    data: ResidualZarrData,
    episode_indices: list[int],
    *,
    weighted_sampling: int,
    correction_horizon: int,
    regular_valid_sampling: RegularValidSamplingMode,
    num_initial_episodes: int,
    deduplicate: bool = False,
) -> list[int]:
    if weighted_sampling < 1:
        raise ValueError("weighted_sampling must be >= 1")
    if correction_horizon < 0:
        raise ValueError("correction_horizon must be >= 0")
    if num_initial_episodes < 0:
        raise ValueError("num_initial_episodes must be >= 0")
    if regular_valid_sampling not in {"all", "initial_episodes", "none"}:
        raise ValueError("regular_valid_sampling must be one of ('all', 'initial_episodes', 'none')")

    sample_indices: list[int] = []
    valid = data.corrected_action_valid
    bounds = data.episode_bounds()

    for subset_episode_idx, episode_idx in enumerate(episode_indices):
        start, end = bounds[episode_idx]
        valid_local = valid[start:end]
        teaching_local = data.is_human_teaching[start:end] & valid_local

        if regular_valid_sampling == "all":
            include_regular_valid = True
        elif regular_valid_sampling == "initial_episodes":
            include_regular_valid = subset_episode_idx < num_initial_episodes
        else:
            include_regular_valid = False

        if include_regular_valid:
            regular_valid_local = valid_local & ~data.is_human_teaching[start:end]
            sample_indices.extend((start + np.flatnonzero(regular_valid_local)).tolist())

        correction_interval = np.flatnonzero(teaching_local)
        sample_indices.extend((start + correction_interval).tolist())

        if correction_interval.size == 0:
            continue

        correction_starts = correction_interval[
            ~np.concatenate([[False], np.diff(correction_interval) == 1])
        ]
        for local_idx in correction_starts.tolist():
            sample_indices.extend([start + local_idx] * weighted_sampling)
            for offset in range(1, correction_horizon + 1):
                next_idx = local_idx + offset
                if next_idx >= len(teaching_local) or not teaching_local[next_idx]:
                    break
                sample_indices.extend([start + next_idx] * weighted_sampling)

    if deduplicate:
        return sorted(set(sample_indices))
    return sample_indices


def compute_normalization_stats(data: ResidualZarrData, sample_indices: list[int]) -> ResidualNormalizationStats:
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    inputs = data.inputs[sample_indices]
    targets = data.targets[sample_indices]
    input_std = np.std(inputs, axis=0).astype(np.float32)
    target_std = np.std(targets, axis=0).astype(np.float32)
    input_std = np.where(input_std < 1e-6, 1.0, input_std)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)
    return ResidualNormalizationStats(
        input_mean=np.mean(inputs, axis=0).astype(np.float32),
        input_std=input_std.astype(np.float32),
        target_mean=np.mean(targets, axis=0).astype(np.float32),
        target_std=target_std.astype(np.float32),
    )


class ResidualDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of CR-DAGGER-style residual training samples."""

    def __init__(
        self,
        data: ResidualZarrData,
        sample_indices: list[int],
        *,
        model_kind: ResidualModelKind = "legacy_mlp",
        stats: ResidualNormalizationStats | None = None,
    ) -> None:
        if not sample_indices:
            raise ValueError("ResidualDataset requires at least one sample index")
        self._data = data
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self._model_kind = model_kind
        self._stats = stats
        if self._model_kind == "xense_single_step_mlp" and self._data.xense_marker3d is None:
            raise ValueError("ResidualDataset requires xense_marker3d when model_kind='xense_single_step_mlp'")

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = int(self.sample_indices[idx])
        inputs = self._data.inputs[sample_idx].copy()
        targets = self._data.targets[sample_idx].copy()
        if self._stats is not None:
            inputs = (inputs - self._stats.input_mean) / self._stats.input_std
            targets = (targets - self._stats.target_mean) / self._stats.target_std
        if self._model_kind == "xense_single_step_mlp":
            return {
                "low_dim_inputs": torch.from_numpy(inputs.astype(np.float32)),
                "xense": torch.from_numpy(self._data.xense_marker3d[sample_idx].astype(np.float32)),
                "targets": torch.from_numpy(targets.astype(np.float32)),
            }
        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),
            "targets": torch.from_numpy(targets.astype(np.float32)),
        }
