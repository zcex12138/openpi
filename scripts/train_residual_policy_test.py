import dataclasses
from pathlib import Path

import numpy as np
import torch
import zarr

from residual_policy.config import ResidualModelConfig
from residual_policy.config import ResidualSamplingConfig
from residual_policy.config import ResidualTrainingConfig
from residual_policy.dataset import build_cr_dagger_like_sample_indices
from residual_policy.dataset import compute_normalization_stats
from residual_policy.dataset import load_residual_zarr
from residual_policy.dataset import split_episode_indices

from . import train_residual_policy


def _create_array(group: zarr.Group, name: str, values) -> None:
    if hasattr(group, "create_array"):
        arr = group.create_array(name, shape=values.shape, dtype=values.dtype, fill_value=0)
        arr[:] = values
    else:
        group.create_dataset(name, data=values)


def _write_test_zarr(path: Path, *, translation_offset: float = 0.0) -> None:
    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")

    num_frames = 16
    robot_tcp_pose = np.zeros((num_frames, 8), dtype=np.float32)
    robot_tcp_pose[:, 3] = 1.0
    robot_tcp_pose[:, 0] = np.linspace(0.0, 0.15, num_frames) + translation_offset
    robot_tcp_pose[:, 7] = np.linspace(0.0, 1.0, num_frames)
    base_action = robot_tcp_pose.copy()
    corrected_action = robot_tcp_pose.copy()
    corrected_action[4:8, 0] += 0.02
    corrected_action[10:14, 0] += 0.03

    corrected_action_valid = np.ones(num_frames, dtype=np.uint8)
    is_human_teaching = np.zeros(num_frames, dtype=np.uint8)
    is_human_teaching[4:8] = 1
    is_human_teaching[10:14] = 1

    _create_array(data, "robot_tcp_pose", robot_tcp_pose)
    _create_array(data, "base_action", base_action)
    _create_array(data, "corrected_action", corrected_action)
    _create_array(data, "corrected_action_valid", corrected_action_valid)
    _create_array(data, "is_human_teaching", is_human_teaching)
    _create_array(meta, "episode_ends", np.array([8, 16], dtype=np.int64))


def test_train_residual_policy_smoke(tmp_path: Path):
    zarr_path = tmp_path / "dataset.zarr"
    _write_test_zarr(zarr_path)
    cfg = ResidualTrainingConfig(
        zarr_path=str(zarr_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        exp_name="smoke",
        batch_size=4,
        num_epochs=2,
        num_workers=0,
        device="cpu",
        save_every_epochs=1,
        log_every_steps=1,
        wandb_enabled=False,
        model=ResidualModelConfig(hidden_dims=(8, 8), dropout=0.0),
        sampling=ResidualSamplingConfig(
            weighted_sampling=2,
            correction_horizon=1,
            regular_valid_sampling="all",
            num_initial_episodes=0,
            val_ratio=0.5,
        ),
    )
    train_residual_policy.main(cfg)

    latest_dir = Path(cfg.checkpoint_dir) / cfg.exp_name / "latest"
    assert (latest_dir / "model.safetensors").exists()
    assert (latest_dir / "optimizer.pt").exists()
    assert (latest_dir / "metadata.pt").exists()
    assert (latest_dir / "residual_stats.pt").exists()

    cfg_resume = dataclasses.replace(cfg, resume=True, num_epochs=3)
    train_residual_policy.main(cfg_resume)
    metadata = torch.load(latest_dir / "metadata.pt", map_location="cpu", weights_only=False)
    assert metadata["epoch"] == 3
    assert metadata["config"]["sampling"]["regular_valid_sampling"] == "all"


def test_resume_rewrites_stats_for_current_dataset(tmp_path: Path):
    first_zarr_path = tmp_path / "dataset_first.zarr"
    second_zarr_path = tmp_path / "dataset_second.zarr"
    _write_test_zarr(first_zarr_path)
    _write_test_zarr(second_zarr_path, translation_offset=0.25)

    cfg = ResidualTrainingConfig(
        zarr_path=str(first_zarr_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        exp_name="resume_stats",
        batch_size=4,
        num_epochs=1,
        num_workers=0,
        device="cpu",
        save_every_epochs=1,
        log_every_steps=1,
        wandb_enabled=False,
        model=ResidualModelConfig(hidden_dims=(8, 8), dropout=0.0),
        sampling=ResidualSamplingConfig(
            weighted_sampling=2,
            correction_horizon=1,
            regular_valid_sampling="all",
            num_initial_episodes=0,
            val_ratio=0.5,
        ),
    )
    train_residual_policy.main(cfg)

    cfg_resume = dataclasses.replace(cfg, zarr_path=str(second_zarr_path), resume=True, num_epochs=2)
    train_residual_policy.main(cfg_resume)

    latest_dir = Path(cfg.checkpoint_dir) / cfg.exp_name / "latest"
    saved_stats = torch.load(latest_dir / "residual_stats.pt", map_location="cpu", weights_only=False)

    data = load_residual_zarr(second_zarr_path)
    train_episode_indices, _ = split_episode_indices(data.num_episodes, cfg.sampling.val_ratio, cfg.sampling.seed)
    train_indices = build_cr_dagger_like_sample_indices(
        data,
        train_episode_indices,
        weighted_sampling=cfg.sampling.weighted_sampling,
        correction_horizon=cfg.sampling.correction_horizon,
        regular_valid_sampling=cfg.sampling.regular_valid_sampling,
        num_initial_episodes=cfg.sampling.num_initial_episodes,
        deduplicate=False,
    )
    expected_stats = compute_normalization_stats(data, train_indices)

    np.testing.assert_allclose(saved_stats["input_mean"], expected_stats.input_mean)
    np.testing.assert_allclose(saved_stats["input_std"], expected_stats.input_std)
    np.testing.assert_allclose(saved_stats["target_mean"], expected_stats.target_mean)
    np.testing.assert_allclose(saved_stats["target_std"], expected_stats.target_std)
