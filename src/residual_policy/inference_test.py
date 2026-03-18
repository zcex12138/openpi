from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from openpi.shared.rotation import quat_to_rotate6d
from openpi.shared.rotation import rotate6d_to_rotmat
import safetensors.torch
import torch

from residual_policy.action_repr import pose8_to_pose10
from residual_policy.config import ResidualModelConfig
from residual_policy.inference import FrankaPolicyPose10Wrapper
from residual_policy.inference import FrankaResidualStepPolicy
from residual_policy.inference import ResidualInferenceConfig
from residual_policy.model import build_residual_model
from residual_policy.model import ResidualMLP


class _DummyPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32)
        self.reset_calls = 0
        self.last_action_prefix = None

    def infer(self, obs: dict) -> dict:
        del obs
        return {
            "actions": self._action.copy(),
            "__chunk_meta": {"chunk_idx": 0, "chunk_size": 5},
            "policy_timing": {"infer_ms": 12.3},
        }

    def infer_realtime(self, obs: dict, *, action_prefix=None, noise=None) -> dict:
        del obs, noise
        self.last_action_prefix = None if action_prefix is None else np.asarray(action_prefix, dtype=np.float32).copy()
        return self.infer({})

    def reset(self) -> None:
        self.reset_calls += 1


def _write_checkpoint(checkpoint_dir: Path, output_bias: np.ndarray) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = ResidualMLP(input_dim=20, output_dim=10, hidden_dims=(16, 16), dropout=0.0)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.output.bias.copy_(torch.from_numpy(np.asarray(output_bias, dtype=np.float32)))

    safetensors.torch.save_model(model, checkpoint_dir / "model.safetensors")
    torch.save(
        {
            "input_mean": np.zeros(20, dtype=np.float32),
            "input_std": np.ones(20, dtype=np.float32),
            "target_mean": np.zeros(10, dtype=np.float32),
            "target_std": np.ones(10, dtype=np.float32),
        },
        checkpoint_dir / "residual_stats.pt",
    )
    torch.save(
        {
            "epoch": 1,
            "best_val_loss": 0.1,
            "config": {"model": {"hidden_dims": (16, 16), "dropout": 0.0}},
            "action_representation": "xyz_r6d_gripper",
        },
        checkpoint_dir / "metadata.pt",
    )


def _write_xense_checkpoint(checkpoint_dir: Path, output_bias: np.ndarray) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = ResidualModelConfig(
        kind="xense_single_step_mlp",
        hidden_dims=(16, 16),
        dropout=0.0,
        marker_hidden_dims=(4, 4),
        marker_embedding_dim=8,
        fusion_nhead=2,
        fusion_dim_feedforward=16,
    )
    model = build_residual_model(model_cfg, low_dim_input_dim=20, output_dim=10)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.residual_head.output.bias.copy_(torch.from_numpy(np.asarray(output_bias, dtype=np.float32)))

    safetensors.torch.save_model(model, checkpoint_dir / "model.safetensors")
    torch.save(
        {
            "input_mean": np.zeros(20, dtype=np.float32),
            "input_std": np.ones(20, dtype=np.float32),
            "target_mean": np.zeros(10, dtype=np.float32),
            "target_std": np.ones(10, dtype=np.float32),
        },
        checkpoint_dir / "residual_stats.pt",
    )
    torch.save(
        {
            "epoch": 1,
            "best_val_loss": 0.1,
            "config": {"model": dataclasses.asdict(model_cfg)},
            "model_kind": model_cfg.kind,
            "action_representation": "xyz_r6d_gripper",
        },
        checkpoint_dir / "metadata.pt",
    )


def _make_obs(*, include_tactile: bool = False) -> dict:
    obs = {
        "observation/state": np.array(
            [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    }
    if include_tactile:
        tactile = np.linspace(0.0, 1.0, 26 * 14 * 3, dtype=np.float32).reshape(26, 14, 3)
        obs["observation/tactile"] = tactile
    return obs


def _rotation_angle_from_pose10(pose10: np.ndarray) -> float:
    rot = rotate6d_to_rotmat(np.asarray(pose10[3:9], dtype=np.float32))
    trace = float(np.trace(rot))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def test_franka_residual_step_policy_outputs_absolute_pose10_and_preserves_base_action(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "residual_ckpt"
    identity_residual = np.concatenate(
        [np.zeros(3, dtype=np.float32), quat_to_rotate6d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), np.zeros(1, dtype=np.float32)]
    )
    _write_checkpoint(checkpoint_dir, identity_residual)

    base_action = pose8_to_pose10(np.array([0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32))
    wrapped = FrankaResidualStepPolicy(
        policy=_DummyPolicy(base_action),
        config=ResidualInferenceConfig(checkpoint_dir=str(checkpoint_dir)),
    )

    result = wrapped.infer(_make_obs())

    np.testing.assert_allclose(result["actions"], base_action, atol=1e-6)
    np.testing.assert_allclose(result["base_action"], base_action, atol=1e-6)
    np.testing.assert_allclose(result["base_action_pose10"], base_action, atol=1e-6)
    assert result["__chunk_meta"]["chunk_idx"] == 0
    assert result["policy_timing"]["infer_ms"] == 12.3
    assert result["policy_timing"]["residual_ms"] >= 0.0


def test_franka_residual_step_policy_applies_scale_and_caps(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "residual_ckpt"
    ninety_deg_z = quat_to_rotate6d(np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32))
    residual = np.concatenate(
        [
            np.array([0.3, 0.0, 0.0], dtype=np.float32),
            ninety_deg_z.astype(np.float32),
            np.array([0.5], dtype=np.float32),
        ]
    )
    _write_checkpoint(checkpoint_dir, residual)

    base_action = pose8_to_pose10(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32))
    wrapped = FrankaResidualStepPolicy(
        policy=_DummyPolicy(base_action),
        config=ResidualInferenceConfig(
            checkpoint_dir=str(checkpoint_dir),
            scale=2.0,
            translation_cap_m=0.1,
            rotation_cap_rad=np.pi / 6.0,
            gripper_cap=0.2,
        ),
    )

    result = wrapped.infer(_make_obs())

    assert result["actions"].shape == (10,)
    np.testing.assert_allclose(result["actions"][:3], np.array([0.1, 0.0, 0.0], dtype=np.float32), atol=1e-5)
    assert _rotation_angle_from_pose10(result["actions"]) <= (np.pi / 6.0) + 1e-5
    np.testing.assert_allclose(result["actions"][9], 0.3, atol=1e-5)


def test_franka_residual_step_policy_can_preserve_base_gripper(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "residual_ckpt"
    residual = np.concatenate(
        [
            np.array([0.05, 0.0, 0.0], dtype=np.float32),
            quat_to_rotate6d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            np.array([0.4], dtype=np.float32),
        ]
    )
    _write_checkpoint(checkpoint_dir, residual)

    base_action = pose8_to_pose10(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32))
    wrapped = FrankaResidualStepPolicy(
        policy=_DummyPolicy(base_action),
        config=ResidualInferenceConfig(
            checkpoint_dir=str(checkpoint_dir),
            apply_gripper_delta=False,
        ),
    )

    result = wrapped.infer(_make_obs())

    np.testing.assert_allclose(result["actions"][:3], np.array([0.05, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(result["actions"][9], 0.25, atol=1e-6)
    np.testing.assert_allclose(result["residual_action_pose10"][9], 0.0, atol=1e-6)


def test_franka_policy_pose10_wrapper_converts_outputs_and_realtime_prefix() -> None:
    base_action8 = np.array([0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    base_action10 = pose8_to_pose10(base_action8)
    inner = _DummyPolicy(base_action8)
    wrapped = FrankaPolicyPose10Wrapper(inner)

    infer_result = wrapped.infer(_make_obs())
    np.testing.assert_allclose(infer_result["actions"], base_action10, atol=1e-6)

    prefix10 = np.stack([base_action10, pose8_to_pose10(base_action8 + np.array([0.1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))])
    realtime_result = wrapped.infer_realtime(_make_obs(), action_prefix=prefix10)

    np.testing.assert_allclose(inner.last_action_prefix[0], base_action10, atol=1e-6)
    np.testing.assert_allclose(realtime_result["actions"], base_action10, atol=1e-6)


def test_franka_residual_step_policy_supports_xense_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "xense_residual_ckpt"
    residual = np.concatenate(
        [
            np.array([0.02, 0.0, 0.0], dtype=np.float32),
            quat_to_rotate6d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            np.array([0.1], dtype=np.float32),
        ]
    )
    _write_xense_checkpoint(checkpoint_dir, residual)

    base_action = pose8_to_pose10(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32))
    wrapped = FrankaResidualStepPolicy(
        policy=_DummyPolicy(base_action),
        config=ResidualInferenceConfig(checkpoint_dir=str(checkpoint_dir)),
    )

    result = wrapped.infer(_make_obs(include_tactile=True))

    np.testing.assert_allclose(result["actions"][:3], np.array([0.02, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(result["actions"][9], 0.35, atol=1e-6)


def test_franka_residual_step_policy_xense_checkpoint_requires_tactile(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "xense_residual_ckpt"
    identity_residual = np.concatenate(
        [np.zeros(3, dtype=np.float32), quat_to_rotate6d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), np.zeros(1, dtype=np.float32)]
    )
    _write_xense_checkpoint(checkpoint_dir, identity_residual)

    base_action = pose8_to_pose10(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32))
    wrapped = FrankaResidualStepPolicy(
        policy=_DummyPolicy(base_action),
        config=ResidualInferenceConfig(checkpoint_dir=str(checkpoint_dir)),
    )

    try:
        wrapped.infer(_make_obs())
    except KeyError as exc:
        assert "observation/tactile" in str(exc)
    else:
        raise AssertionError("Expected xense checkpoint inference to require tactile input")
