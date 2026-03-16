"""Residual-policy package with lazy top-level exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ResidualDataset": ("residual_policy.dataset", "ResidualDataset"),
    "ResidualMLP": ("residual_policy.model", "ResidualMLP"),
    "ResidualModelConfig": ("residual_policy.config", "ResidualModelConfig"),
    "ResidualNormalizationStats": ("residual_policy.dataset", "ResidualNormalizationStats"),
    "ResidualSamplingConfig": ("residual_policy.config", "ResidualSamplingConfig"),
    "ResidualTrainingConfig": ("residual_policy.config", "ResidualTrainingConfig"),
    "ResidualZarrData": ("residual_policy.dataset", "ResidualZarrData"),
    "build_cr_dagger_like_sample_indices": ("residual_policy.dataset", "build_cr_dagger_like_sample_indices"),
    "build_input_features": ("residual_policy.action_repr", "build_input_features"),
    "as_pose10": ("residual_policy.action_repr", "as_pose10"),
    "canonicalize_quaternion_sign": ("residual_policy.action_repr", "canonicalize_quaternion_sign"),
    "compute_normalization_stats": ("residual_policy.dataset", "compute_normalization_stats"),
    "decode_residual_action": ("residual_policy.action_repr", "decode_residual_action"),
    "decode_residual_pose10": ("residual_policy.action_repr", "decode_residual_pose10"),
    "FrankaPolicyPose10Wrapper": ("residual_policy.inference", "FrankaPolicyPose10Wrapper"),
    "encode_residual_action": ("residual_policy.action_repr", "encode_residual_action"),
    "FrankaResidualStepPolicy": ("residual_policy.inference", "FrankaResidualStepPolicy"),
    "load_residual_zarr": ("residual_policy.dataset", "load_residual_zarr"),
    "pose8_to_pose10": ("residual_policy.action_repr", "pose8_to_pose10"),
    "pose10_to_pose8": ("residual_policy.action_repr", "pose10_to_pose8"),
    "ResidualInferenceConfig": ("residual_policy.inference", "ResidualInferenceConfig"),
    "split_episode_indices": ("residual_policy.dataset", "split_episode_indices"),
    "train": ("residual_policy.trainer", "train"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
