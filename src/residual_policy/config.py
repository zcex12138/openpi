"""Configuration objects for residual-policy training."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

from openpi.shared.yaml_config import load_yaml_mapping
from openpi.shared.yaml_config import require_mapping

RegularValidSamplingMode = Literal["all", "initial_episodes", "none"]
ResidualModelKind = Literal["legacy_mlp", "xense_single_step_mlp"]


@dataclasses.dataclass(frozen=True)
class ResidualSamplingConfig:
    weighted_sampling: int = 4
    correction_horizon: int = 10
    regular_valid_sampling: RegularValidSamplingMode = "all"
    num_initial_episodes: int = 0
    val_ratio: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        if self.weighted_sampling < 1:
            raise ValueError(f"weighted_sampling must be >= 1, got {self.weighted_sampling}")
        if self.correction_horizon < 0:
            raise ValueError(f"correction_horizon must be >= 0, got {self.correction_horizon}")
        if self.num_initial_episodes < 0:
            raise ValueError(f"num_initial_episodes must be >= 0, got {self.num_initial_episodes}")
        if self.regular_valid_sampling not in {"all", "initial_episodes", "none"}:
            raise ValueError(
                "regular_valid_sampling must be one of "
                f"('all', 'initial_episodes', 'none'), got {self.regular_valid_sampling!r}"
            )


@dataclasses.dataclass(frozen=True)
class ResidualModelConfig:
    kind: ResidualModelKind = "legacy_mlp"
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1
    xense_required: bool = True
    xense_shape: tuple[int, int, int] = (26, 14, 3)
    marker_hidden_dims: tuple[int, ...] = (32, 64)
    marker_embedding_dim: int = 768
    fusion_nhead: int = 8
    fusion_dim_feedforward: int = 2048
    fusion_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"legacy_mlp", "xense_single_step_mlp"}:
            raise ValueError(f"Unsupported residual model kind: {self.kind!r}")
        if not self.hidden_dims:
            raise ValueError("model.hidden_dims must not be empty")
        if not self.marker_hidden_dims:
            raise ValueError("model.marker_hidden_dims must not be empty")
        if len(self.xense_shape) != 3 or self.xense_shape[-1] != 3:
            raise ValueError(f"model.xense_shape must be (H, W, 3), got {self.xense_shape}")
        if any(dim <= 0 for dim in self.xense_shape):
            raise ValueError(f"model.xense_shape must contain only positive dims, got {self.xense_shape}")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(f"model.hidden_dims must contain only positive dims, got {self.hidden_dims}")
        if any(dim <= 0 for dim in self.marker_hidden_dims):
            raise ValueError(
                f"model.marker_hidden_dims must contain only positive dims, got {self.marker_hidden_dims}"
            )
        if self.marker_embedding_dim <= 0:
            raise ValueError(
                f"model.marker_embedding_dim must be > 0, got {self.marker_embedding_dim}"
            )
        if self.fusion_nhead <= 0:
            raise ValueError(f"model.fusion_nhead must be > 0, got {self.fusion_nhead}")
        if self.fusion_dim_feedforward <= 0:
            raise ValueError(
                f"model.fusion_dim_feedforward must be > 0, got {self.fusion_dim_feedforward}"
            )
        if self.marker_embedding_dim % self.fusion_nhead != 0:
            raise ValueError(
                "model.marker_embedding_dim must be divisible by model.fusion_nhead, "
                f"got {self.marker_embedding_dim} and {self.fusion_nhead}"
            )
        if self.dropout < 0:
            raise ValueError(f"model.dropout must be >= 0, got {self.dropout}")
        if self.fusion_dropout < 0:
            raise ValueError(f"model.fusion_dropout must be >= 0, got {self.fusion_dropout}")


@dataclasses.dataclass(frozen=True)
class ResidualTrainingConfig:
    zarr_path: str
    checkpoint_dir: str
    exp_name: str = "residual_policy"
    batch_size: int = 128
    num_epochs: int = 100
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-6
    num_workers: int = 0
    device: str = "auto"
    save_every_epochs: int = 5
    log_every_steps: int = 10
    resume: bool = False
    wandb_enabled: bool = False
    wandb_project: str = "residual-policy"
    seed: int = 42
    sampling: ResidualSamplingConfig = dataclasses.field(default_factory=ResidualSamplingConfig)
    model: ResidualModelConfig = dataclasses.field(default_factory=ResidualModelConfig)


def _build_sampling_config(value: Any) -> ResidualSamplingConfig:
    data = require_mapping(value, field_name="sampling")
    try:
        return ResidualSamplingConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid sampling config: {exc}") from exc


def _as_tuple_of_ints(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a YAML sequence")
    return tuple(int(dim) for dim in value)


def build_model_config(value: Any) -> ResidualModelConfig:
    data = require_mapping(value, field_name="model")
    if "hidden_dims" in data:
        data["hidden_dims"] = _as_tuple_of_ints(data["hidden_dims"], field_name="model.hidden_dims")
    if "xense_shape" in data:
        xense_shape = _as_tuple_of_ints(data["xense_shape"], field_name="model.xense_shape")
        if len(xense_shape) != 3:
            raise ValueError(f"model.xense_shape must have length 3, got {xense_shape}")
        data["xense_shape"] = xense_shape
    if "marker_hidden_dims" in data:
        data["marker_hidden_dims"] = _as_tuple_of_ints(
            data["marker_hidden_dims"], field_name="model.marker_hidden_dims"
        )
    try:
        return ResidualModelConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid model config: {exc}") from exc


def load_training_config(config_path: str | Path) -> ResidualTrainingConfig:
    path = Path(config_path)
    config_data = load_yaml_mapping(path)
    if "sampling" in config_data:
        config_data["sampling"] = _build_sampling_config(config_data["sampling"])
    if "model" in config_data:
        config_data["model"] = build_model_config(config_data["model"])

    try:
        return ResidualTrainingConfig(**config_data)
    except TypeError as exc:
        raise ValueError(f"Invalid residual training config in {path}: {exc}") from exc
