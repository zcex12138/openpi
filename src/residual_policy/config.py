"""Configuration objects for residual-policy training."""

from __future__ import annotations

import dataclasses
from typing import Literal

RegularValidSamplingMode = Literal["all", "initial_episodes", "none"]


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
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1


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
