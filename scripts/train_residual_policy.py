"""Train a CR-DAGGER-style residual policy from a Zarr dataset."""

from __future__ import annotations

from collections.abc import Sequence
import sys

import tyro

from openpi.shared.yaml_config import extract_config_arg
from residual_policy.config import ResidualTrainingConfig
from residual_policy.config import load_training_config
from residual_policy.trainer import train


def parse_args(argv: Sequence[str] | None = None) -> ResidualTrainingConfig:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config_path, tyro_args = extract_config_arg(raw_args)
    if config_path is None:
        return tyro.cli(ResidualTrainingConfig, args=tyro_args)
    return tyro.cli(ResidualTrainingConfig, args=tyro_args, default=load_training_config(config_path))


def main(args: ResidualTrainingConfig | Sequence[str] | None = None) -> None:
    cfg = args if isinstance(args, ResidualTrainingConfig) else parse_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
