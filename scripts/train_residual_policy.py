"""Train a CR-DAGGER-style residual policy from a Zarr dataset."""

from __future__ import annotations

import tyro

from residual_policy.config import ResidualTrainingConfig
from residual_policy.trainer import train


def main(args: ResidualTrainingConfig) -> None:
    train(args)


if __name__ == "__main__":
    main(tyro.cli(ResidualTrainingConfig))
