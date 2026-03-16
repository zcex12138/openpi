"""PyTorch residual MLP."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """A small MLP that predicts residual actions from state+base features."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, output_dim)

        nn.init.normal_(self.output.weight, std=1e-4)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        return self.output(hidden)
