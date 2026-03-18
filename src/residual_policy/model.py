"""PyTorch residual models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from residual_policy.config import ResidualModelConfig


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


class XenseMarkerCNNEncoder(nn.Module):
    """Encode xense marker3d into a single embedding token."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_shape[-1] != 3:
            raise ValueError(f"Expected xense input shape (H, W, 3), got {input_shape}")

        layers: list[nn.Module] = []
        in_channels = input_shape[-1]
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_dim

        self.input_shape = input_shape
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, output_dim)

    def forward(self, xense: torch.Tensor) -> torch.Tensor:
        if xense.ndim != 4 or tuple(xense.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected xense tensor shape (B, {self.input_shape[0]}, {self.input_shape[1]}, 3), "
                f"got {tuple(xense.shape)}"
            )
        hidden = torch.nan_to_num(xense.float(), nan=0.0, posinf=0.0, neginf=0.0).permute(0, 3, 1, 2)
        hidden = self.backbone(hidden)
        hidden = hidden.mean(dim=(2, 3))
        return self.proj(hidden)


class CrDaggerStyleFusion(nn.Module):
    """Single-token modality-attention block mirroring CR-DAGGER's structure."""

    def __init__(
        self,
        d_model: int,
        *,
        num_tokens: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.position_embedding = nn.Parameter(torch.randn(num_tokens, d_model) / (d_model**0.5))
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.linear_projection = nn.Linear(d_model * num_tokens, d_model)

    def forward(self, modality_tokens: torch.Tensor) -> torch.Tensor:
        if modality_tokens.ndim != 3:
            raise ValueError(f"Expected modality_tokens rank 3, got {tuple(modality_tokens.shape)}")
        if modality_tokens.shape[1] != self.num_tokens or modality_tokens.shape[2] != self.d_model:
            raise ValueError(
                f"Expected modality_tokens shape (B, {self.num_tokens}, {self.d_model}), "
                f"got {tuple(modality_tokens.shape)}"
            )
        hidden = modality_tokens + self.position_embedding.unsqueeze(0)
        hidden = self.transformer_encoder(hidden)
        return self.linear_projection(hidden.reshape(hidden.shape[0], -1))


class XenseResidualMLP(nn.Module):
    """Residual policy with xense CNN encoder and CR-DAGGER-style fusion."""

    def __init__(
        self,
        *,
        low_dim_input_dim: int,
        output_dim: int,
        model_config: ResidualModelConfig,
    ) -> None:
        super().__init__()
        self.low_dim_input_dim = low_dim_input_dim
        self.xense_shape = model_config.xense_shape
        self.xense_encoder = XenseMarkerCNNEncoder(
            input_shape=model_config.xense_shape,
            hidden_dims=model_config.marker_hidden_dims,
            output_dim=model_config.marker_embedding_dim,
        )
        self.modality_fusion = CrDaggerStyleFusion(
            d_model=model_config.marker_embedding_dim,
            num_tokens=1,
            nhead=model_config.fusion_nhead,
            dim_feedforward=model_config.fusion_dim_feedforward,
            dropout=model_config.fusion_dropout,
        )
        self.residual_head = ResidualMLP(
            input_dim=low_dim_input_dim + model_config.marker_embedding_dim,
            output_dim=output_dim,
            hidden_dims=model_config.hidden_dims,
            dropout=model_config.dropout,
        )

    def forward(self, low_dim_inputs: torch.Tensor, xense: torch.Tensor) -> torch.Tensor:
        if low_dim_inputs.ndim != 2 or low_dim_inputs.shape[-1] != self.low_dim_input_dim:
            raise ValueError(
                f"Expected low_dim_inputs shape (B, {self.low_dim_input_dim}), got {tuple(low_dim_inputs.shape)}"
            )
        tactile_token = self.xense_encoder(xense)[:, None, :]
        fused = self.modality_fusion(tactile_token)
        return self.residual_head(torch.cat([fused, low_dim_inputs.float()], dim=-1))


def get_model_kind_from_metadata(metadata: dict[str, Any]) -> str:
    model_kind = metadata.get("model_kind")
    if isinstance(model_kind, str) and model_kind:
        return model_kind
    config = metadata.get("config", {})
    if isinstance(config, dict):
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            cfg_kind = model_cfg.get("kind")
            if isinstance(cfg_kind, str) and cfg_kind:
                return cfg_kind
    return "legacy_mlp"


def build_residual_model(
    model_config: ResidualModelConfig,
    *,
    low_dim_input_dim: int,
    output_dim: int,
) -> nn.Module:
    if model_config.kind == "legacy_mlp":
        return ResidualMLP(
            input_dim=low_dim_input_dim,
            output_dim=output_dim,
            hidden_dims=model_config.hidden_dims,
            dropout=model_config.dropout,
        )
    if model_config.kind == "xense_single_step_mlp":
        return XenseResidualMLP(
            low_dim_input_dim=low_dim_input_dim,
            output_dim=output_dim,
            model_config=model_config,
        )
    raise ValueError(f"Unsupported residual model kind: {model_config.kind!r}")
