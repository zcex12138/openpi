import torch

from residual_policy.config import ResidualModelConfig
from residual_policy.model import build_residual_model


def test_build_xense_residual_model_forward_shape():
    model_cfg = ResidualModelConfig(
        kind="xense_single_step_mlp",
        hidden_dims=(8, 8),
        dropout=0.0,
        marker_hidden_dims=(4, 8),
        marker_embedding_dim=16,
        fusion_nhead=4,
        fusion_dim_feedforward=32,
    )
    model = build_residual_model(model_cfg, low_dim_input_dim=20, output_dim=10)

    outputs = model(
        torch.randn(2, 20, dtype=torch.float32),
        torch.randn(2, 26, 14, 3, dtype=torch.float32),
    )

    assert outputs.shape == (2, 10)
