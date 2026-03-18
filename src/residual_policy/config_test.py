from residual_policy.config import build_model_config


def test_build_model_config_accepts_yaml_sequences() -> None:
    cfg = build_model_config(
        {
            "hidden_dims": [8, 16],
            "xense_shape": [26, 14, 3],
            "marker_hidden_dims": [4, 8],
        }
    )

    assert cfg.hidden_dims == (8, 16)
    assert cfg.xense_shape == (26, 14, 3)
    assert cfg.marker_hidden_dims == (4, 8)
