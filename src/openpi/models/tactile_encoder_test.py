import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models.tactile_encoder import TactileCNNEncoder


def test_tactile_encoder_output_shape():
    """Test that TactileCNNEncoder outputs correct shape (b, 1, d)."""
    output_dim = 2048  # PaliGemma width
    encoder = TactileCNNEncoder(
        output_dim=output_dim,
        hidden_dims=(32, 64),
        dropout=0.0,
        rngs=nnx.Rngs(0),
    )

    # Input: batch of tactile grids (26, 14, 3)
    batch_size = 4
    x = jnp.ones((batch_size, 26, 14, 3), dtype=jnp.float32)

    out = encoder(x, train=False)
    assert out.shape == (batch_size, 1, output_dim)


def test_tactile_encoder_train_mode():
    """Test encoder runs in train mode with dropout."""
    encoder = TactileCNNEncoder(
        output_dim=256,
        hidden_dims=(16,),
        dropout=0.5,
        rngs=nnx.Rngs(0),
    )

    x = jnp.ones((2, 26, 14, 3), dtype=jnp.float32)
    out = encoder(x, train=True)
    assert out.shape == (2, 1, 256)
