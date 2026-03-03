import flax.nnx as nnx
import jax.numpy as jnp

from openpi.shared import array_typing as at


class TactileCNNEncoder(nnx.Module):
    """Encodes a fixed-size tactile grid (H, W, 3) into a single embedding token.

    Architecture: Conv2d stack → global average pool → linear projection.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (32, 64),
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        in_ch = 3
        self._num_convs = len(hidden_dims)
        for i, out_ch in enumerate(hidden_dims):
            setattr(self, f"conv_{i}", nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding="SAME", rngs=rngs))
            in_ch = out_ch
        self.proj = nnx.Linear(in_ch, output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    @at.typecheck
    def __call__(
        self, x: at.Float[at.Array, "b h w c"], *, train: bool = False
    ) -> at.Float[at.Array, "b 1 d"]:
        for i in range(self._num_convs):
            x = nnx.relu(getattr(self, f"conv_{i}")(x))
        # global average pool → (b, c)
        x = jnp.mean(x, axis=(1, 2))
        x = self.proj(x)
        x = self.dropout(x, deterministic=not train)
        return x[:, None, :]
