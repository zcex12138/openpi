# Design: Tactile Point Cloud Fusion Architecture

## Context

The π₀.₅ model uses a dual-expert Transformer architecture:
- **PaliGemma**: Vision-language model processing images + text
- **Action Expert**: Dedicated model for action prediction via flow matching

Current data flow:
```
Images → SigLIP → image_tokens (256 tokens per image)
Language → Tokenizer → lang_tokens (≤200 tokens)
State → Linear → state_token (1 token, only pi0; pi05 uses discrete state)

[image_tokens, lang_tokens] → PaliGemma → prefix_embeddings
[action_tokens, time_tokens] → Action Expert (with adaRMS conditioning for pi05)
```

## Goals / Non-Goals

### Goals
- Add tactile point cloud as a new observation modality
- Maintain backward compatibility with existing checkpoints
- Expose CNN encoder hyperparameters in training config

### Non-Goals
- Tactile prediction as auxiliary training objective
- Multi-sensor tactile fusion (e.g., multiple Xense cameras)
- Force/torque sensor integration (different data format)
- Optional tactile input at runtime (tactile=None with tactile_enabled=True is an error)

## Architectural Decisions

### Decision 1: Fusion Strategy

**Options Considered:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Token-level (prefix) | Add tactile token alongside image tokens | Simple, follows existing pattern | Tactile in all attention layers |
| B. State-level (suffix) | Concat tactile with state in embed_suffix | Separates modalities | Only affects action expert |
| C. adaRMS conditioning | Modulate action expert with tactile | Fine-grained control | Complex implementation |

**Decision: Option A (Token-level fusion)**

Rationale:
- Consistent with how images are processed
- Tactile information available to both PaliGemma and Action Expert
- Single integration point (`embed_prefix`)
- Natural attention between tactile and visual features

### Decision 2: Tactile Encoder Architecture

**FINAL CONSTRAINTS (User Approved):**
- **Framework**: flax.linen + nnx_bridge.ToNNX (consistent with SigLIP/Gemma)
- **Input**: Fixed (batch, 26, 14, 3) float32 - Xense tactile grid
- **Output**: (batch, 1, 2048) float32 - forced to match paligemma_config.width
- **Architecture**: 2D CNN with 3×3 kernel, stride=1, SAME padding, GELU activation, Global Average Pooling
- **Hidden dims**: (64, 128, 256) - fixed, not configurable
- **No mask/padding**: Fixed grid shape eliminates need for per-point masks
- **NaN handling**: jnp.nan_to_num at encoder entry

**Final Architecture:**

```python
class TactileEncoder(nn.Module):
    """Encodes tactile grid (26, 14, 3) to a single embedding token.

    CONSTRAINTS:
    - Input: (batch, 26, 14, 3) float32
    - Output: (batch, 1, paligemma_width) float32
    - NaN/Inf in input → replaced with 0 via nan_to_num
    """
    embed_dim: int  # Must equal paligemma_config.width (2048 for gemma_2b)
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, tactile, train: bool = False):
        # tactile: (batch, 26, 14, 3)
        x = jnp.nan_to_num(tactile, nan=0.0, posinf=0.0, neginf=0.0)

        x = nn.Conv(64, (3, 3), strides=(1, 1), padding="SAME", dtype=self.dtype_mm)(x)
        x = nn.gelu(x)
        x = nn.Conv(128, (3, 3), strides=(1, 1), padding="SAME", dtype=self.dtype_mm)(x)
        x = nn.gelu(x)
        x = nn.Conv(256, (3, 3), strides=(1, 1), padding="SAME", dtype=self.dtype_mm)(x)
        x = nn.gelu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch, 256)

        x = nn.Dense(self.embed_dim, dtype=self.dtype_mm)(x)
        return x[:, None, :]  # (batch, 1, embed_dim)
```

**Parameter Count:** ~200K (negligible compared to PaliGemma's 2B)

### Decision 3: Data Format

**Tactile Data Shape:** `(26, 14, 3)` - Fixed 26×14 grid

**FINAL CONSTRAINTS (User Approved):**
- **Representation**: Raw u/v/depth (pixel coordinates + depth), NOT reprojected XYZ
- **Normalization**: None - CNN learns to adapt to raw value ranges
- **Augmentation**: None - tactile data is small and structure-sensitive
- **Time sync**: Hard sync assumed (captured together with images)

### Decision 4: Configuration Extension

**FINAL CONSTRAINTS (User Approved):**

**Pi0Config additions:**

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # Existing fields...

    # Tactile encoder settings
    tactile_enabled: bool = False
    # Note: hidden_dims=(64,128,256) is FIXED in encoder, not configurable
    # Note: No dropout - removed for simplicity
    # Note: No mask fields - fixed 26×14 grid
```

**Training config additions:**

```python
@dataclasses.dataclass(frozen=True)
class LeRobotFrankaTactileDataConfig(LeRobotFrankaDataConfigV2):
    """Extends V2 config with tactile support."""

    tactile_key: str = "observation.tactile.xense1_marker3d"
    # Tactile grid shape: (26, 14, 3) - fixed
```

### Decision 5: Inference Behavior

**FINAL CONSTRAINTS (User Approved):**

| Scenario | Behavior |
|----------|----------|
| `tactile_enabled=True` + `tactile=None` | **Raise ValueError** (expose pipeline issues) |
| `tactile_enabled=True` + `tactile` contains NaN/Inf | **Clean in encoder** via `jnp.nan_to_num` |
| `tactile_enabled=False` | Ignore tactile field entirely, identical to baseline |

### Decision 6: Training Configuration

**FINAL CONSTRAINTS (User Approved):**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Trainable params | LoRA + tactile_encoder only | Minimal fine-tuning |
| Learning rate | Same as LoRA (warmup=500, peak_lr=1.5e-5) | Simplicity |
| Modality dropout | None | Let model learn fusion freely |

### Decision 7: Checkpoint Handling

**FINAL CONSTRAINTS (User Approved):**

| Scenario | Behavior |
|----------|----------|
| Load pi05_base with `tactile_enabled=True` | **Allow missing** - tactile params randomly initialized (extend missing_regex) |
| Shape mismatch on any parameter | **Always fail-fast** (both training and inference) |
| Extra keys in checkpoint | Warn and skip |

## Integration Points

### 1. Pi0 Model (`src/openpi/models/pi0.py`)

```python
class Pi0(BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # Existing initialization...
        self.tactile_enabled = bool(getattr(config, "tactile_enabled", False))

        if self.tactile_enabled:
            tactile_encoder = nnx_bridge.ToNNX(
                TactileEncoder(embed_dim=paligemma_config.width, dtype_mm=config.dtype)
            )
            tactile_encoder.lazy_init(jnp.zeros((1, 26, 14, 3), dtype=jnp.float32), train=False, rngs=rngs)
            self.tactile_encoder = tactile_encoder

    def embed_prefix(self, obs: Observation) -> tuple[...]:
        # Existing image + language embedding...

        if self.tactile_enabled:
            if obs.tactile is None:
                raise ValueError("tactile_enabled=True but obs.tactile is None")
            tactile_tokens = self.tactile_encoder(obs.tactile, train=False)
            tokens.append(tactile_tokens)
            input_mask.append(jnp.ones((obs.tactile.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [False]  # Tactile attends to images/language

        # Continue with concatenation...
```

### 2. Observation Dataclass (`src/openpi/models/model.py`)

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    # Existing fields...

    # Tactile grid (optional, but required when tactile_enabled=True)
    tactile: at.Float[ArrayT, "*b 26 14 3"] | None = None
    # Note: No tactile_mask - fixed grid shape
```

### 3. FrankaInputs Transform (`src/openpi/policies/franka_policy.py`)

```python
@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    # Existing fields...

    tactile_key: str | None = None  # "observation/tactile"

    def __call__(self, data: dict) -> dict:
        # Existing processing...

        if self.tactile_key and self.tactile_key in data:
            marker3d = np.asarray(data[self.tactile_key])  # (26, 14, 3)
            inputs["tactile"] = marker3d.astype(np.float32)
            # No padding/mask - fixed shape

        return inputs
```

### 4. Data Config (`src/openpi/training/config.py`)

New config class and training config entry.

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/openpi/models/tactile_encoder.py` | NEW | TactileEncoder (linen) module |
| `src/openpi/models/pi0_config.py` | MODIFY | Add tactile_enabled field |
| `src/openpi/models/pi0.py` | MODIFY | Integrate tactile in embed_prefix |
| `src/openpi/models/model.py` | MODIFY | Add tactile field to Observation |
| `src/openpi/policies/franka_policy.py` | MODIFY | Extend FrankaInputs |
| `src/openpi/training/config.py` | MODIFY | Add data config and training config |
| `examples/convert_zarr_to_lerobot_v2.0.py` | MODIFY | Include xense1_marker3d |

## Property-Based Testing (PBT) Properties

### Shape Invariants

**S1: Encoder I/O shape and dtype**
- [INVARIANT] For any `B≥1`: if `x ∈ float32^{B×26×14×3}`, then `E(x) ∈ float32^{B×1×2048}`
- [FALSIFICATION] Generate B∈[1..8], vary input shape/dtype, verify output shape
- [PROPERTY TYPE] InvariantPreservation

**S2: Batch permutation equivariance**
- [INVARIANT] For any batch permutation π: `E(π(x)) = π(E(x))`
- [FALSIFICATION] Random permutations, compare outputs
- [PROPERTY TYPE] Commutativity

**S3: embed_prefix() token concatenation**
- [INVARIANT] With tactile_enabled=True: `tokens.shape[1] = tokens_baseline.shape[1] + 1` and `tokens[:, -1, :] = tactile_token`
- [FALSIFICATION] Compare with/without tactile, verify exact position
- [PROPERTY TYPE] InvariantPreservation

### Numerical Stability

**N1: nan_to_num idempotency**
- [INVARIANT] `E(x) = E(nan_to_num(x))` for all x
- [FALSIFICATION] Inject NaN/Inf at random positions, compare outputs
- [PROPERTY TYPE] Idempotency

**N2: Output finiteness**
- [INVARIANT] `isfinite(E(x))` for all elements, even if x contains NaN/Inf
- [FALSIFICATION] Extreme values, mixed NaN/Inf/large numbers
- [PROPERTY TYPE] Bounds

**N3: Gradient finiteness**
- [INVARIANT] For finite x: `isfinite(∇_x L)` where `L = sum(E(x))`
- [FALSIFICATION] Test near-zero, large values, sparse distributions
- [PROPERTY TYPE] Bounds

### Backward Compatibility

**B1: tactile_enabled=False ignores tactile**
- [INVARIANT] `P_off(obs_with_tactile) = P_off(obs_without_tactile)` for same non-tactile fields
- [FALSIFICATION] Random tactile values with tactile_enabled=False, verify identical output
- [PROPERTY TYPE] InvariantPreservation

**B2: tactile_enabled=True + tactile=None raises ValueError**
- [INVARIANT] `embed_prefix(obs)` raises `ValueError` when `tactile_enabled=True ∧ obs.tactile is None`
- [FALSIFICATION] Various None representations (field missing, explicit None)
- [PROPERTY TYPE] Bounds

### Attention Integrity

**A1: Full bidirectional attention in prefix**
- [INVARIANT] `attn_mask[b,i,j] = prefix_mask[b,i] ∧ prefix_mask[b,j]` for all prefix positions including tactile
- [FALSIFICATION] Random padding patterns, verify attention mask consistency
- [PROPERTY TYPE] InvariantPreservation

### Checkpoint Invariants

**C1: Save/restore round-trip**
- [INVARIANT] `restore(save(θ)) = θ` for tactile-enabled params
- [FALSIFICATION] Random init, save, restore, compare pytrees and forward outputs
- [PROPERTY TYPE] RoundTrip

**C2: Old checkpoint compatibility**
- [INVARIANT] Checkpoint without tactile params loads successfully with `tactile_enabled=False`
- [FALSIFICATION] Create checkpoint without tactile, load with both True/False settings
- [PROPERTY TYPE] InvariantPreservation

## Verification Plan

1. **Unit Test**: TactileEncoder forward pass with shape/dtype assertions
2. **PBT Tests**: Implement all properties above using hypothesis
3. **Integration Test**: Training loop with tactile-enabled config
4. **Backward Compatibility**: Verify pi05_franka_cola_relative_lora unchanged
5. **Data Pipeline**: Validate marker3d flows from Zarr to model input
