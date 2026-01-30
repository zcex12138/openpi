# Capability: Tactile Point Cloud Fusion for π₀.₅

## Overview

This capability adds tactile sensing (xense1_marker3d point cloud) as a new observation modality for the π₀.₅ model, enabling multi-modal policy learning with vision, language, and touch.

---

## ADDED Requirements

### Requirement: Tactile Encoder

The system SHALL provide a `TactileEncoder` (flax.linen Module, bridged via nnx_bridge.ToNNX) that encodes the fixed-size Xense tactile grid `(26, 14, 3)` into a single embedding token.

**Fixed Architecture:**
- 3 Conv2D layers: kernel=(3,3), stride=1, SAME padding, GELU activation
- Hidden dims: (64, 128, 256) — NOT configurable
- Global Average Pooling → Linear → (batch, 1, paligemma_config.width)
- NaN/Inf handling: `jnp.nan_to_num(input, nan=0.0, posinf=0.0, neginf=0.0)` at encoder entry
- No dropout, no mask

#### Scenario: Encode tactile grid
- Given a tactile grid of shape `(batch, 26, 14, 3)` with dtype float32
- When the encoder processes the input
- Then the output SHALL have shape `(batch, 1, 2048)` matching gemma_2b width
- And the output dtype SHALL be float32

#### Scenario: Handle zero tactile input
- Given a tactile grid where all values are zero
- When the encoder processes the input
- Then the output SHALL be a valid tensor (not NaN or Inf)

#### Scenario: Handle NaN/Inf in input
- Given a tactile grid containing NaN or Inf values
- When the encoder processes the input
- Then the output SHALL be a valid finite tensor
- And `E(x) = E(nan_to_num(x))` SHALL hold

#### Scenario: Batch permutation equivariance
- Given inputs x and a batch permutation π
- When the encoder processes both `E(π(x))` and `π(E(x))`
- Then the results SHALL be identical

### Requirement: Tactile Configuration in Pi0Config

`Pi0Config` SHALL support a single tactile-related configuration field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| tactile_enabled | bool | False | Enable tactile encoder |

No other tactile config fields (hidden_dims, dropout, max_points) — architecture is fixed.

#### Scenario: Backward compatible config creation
- Given a Pi0Config with only existing fields (no tactile fields specified)
- When the config is instantiated
- Then tactile_enabled SHALL be False
- And the model SHALL function identically to before this change

### Requirement: Tactile in Observation Dataclass

The `Observation` dataclass SHALL include an optional tactile field:

```python
tactile: at.Float[ArrayT, "*b 26 14 3"] | None = None
```

No mask field — Xense tactile grid has fixed shape (26, 14, 3).

#### Scenario: Observation without tactile
- Given an observation dict without tactile key
- When Observation.from_dict() is called
- Then tactile SHALL be None
- And no exception SHALL be raised

#### Scenario: Observation with tactile
- Given an observation dict with key "tactile" containing shape (26, 14, 3)
- When Observation.from_dict() is called
- Then the Observation SHALL contain the tactile array
- And the array SHALL preserve dtype float32 and shape (26, 14, 3)

### Requirement: Tactile Token Embedding in Pi0

When tactile_enabled is True, the Pi0 model SHALL embed tactile data as a single token in `embed_prefix()`.

#### Scenario: Tactile token concatenation
- Given a Pi0 model with tactile_enabled=True
- And an observation containing images, language tokens, and tactile
- When embed_prefix() is called
- Then the output tokens SHALL include: [image_tokens, lang_tokens, tactile_token]
- And the tactile token SHALL have full bidirectional attention to all prefix tokens
- And `tokens.shape[1] = tokens_baseline.shape[1] + 1`

#### Scenario: Missing tactile at inference (ERROR)
- Given a Pi0 model with tactile_enabled=True
- And an observation with tactile=None
- When embed_prefix() is called
- Then the model SHALL raise ValueError
- And the error message SHALL indicate tactile data is required

#### Scenario: tactile_enabled=False ignores tactile
- Given a Pi0 model with tactile_enabled=False
- And observations with varying tactile values (including None)
- When embed_prefix() is called
- Then outputs SHALL be identical regardless of tactile content
- And outputs SHALL be identical to baseline model

### Requirement: FrankaInputs Tactile Transform

`FrankaInputs` transform SHALL support optional tactile preprocessing:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tactile_key | str \| None | None | Data dict key for tactile grid |

**Data representation:** Raw u/v/depth (pixel coordinates + depth), no normalization, no augmentation.

#### Scenario: Process tactile when key present
- Given FrankaInputs configured with tactile_key="observation/tactile"
- And input data containing "observation/tactile" with shape (26, 14, 3)
- When the transform is applied
- Then output SHALL contain "tactile" with shape (26, 14, 3) dtype float32

#### Scenario: Skip tactile when key absent
- Given FrankaInputs configured with tactile_key="observation/tactile"
- And input data NOT containing "observation/tactile"
- When the transform is applied
- Then output SHALL NOT contain "tactile"

### Requirement: LeRobot Tactile Feature

The LeRobot conversion script SHALL include xense1_marker3d as a dataset feature.

**Key Mapping:**
- Zarr source: `data/xense1_marker3d`
- LeRobot feature: `observation.tactile.xense1_marker3d`
- Repack target: `observation/tactile`
- Model input: `tactile`

#### Scenario: Convert Zarr with marker3d to LeRobot
- Given a Zarr dataset containing "data/xense1_marker3d" with shape (N, 26, 14, 3)
- When convert_zarr_to_lerobot_v2.0.py is run
- Then the LeRobot dataset SHALL contain feature "observation.tactile.xense1_marker3d"
- And the feature shape SHALL be (26, 14, 3)
- And the feature dtype SHALL be float32

### Requirement: Training Config for Tactile

A training configuration `pi05_franka_tactile_lora` SHALL be available with:
- Model: Pi0Config with tactile_enabled=True
- Data: LeRobotFrankaTactileDataConfig with tactile key mapping
- LoRA: Consistent with pi05_franka_cola_relative_lora
- Trainable params: LoRA + tactile_encoder ONLY
- Learning rate: Same schedule as base LoRA (warmup=500, peak_lr=1.5e-5)
- No modality dropout

#### Scenario: Load tactile training config
- Given the config name "pi05_franka_tactile_lora"
- When get_config() is called
- Then the returned config SHALL have model.tactile_enabled=True
- And the data config SHALL map "observation/tactile" to LeRobot tactile feature

---

## MODIFIED Requirements

### Requirement: Pi0 Freeze Filter (MODIFIED)

The freeze filter returned by `Pi0Config.get_freeze_filter()` SHALL exclude tactile encoder parameters from freezing during LoRA fine-tuning.

#### Scenario: LoRA config with tactile
- Given Pi0Config with paligemma_variant="gemma_2b_lora" and tactile_enabled=True
- When get_freeze_filter() is called
- Then the filter SHALL NOT freeze tactile encoder parameters
- And the filter SHALL freeze PaliGemma non-LoRA parameters as before

### Requirement: Checkpoint Weight Loader (MODIFIED)

`CheckpointWeightLoader` SHALL allow missing tactile encoder parameters when loading pi05_base checkpoint with tactile_enabled=True.

#### Scenario: Load pi05_base with tactile enabled
- Given a checkpoint without tactile encoder parameters
- And a model with tactile_enabled=True
- When the weight loader processes the checkpoint
- Then missing tactile parameters SHALL be randomly initialized
- And all other parameters SHALL be loaded normally
- And a log message SHALL indicate which parameters were randomly initialized

#### Scenario: Shape mismatch detection
- Given a checkpoint with mismatched parameter shapes
- When the weight loader processes the checkpoint
- Then the loader SHALL raise an error (fail-fast in both training and inference)

---

## Constraints

- **Token Budget**: max_token_len=200 for pi05; tactile adds 1 token (negligible impact)
- **Memory**: TactileEncoder ~200K params (0.01% of gemma_2b)
- **Tactile Grid**: Fixed shape (26, 14, 3) from Xense sensor - 364 points total
- **Backward Compatibility**: All existing configs and checkpoints MUST work unchanged
- **Data Representation**: Raw u/v/depth — no normalization, no augmentation
- **Time Sync**: Tactile and images are captured together (hard sync)
- **Framework**: flax.linen + nnx_bridge.ToNNX (consistent with SigLIP/Gemma)
- **Success Criteria**: Training converges + backward compatibility verified
