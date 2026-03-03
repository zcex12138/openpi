# Tasks: Add Tactile Fusion to π₀.₅

## Phase 1: Data Pipeline Extension

- [x] **T1.1** Modify `examples/convert_zarr_to_lerobot_v2.0.py` to include xense1_marker3d as LeRobot feature
  - Add feature definition: `observation.tactile.xense1_marker3d` with shape (26, 14, 3)
  - Verify with existing Zarr dataset

- [x] **T1.2** Extend `FrankaInputs` transform with tactile support
  - Add optional `tactile_key` parameter
  - Pass through tactile grid when key present

## Phase 2: Model Architecture

- [x] **T2.1** Create `src/openpi/models/tactile_encoder.py`
  - Implement `TactileCNNEncoder` class
  - 2D CNN for fixed (26, 14, 3) tactile grid
  - Configurable hidden_dims, output_dim, dropout

- [x] **T2.2** Extend `Pi0Config` in `src/openpi/models/pi0_config.py`
  - Add fields: tactile_enabled, tactile_hidden_dims, tactile_dropout
  - Default tactile_enabled=False for backward compatibility

- [x] **T2.3** Extend `Observation` dataclass in `src/openpi/models/model.py`
  - Add optional field: tactile
  - Update `from_dict()` to parse tactile if present

- [x] **T2.4** Integrate tactile encoder in `Pi0.__init__()` and `embed_prefix()`
  - Conditional initialization based on config.tactile_enabled
  - Concatenate tactile token with image/language tokens
  - Update attention mask (tactile attends to all prefix tokens)

## Phase 3: Training Configuration

- [x] **T3.1** Create `LeRobotFrankaTactileDataConfig` in `src/openpi/training/config.py`
  - Extend LeRobotFrankaDataConfigV2
  - Add tactile_key repack mapping
  - Configure FrankaInputs with tactile parameters

- [x] **T3.2** Add training config `pi05_franka_tactile_lora`
  - Based on pi05_franka_cola_relative_lora
  - Enable tactile in Pi0Config
  - Use LeRobotFrankaTactileDataConfig

- [x] **T3.3** Update freeze_filter to exclude tactile encoder from freezing
  - Tactile encoder should be trainable during LoRA fine-tuning
  - Note: No changes needed - tactile_encoder path doesn't match `.*llm.*` pattern

## Phase 4: Validation

- [x] **T4.1** Unit test TactileCNNEncoder
  - Test forward pass with batch of tactile grids (26, 14, 3)
  - Test output shape matches PaliGemma width

- [x] **T4.2** Integration test training loop
  - Config loads correctly with tactile_enabled=True
  - Full training validation deferred to manual run with tactile dataset

- [x] **T4.3** Backward compatibility test
  - pi05_franka_cola_relative_lora loads with tactile_enabled=False
  - Pi0Config defaults tactile_enabled=False
  - Existing lora tests pass without regression

## Dependencies

```
T1.1 ──► T1.2 ──► T3.1 ──► T3.2

T2.1 ──► T2.4 ──► T3.2
     │
T2.2 ┘
     │
T2.3 ┘

T3.2 ──► T3.3 ──► T4.2
              │
T4.1 ─────────┘

T4.3 (can run in parallel after T2.4)
```

## Effort Estimate

| Phase | Tasks | Estimated Lines Changed |
|-------|-------|------------------------|
| 1 | T1.1-T1.2 | ~60 |
| 2 | T2.1-T2.4 | ~180 |
| 3 | T3.1-T3.3 | ~80 |
| 4 | T4.1-T4.3 | ~50 (test code) |

**Total: ~370 lines** (excluding test boilerplate)
