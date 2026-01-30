# Proposal: Add Tactile Point Cloud Fusion to π₀.₅ Model

## Summary

Integrate tactile sensing (xense1_marker3d point cloud) into the π₀.₅ model architecture. The tactile point cloud will be encoded via a lightweight CNN encoder, concatenated with VLM visual embeddings, and processed by the action expert for multi-modal policy learning.

## Motivation

Current Franka evaluation pipeline already collects tactile data (xense1_marker3d) but this modality is not utilized during training. Tactile feedback provides crucial contact information for manipulation tasks that cannot be inferred from vision alone.

## Approach

### Architecture Integration

The tactile encoder will output tokens that are concatenated with visual embeddings in `embed_prefix()`, before being passed to the action expert. This follows the existing multi-modal fusion pattern used for images.

```
                                    ┌─────────────────┐
                                    │   SigLIP        │
  Images ───────────────────────────►   (frozen)      ├──► Image Tokens
                                    └─────────────────┘         │
                                                                │
                                    ┌─────────────────┐         │
  marker3d ─────────────────────────►   TactileCNN    ├──► Tactile Token ──► Concat ──► PaliGemma+ActionExpert
  (point cloud)                     │   (trainable)   │         │
                                    └─────────────────┘         │
                                                                │
  Language ─────────────────────────────────────────────► Lang Tokens
```

### Key Design Decisions

1. **Fusion Location**: Token-level fusion in `embed_prefix()` (early fusion)
   - Tactile token joins image + language tokens before attention
   - Action expert sees all modalities simultaneously

2. **Tactile Encoder**: Lightweight CNN
   - Input: `(N, M, 3)` marker3d point cloud → flatten to `(N*M, 3)`
   - Architecture: 1D Conv layers with global pooling
   - Output: Single token (1, embedding_dim) matching PaliGemma width

3. **Backward Compatibility**: Tactile is optional
   - `tactile_enabled: bool = False` in Pi0Config
   - Existing checkpoints work without modification
   - Tactile mask controls attention visibility

4. **Configuration Extension**: New training config
   - `pi05_franka_tactile_lora` based on `pi05_franka_cola_relative_lora`
   - CNN encoder hyperparameters exposed in config

## Scope

### In Scope
- TactileCNNEncoder module in `src/openpi/models/`
- Extended Pi0Config with tactile parameters
- Extended Pi0 model with tactile token embedding
- TactileInputs transform for data preprocessing
- New LeRobotFrankaTactileDataConfig
- New training config `pi05_franka_tactile_lora`
- LeRobot conversion script extension for xense1_marker3d

### Out of Scope
- Force/torque sensor integration (separate tactile modality)
- Tactile prediction as auxiliary task
- Multi-sensor tactile fusion
- Real-time inference optimizations

## Success Criteria

1. Training converges with tactile-enabled config
2. Tactile-disabled config produces identical results to baseline
3. Evaluation script supports tactile observation input
4. Data pipeline correctly loads marker3d from LeRobot dataset

## Risks

| Risk | Mitigation |
|------|------------|
| Variable point cloud size | Pad to max_tactile_points with mask |
| Increased training memory | CNN is lightweight (~1M params) |
| Attention overhead from extra token | Single tactile token (minimal) |

## Dependencies

- xense1_marker3d must be present in LeRobot dataset
- Existing pi05_base checkpoint for weight initialization
