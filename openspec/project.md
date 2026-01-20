# Project Context

## Purpose
openpi is an open-source robotics model repository from Physical Intelligence. It provides vision-language-action
models (pi0, pi0-FAST, pi0.5), pretrained and fine-tuned checkpoints, and tooling for inference, policy serving, and
fine-tuning on robot platforms such as DROID, ALOHA, and LIBERO.

## Tech Stack
- Python 3.11
- JAX-based training/inference (`scripts/train.py`)
- PyTorch training/inference (`scripts/train_pytorch.py`)
- uv for dependency management and scripted runs (`uv run ...`)
- WebSocket client for remote inference (`packages/openpi-client`)
- Transformers (patched for PyTorch support)

## Project Conventions

### Code Style
- 4-space indentation, line length 120
- `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Lint/format with `ruff` and `pre-commit` before PRs

### Architecture Patterns
- Core library lives in `src/openpi/` (models, policies, training, transforms, shared utilities)
- Training configuration is centralized in `src/openpi/training/config.py`
- Runtime entrypoints in `scripts/` (train, serve policy, compute norm stats)
- Remote inference client in `packages/openpi-client/`
- End-to-end examples and platform adapters in `examples/`

### Testing Strategy
- pytest with `testpaths = ["src", "scripts", "packages"]`
- Use `manual` marker for long-running or manual tests
- Prefer targeted runs during iteration (e.g. `uv run pytest path/to/test.py`)

### Git Workflow
- Short, verb-led commit messages; include PR number if applicable
- PRs should pass `pre-commit`, `ruff`, and `pytest`
- Discuss large changes (new robots, major refactors) before implementation

## Domain Context
- Vision-language-action (VLA) robotics models with pretrained checkpoints in GCS
- Normalization stats are required for training; compute via `scripts/compute_norm_stats.py`
- Datasets include DROID, ALOHA, LIBERO; conversion examples provided under `examples/`
- Policy serving and remote inference are supported via a server/client split

## Important Constraints
- NVIDIA GPU required; memory needs vary by inference vs fine-tuning
- Supported OS: Ubuntu 22.04 (others not officially supported)
- Training is single-node only (no multi-node support)
- `GIT_LFS_SKIP_SMUDGE=1` is required for dependency setup
- PyTorch limitations: no pi0-FAST, mixed precision, FSDP, LoRA, or EMA during training

## External Dependencies
- Checkpoints hosted at `gs://openpi-assets`
- Hugging Face datasets and LeRobot data format (for training/fine-tuning)
- Weights & Biases for experiment tracking (optional)
- Transformers with required patching for PyTorch support