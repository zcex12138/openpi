"""Training loop for residual-policy models."""

from __future__ import annotations

import dataclasses
import gc
import logging
from pathlib import Path
import random
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb
from torch import nn

from residual_policy.config import ResidualTrainingConfig
from residual_policy.dataset import ResidualDataset
from residual_policy.dataset import build_cr_dagger_like_sample_indices
from residual_policy.dataset import compute_normalization_stats
from residual_policy.dataset import load_residual_zarr
from residual_policy.dataset import split_episode_indices
from residual_policy.model import build_residual_model
from residual_policy.model import get_model_kind_from_metadata

logger = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_name(cfg: ResidualTrainingConfig) -> str:
    return cfg.exp_name


def _output_dir(cfg: ResidualTrainingConfig) -> Path:
    return Path(cfg.checkpoint_dir) / _run_name(cfg)


def _checkpoint_dir(output_dir: Path, name: str) -> Path:
    return output_dir / name


def _save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ResidualTrainingConfig,
    *,
    epoch: int,
    best_val_loss: float | None,
    stats: dict[str, np.ndarray],
) -> None:
    tmp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(stats, tmp_dir / "residual_stats.pt")
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": dataclasses.asdict(cfg),
            "model_kind": cfg.model.kind,
            "action_representation": "xyz_r6d_gripper",
            "sampling_style": "cr_dagger_like",
            "weighted_sampling": cfg.sampling.weighted_sampling,
            "correction_horizon": cfg.sampling.correction_horizon,
            "num_initial_episodes": cfg.sampling.num_initial_episodes,
        },
        tmp_dir / "metadata.pt",
    )

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def _load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None, dict[str, np.ndarray]]:
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    safetensors.torch.load_model(model, checkpoint_dir / "model.safetensors", device=str(device))
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device, weights_only=False))
    stats = torch.load(checkpoint_dir / "residual_stats.pt", map_location="cpu", weights_only=False)
    return int(metadata["epoch"]), metadata.get("best_val_loss"), stats


def _stats_match(current_stats: dict[str, np.ndarray], checkpoint_stats: dict[str, np.ndarray]) -> bool:
    if current_stats.keys() != checkpoint_stats.keys():
        return False
    return all(
        np.array_equal(np.asarray(current_stats[key]), np.asarray(checkpoint_stats[key])) for key in current_stats
    )


def _build_datasets(cfg: ResidualTrainingConfig) -> tuple[ResidualDataset, ResidualDataset | None, dict[str, np.ndarray]]:
    require_xense = cfg.model.kind == "xense_single_step_mlp" and cfg.model.xense_required
    data = load_residual_zarr(
        cfg.zarr_path,
        require_xense=require_xense,
        xense_shape=cfg.model.xense_shape,
    )
    train_episode_indices, val_episode_indices = split_episode_indices(
        data.num_episodes, cfg.sampling.val_ratio, cfg.sampling.seed
    )

    train_indices = build_cr_dagger_like_sample_indices(
        data,
        train_episode_indices,
        weighted_sampling=cfg.sampling.weighted_sampling,
        correction_horizon=cfg.sampling.correction_horizon,
        regular_valid_sampling=cfg.sampling.regular_valid_sampling,
        num_initial_episodes=cfg.sampling.num_initial_episodes,
        deduplicate=False,
    )
    stats_obj = compute_normalization_stats(data, train_indices)
    stats = {
        "input_mean": stats_obj.input_mean,
        "input_std": stats_obj.input_std,
        "target_mean": stats_obj.target_mean,
        "target_std": stats_obj.target_std,
    }
    train_dataset = ResidualDataset(data, train_indices, model_kind=cfg.model.kind, stats=stats_obj)

    if not val_episode_indices:
        return train_dataset, None, stats

    val_indices = build_cr_dagger_like_sample_indices(
        data,
        val_episode_indices,
        weighted_sampling=cfg.sampling.weighted_sampling,
        correction_horizon=cfg.sampling.correction_horizon,
        regular_valid_sampling=cfg.sampling.regular_valid_sampling,
        num_initial_episodes=cfg.sampling.num_initial_episodes,
        deduplicate=True,
    )
    if not val_indices:
        return train_dataset, None, stats
    val_dataset = ResidualDataset(data, val_indices, model_kind=cfg.model.kind, stats=stats_obj)
    return train_dataset, val_dataset, stats


def _forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch["targets"].to(device)
    if "inputs" in batch:
        preds = model(batch["inputs"].to(device))
        return preds, targets
    preds = model(batch["low_dim_inputs"].to(device), batch["xense"].to(device))
    return preds, targets


def _evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            preds, targets = _forward_batch(model, batch, device)
            loss = torch.nn.functional.mse_loss(preds, targets)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train(cfg: ResidualTrainingConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, stats = _build_datasets(cfg)
    logger.info("Training samples: %d", len(train_dataset))
    if val_dataset is not None:
        logger.info("Validation samples: %d", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

    model = build_residual_model(cfg.model, low_dim_input_dim=20, output_dim=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    best_val_loss: float | None = None
    latest_dir = _checkpoint_dir(output_dir, "latest")
    if cfg.resume and latest_dir.exists():
        resume_metadata = torch.load(latest_dir / "metadata.pt", map_location="cpu", weights_only=False)
        checkpoint_kind = get_model_kind_from_metadata(resume_metadata)
        if checkpoint_kind != cfg.model.kind:
            raise ValueError(
                "Resume checkpoint model_kind mismatch: "
                f"checkpoint={checkpoint_kind!r}, config={cfg.model.kind!r}"
            )
        start_epoch, best_val_loss, checkpoint_stats = _load_checkpoint(latest_dir, model, optimizer, device)
        logger.info("Resumed from epoch %d", start_epoch)
        if not _stats_match(stats, checkpoint_stats):
            logger.warning(
                "Resume dataset normalization stats differ from the checkpoint; "
                "continuing with stats recomputed from the current training data."
            )

    run = None
    if cfg.wandb_enabled:
        run = wandb.init(project=cfg.wandb_project, name=_run_name(cfg), config=dataclasses.asdict(cfg))

    global_step = 0
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_losses: list[float] = []
        start_time = time.time()
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}") as progress:
            for batch in progress:
                preds, targets = _forward_batch(model, batch, device)
                loss = torch.nn.functional.mse_loss(preds, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                epoch_losses.append(loss_value)
                global_step += 1
                if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
                    progress.set_postfix({"loss": f"{loss_value:.4f}"})
                    if run is not None:
                        wandb.log({"train_step_loss": loss_value, "global_step": global_step}, step=global_step)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = _evaluate(model, val_loader, device) if val_loader is not None else None
        elapsed = time.time() - start_time
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%s elapsed=%.1fs",
            epoch + 1,
            train_loss,
            f"{val_loss:.6f}" if val_loss is not None else "n/a",
            elapsed,
        )
        if run is not None:
            payload = {"epoch": epoch + 1, "train_loss": train_loss, "elapsed_s": elapsed}
            if val_loss is not None:
                payload["val_loss"] = val_loss
            wandb.log(payload, step=global_step)

        monitor = val_loss if val_loss is not None else train_loss
        if best_val_loss is None or monitor < best_val_loss:
            best_val_loss = monitor
            _save_checkpoint(_checkpoint_dir(output_dir, "best"), model, optimizer, cfg, epoch=epoch + 1, best_val_loss=best_val_loss, stats=stats)

        if (epoch + 1) % cfg.save_every_epochs == 0 or epoch + 1 == cfg.num_epochs:
            _save_checkpoint(latest_dir, model, optimizer, cfg, epoch=epoch + 1, best_val_loss=best_val_loss, stats=stats)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if run is not None:
        wandb.finish()
    return output_dir
