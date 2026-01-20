import atexit
from collections.abc import Sequence
import logging
import math
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._dump_every = int(os.environ.get("OPENPI_DUMP_IMAGES_EVERY", "0"))
        self._dump_dir = pathlib.Path(os.environ.get("OPENPI_DUMP_IMAGE_DIR", "policy_records/processed_images"))
        self._dump_step = 0
        self._dump_video_every = int(os.environ.get("OPENPI_DUMP_VIDEO_EVERY", str(self._dump_every)))
        self._dump_video_dir = pathlib.Path(os.environ.get("OPENPI_DUMP_VIDEO_DIR", str(self._dump_dir)))
        self._dump_video_fps = float(os.environ.get("OPENPI_DUMP_VIDEO_FPS", "30"))
        self._dump_video_enabled = os.environ.get("OPENPI_DUMP_VIDEO", "0") not in ("0", "", "false", "False")
        self._video_writers: dict[str, Any] = {}
        self._allow_dump_images = True

        if self._dump_every > 0:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
        if self._dump_video_enabled and self._dump_video_every > 0:
            self._dump_video_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(self._close_video_writers)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self, obs: dict, *, noise: np.ndarray | None = None, return_processed_inputs: bool = False
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        processed_inputs = inputs
        self._maybe_dump_images(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if return_processed_inputs:
            outputs["__openpi_processed_inputs"] = processed_inputs
        return outputs

    def _maybe_dump_images(self, inputs: dict) -> None:
        if not self._allow_dump_images:
            return
        if self._dump_every <= 0 and (not self._dump_video_enabled or self._dump_video_every <= 0):
            return
        step = self._dump_step
        self._dump_step += 1
        dump_image = self._dump_every > 0 and step % self._dump_every == 0
        dump_video = self._dump_video_enabled and self._dump_video_every > 0 and step % self._dump_video_every == 0
        if not dump_image and not dump_video:
            return
        images = inputs.get("image")
        if not images:
            return
        for name, image in images.items():
            arr = np.asarray(image)
            if arr.ndim == 4:
                arr = arr[0]
            arr = self._prepare_image_for_saving(arr)
            if dump_image:
                out_path = self._dump_dir / f"step_{step:06d}_{name}.png"
                Image.fromarray(arr).save(out_path)
            if dump_video:
                self._append_video_frame(name, arr)

    def _append_video_frame(self, name: str, frame: np.ndarray) -> None:
        writer = self._video_writers.get(name)
        if writer is None:
            writer = self._create_video_writer(name)
            if writer is None:
                return
            self._video_writers[name] = writer
        try:
            writer.append_data(frame)
        except Exception:
            logging.exception("Failed to append frame to video for %s", name)

    def _create_video_writer(self, name: str) -> Any | None:
        try:
            out_path = self._dump_video_dir / f"{name}.mp4"
            return imageio.get_writer(out_path, fps=self._dump_video_fps)
        except Exception:
            logging.exception("Failed to create video writer for %s", name)
            return None

    def _close_video_writers(self) -> None:
        for writer in self._video_writers.values():
            try:
                writer.close()
            except Exception:
                logging.exception("Failed to close video writer")

    @staticmethod
    def _prepare_image_for_saving(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 HWC for saving."""
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        if np.issubdtype(image.dtype, np.floating):
            # Assume image is in [-1, 1] or [0, 1]. Clamp and convert to [0, 255].
            image = np.clip(image, -1.0, 1.0)
            image = ((image + 1.0) * 0.5 * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        record_root = os.environ.get("OPENPI_RECORD_DIR", record_dir)
        self._record_root = pathlib.Path(record_root)
        self._config_name = self._get_config_name(policy)
        self._record_dir = self._record_root / self._config_name
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._episode_records: list[dict[str, Any]] = []
        self._episode_index: int | None = None
        self._source_episode_index: int | None = None
        self._episode_dir: pathlib.Path | None = None
        self._episode_record_path: pathlib.Path | None = None
        self._episode_index_offset: int | None = None
        self._video_fps_override = self._get_video_fps_override()
        self._video_writer: Any | None = None
        self._video_path: pathlib.Path | None = None
        self._video_frames: list[np.ndarray] = []
        self._episode_start_time: float | None = None
        self._episode_end_time: float | None = None

        logging.info("Dumping policy records to: %s", self._record_dir)
        atexit.register(self._finalize_episode_video)
        self._disable_policy_dumps()

    @staticmethod
    def _get_config_name(policy: _base_policy.BasePolicy) -> str:
        metadata = getattr(policy, "metadata", None)
        if isinstance(metadata, dict):
            name = metadata.get("config_name")
            if isinstance(name, str):
                name = name.strip()
                if name:
                    name = name.replace(os.sep, "_")
                    if os.altsep:
                        name = name.replace(os.altsep, "_")
                    return name
        return "unknown"

    @staticmethod
    def _get_video_fps_override() -> float | None:
        raw = os.environ.get("OPENPI_RECORD_VIDEO_FPS")
        if raw is None or raw == "":
            return None
        try:
            fps = float(raw)
        except ValueError:
            logging.warning("Invalid OPENPI_RECORD_VIDEO_FPS=%r, falling back to auto fps", raw)
            return None
        if not math.isfinite(fps) or fps <= 0:
            logging.warning("Invalid OPENPI_RECORD_VIDEO_FPS=%r, falling back to auto fps", raw)
            return None
        return fps

    def _disable_policy_dumps(self) -> None:
        if isinstance(self._policy, Policy):
            self._policy._dump_every = 0
            self._policy._dump_video_every = 0
            self._policy._dump_video_enabled = False
            self._policy._allow_dump_images = False

    @staticmethod
    def _get_episode_info(obs: dict) -> tuple[int | None, int | None]:
        meta = obs.get("__openpi")
        if isinstance(meta, dict):
            episode_index = meta.get("episode_index")
            episode_step = meta.get("episode_step")
            if isinstance(episode_index, (int, np.integer)):
                episode_index = int(episode_index)
            else:
                episode_index = None
            if isinstance(episode_step, (int, np.integer)):
                episode_step = int(episode_step)
            else:
                episode_step = None
            return episode_index, episode_step
        return None, None

    def _get_next_episode_index(self) -> int:
        max_index = -1
        for path in self._record_dir.iterdir():
            if not path.is_dir():
                continue
            name = path.name
            if not name.startswith("episode_"):
                continue
            suffix = name[len("episode_") :]
            if len(suffix) != 3 or not suffix.isdigit():
                continue
            max_index = max(max_index, int(suffix))
        return max_index + 1

    def _ensure_episode(self, episode_index: int) -> None:
        if self._episode_index == episode_index:
            return
        if self._episode_index is not None:
            self._finalize_episode_video()
        self._episode_index = episode_index
        self._episode_records = []
        self._video_frames = []
        self._episode_start_time = None
        self._episode_end_time = None
        self._episode_dir = self._record_dir / f"episode_{episode_index:03d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._episode_record_path = self._episode_dir / "records.npy"
        if self._episode_record_path.exists():
            logging.info("Overwriting existing policy record file: %s", self._episode_record_path)
        self._close_video_writer()
        self._video_path = self._episode_dir / "video.mp4"
        if self._video_path.exists():
            logging.info("Overwriting existing policy video file: %s", self._video_path)

    def _append_video_frame(self, processed_inputs: dict) -> None:
        images = processed_inputs.get("image")
        if not isinstance(images, dict) or not images:
            return

        frames: list[np.ndarray] = []
        for name in sorted(images.keys()):
            frame = self._prepare_video_frame(images[name])
            if frame is not None:
                frames.append(frame)

        if not frames:
            return

        grid = self._compose_grid(frames)
        if grid is None:
            return

        if self._video_fps_override is None:
            now = time.monotonic()
            if self._episode_start_time is None:
                self._episode_start_time = now
            self._episode_end_time = now
            self._video_frames.append(grid)
            return

        if self._video_writer is None:
            self._video_writer = self._create_video_writer(self._video_fps_override)
            if self._video_writer is None:
                return
        try:
            self._video_writer.append_data(grid)
        except Exception:
            logging.exception("Failed to append frame to policy video")

    @staticmethod
    def _prepare_video_frame(image: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim not in (2, 3):
            return None

        arr = Policy._prepare_image_for_saving(arr)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[..., :3]
        return arr

    @staticmethod
    def _compose_grid(frames: list[np.ndarray]) -> np.ndarray | None:
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        grid_size = math.ceil(math.sqrt(len(frames)))
        max_h = max(frame.shape[0] for frame in frames)
        max_w = max(frame.shape[1] for frame in frames)
        canvas = np.zeros((grid_size * max_h, grid_size * max_w, 3), dtype=np.uint8)
        for idx, frame in enumerate(frames):
            row = idx // grid_size
            col = idx % grid_size
            h, w = frame.shape[:2]
            canvas[
                row * max_h : row * max_h + h,
                col * max_w : col * max_w + w,
                :3,
            ] = frame[..., :3]
        return canvas

    def _create_video_writer(self, fps: float) -> Any | None:
        if self._video_path is None:
            return None
        try:
            return imageio.get_writer(self._video_path, fps=fps)
        except Exception:
            logging.exception("Failed to create video writer for %s", self._video_path)
            return None

    def _close_video_writer(self) -> None:
        if self._video_writer is None:
            return
        try:
            self._video_writer.close()
        except Exception:
            logging.exception("Failed to close video writer")
        self._video_writer = None

    def _finalize_episode_video(self) -> None:
        if self._video_fps_override is not None:
            self._close_video_writer()
            return
        if self._video_path is None or not self._video_frames:
            return

        fps = 30.0
        if self._episode_start_time is not None and self._episode_end_time is not None:
            duration = self._episode_end_time - self._episode_start_time
            if duration > 0 and len(self._video_frames) > 1:
                fps = len(self._video_frames) / duration
                if not math.isfinite(fps) or fps <= 0:
                    fps = 30.0

        try:
            writer = self._create_video_writer(fps)
            if writer is None:
                return
            for frame in self._video_frames:
                writer.append_data(frame)
            writer.close()
        except Exception:
            logging.exception("Failed to write policy video for %s", self._video_path)
        finally:
            self._video_frames = []
            self._episode_start_time = None
            self._episode_end_time = None

    def finalize_episode(self) -> None:
        """Finalize the current episode recording (used when sessions end)."""
        self._finalize_episode_video()
        self._episode_index = None
        self._source_episode_index = None

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        processed_inputs: dict | None = None
        if isinstance(self._policy, Policy):
            results = self._policy.infer(obs, return_processed_inputs=True)
            if isinstance(results, dict):
                processed_inputs = results.pop("__openpi_processed_inputs", None)
        else:
            results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        episode_index, _ = self._get_episode_info(obs)
        if episode_index is None:
            episode_index = 0
        if (
            self._episode_index_offset is None
            or self._episode_index is None
            or self._source_episode_index != episode_index
        ):
            next_index = self._get_next_episode_index()
            self._episode_index_offset = next_index - episode_index
            self._source_episode_index = episode_index
        episode_index += self._episode_index_offset
        self._ensure_episode(episode_index)

        self._episode_records.append(data)
        if self._episode_record_path is not None:
            np.save(self._episode_record_path, np.asarray(self._episode_records, dtype=object))

        if isinstance(processed_inputs, dict):
            self._append_video_frame(processed_inputs)
        return results
