"""Policy transforms for Franka Panda robot."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    base_image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    state_key: str = "observation/state"
    state_dim: int | None = 7  # None means use all dimensions

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data[self.base_image_key])
        wrist_image = _parse_image(data[self.wrist_image_key])
        state = np.asarray(data[self.state_key])
        if self.state_dim is not None:
            state = state[..., : self.state_dim]

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), wrist_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
