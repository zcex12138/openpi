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
        "observation/state": np.random.rand(8),
        "observation/tactile": np.random.rand(26, 14, 3).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _normalize_quat_sign(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion sign so that the dominant component is positive.

    Quaternion q and -q represent the same rotation. This function ensures
    consistent sign convention by making the component with largest absolute
    value positive. This matches the convention used in ShiftedStateToAction
    and ensures state input is consistent with action targets during training.

    Args:
        quat: Quaternion array of shape (4,) in (qw, qx, qy, qz) order.

    Returns:
        Normalized quaternion with dominant component positive.
    """
    quat = np.asarray(quat, dtype=np.float32)
    dominant_idx = np.argmax(np.abs(quat), axis=-1)
    dominant_component = np.take_along_axis(quat, np.expand_dims(dominant_idx, axis=-1), axis=-1)
    sign = np.where(dominant_component < 0, -1.0, 1.0).astype(quat.dtype, copy=False)
    return quat * sign


def _extract_pose10_state(
    state: np.ndarray,
    *,
    normalize_quat_sign: bool,
    quat_indices: tuple[int, int, int, int],
) -> np.ndarray:
    """Convert legacy pose8 state to canonical pose10 when needed."""
    state = np.asarray(state, dtype=np.float32)
    raw_state_dim = state.shape[-1]

    if raw_state_dim in (8, 14):
        from openpi.shared.rotation import quat_to_rotate6d

        position = state[..., :3]
        quat_slice = slice(quat_indices[0], quat_indices[-1] + 1)
        quat = state[..., quat_slice]
        if normalize_quat_sign:
            quat = _normalize_quat_sign(quat)
        gripper = state[..., 7:8]
        r6d = quat_to_rotate6d(quat)
        return np.concatenate([position, r6d, gripper], axis=-1)

    if raw_state_dim >= 10:
        return state[..., :10].copy()

    raise ValueError(f"Expected Franka state with at least 8 dims, got {state.shape}")


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Transform for Franka robot inputs.

    Handles image parsing, state extraction, and quaternion normalization before
    the Franka rotate6d conversion step runs in the data transform pipeline.
    When `state_dim=10`, legacy pose8 inputs are promoted to canonical pose10.
    """

    model_type: _model.ModelType
    base_image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    state_key: str = "observation/state"
    state_dim: int | None = 7  # None means use all dimensions
    normalize_quat_sign: bool = True  # Normalize quaternion sign for consistency
    quat_indices: tuple[int, int, int, int] = (3, 4, 5, 6)  # (qw, qx, qy, qz) indices
    tactile_key: str | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data[self.base_image_key])
        wrist_image = _parse_image(data[self.wrist_image_key])
        state = np.asarray(data[self.state_key]).copy()
        if self.state_dim == 10:
            state = _extract_pose10_state(
                state,
                normalize_quat_sign=self.normalize_quat_sign,
                quat_indices=self.quat_indices,
            )
        elif self.state_dim is not None:
            state = state[..., : self.state_dim]

            # Normalize quaternion sign to match action target convention.
            # This ensures state input and action target use consistent quaternion signs,
            # preventing the model from learning incorrect rotation mappings.
            if self.normalize_quat_sign and self.state_dim >= 7:
                quat_slice = slice(self.quat_indices[0], self.quat_indices[-1] + 1)
                state[..., quat_slice] = _normalize_quat_sign(state[..., quat_slice])

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

        if self.tactile_key is not None and self.tactile_key in data:
            inputs["tactile"] = np.asarray(data[self.tactile_key], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
