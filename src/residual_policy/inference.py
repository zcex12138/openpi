"""Online Franka residual-policy inference."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time
from typing import Any

import numpy as np
from openpi.shared import download
from openpi.shared.rotation import quat_to_rotate6d
from openpi.shared.rotation import rotate6d_to_rotmat
from openpi.shared.rotation import rotmat_to_quat
import torch
from openpi_client import base_policy as _base_policy

from residual_policy.config import ResidualModelConfig
from residual_policy.config import build_model_config
from residual_policy.action_repr import build_input_features
from residual_policy.action_repr import canonicalize_quaternion_sign
from residual_policy.action_repr import decode_residual_pose10
from residual_policy.action_repr import pose8_to_pose10
from residual_policy.model import build_residual_model
from residual_policy.model import get_model_kind_from_metadata

_EXPECTED_STATE_DIM = 10
_LEGACY_STATE_DIM = 8
_EXPECTED_ACTION_DIM = 10
_EXPECTED_INPUT_DIM = 20
_EXPECTED_OUTPUT_DIM = 10
_IDENTITY_R6D = quat_to_rotate6d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


@dataclasses.dataclass(frozen=True)
class ResidualInferenceConfig:
    checkpoint_dir: str
    device: str = "auto"
    scale: float = 1.0
    translation_cap_m: float | None = None
    rotation_cap_rad: float | None = None
    gripper_cap: float | None = None
    apply_gripper_delta: bool = True

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ValueError(f"residual scale must be >= 0, got {self.scale}")
        if self.translation_cap_m is not None and self.translation_cap_m <= 0:
            raise ValueError(f"translation cap must be > 0, got {self.translation_cap_m}")
        if self.rotation_cap_rad is not None and self.rotation_cap_rad <= 0:
            raise ValueError(f"rotation cap must be > 0, got {self.rotation_cap_rad}")
        if self.gripper_cap is not None and self.gripper_cap <= 0:
            raise ValueError(f"gripper cap must be > 0, got {self.gripper_cap}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _extract_state_pose10(obs: dict[str, Any]) -> np.ndarray:
    if "observation/state" not in obs:
        raise KeyError("Franka residual inference requires observation['observation/state']")
    state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
    if state.shape[0] < _LEGACY_STATE_DIM:
        raise ValueError(f"Expected observation/state with at least 8 dims, got {state.shape}")
    return pose8_to_pose10(state[:_LEGACY_STATE_DIM])


def _extract_xense_tactile(
    obs: dict[str, Any],
    *,
    expected_shape: tuple[int, int, int],
    required: bool,
) -> np.ndarray:
    tactile = obs.get("observation/tactile")
    if tactile is None:
        if required:
            raise KeyError("Franka residual inference requires observation['observation/tactile']")
        return np.zeros(expected_shape, dtype=np.float32)
    arr = np.asarray(tactile, dtype=np.float32)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"Expected observation/tactile shape {expected_shape}, got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _canonical_pose10_action(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] == _EXPECTED_ACTION_DIM:
            return arr.copy()
        if arr.shape[-1] == _LEGACY_STATE_DIM:
            return pose8_to_pose10(arr)
        raise ValueError(f"Expected action chunk last dim 8 or 10, got {arr.shape}")
    if arr.shape[-1] == _EXPECTED_ACTION_DIM:
        return arr.copy()
    if arr.shape[-1] == _LEGACY_STATE_DIM:
        return pose8_to_pose10(arr)
    raise ValueError(f"Expected action last dim 8 or 10, got {arr.shape}")


def _rotation_angle_from_r6d(r6d: np.ndarray) -> float:
    rot = rotate6d_to_rotmat(np.asarray(r6d, dtype=np.float32))
    trace = float(np.trace(rot))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _scale_rotation_delta(r6d: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        return _IDENTITY_R6D.copy()
    if np.isclose(factor, 1.0):
        return np.asarray(r6d, dtype=np.float32).copy()

    quat = rotmat_to_quat(rotate6d_to_rotmat(np.asarray(r6d, dtype=np.float32))).astype(np.float64)
    quat = canonicalize_quaternion_sign(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    half_angle = float(np.arccos(w))
    sin_half = float(np.sin(half_angle))
    if sin_half < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = quat[1:4] / sin_half

    scaled_half_angle = half_angle * factor
    scaled_quat = np.concatenate(
        [
            np.array([np.cos(scaled_half_angle)], dtype=np.float64),
            axis * np.sin(scaled_half_angle),
        ]
    ).astype(np.float32)
    scaled_quat = canonicalize_quaternion_sign(scaled_quat)
    return quat_to_rotate6d(scaled_quat).astype(np.float32)


def _apply_residual_limits(residual10: np.ndarray, config: ResidualInferenceConfig) -> np.ndarray:
    residual = np.asarray(residual10, dtype=np.float32).copy()

    residual[:3] *= config.scale
    residual[3:9] = _scale_rotation_delta(residual[3:9], config.scale)
    residual[9] *= config.scale

    if config.translation_cap_m is not None:
        translation_norm = float(np.linalg.norm(residual[:3]))
        if translation_norm > config.translation_cap_m:
            residual[:3] *= config.translation_cap_m / max(translation_norm, 1e-6)

    if config.rotation_cap_rad is not None:
        angle = _rotation_angle_from_r6d(residual[3:9])
        if angle > config.rotation_cap_rad:
            residual[3:9] = _scale_rotation_delta(residual[3:9], config.rotation_cap_rad / max(angle, 1e-6))

    if config.gripper_cap is not None:
        residual[9] = float(np.clip(residual[9], -config.gripper_cap, config.gripper_cap))

    return residual.astype(np.float32)


class FrankaResidualStepPolicy(_base_policy.BasePolicy):
    """Wrap a single-step Franka policy with a learned residual correction."""

    def __init__(self, policy: _base_policy.BasePolicy, config: ResidualInferenceConfig) -> None:
        self._policy = policy
        self._config = config
        self._device = _resolve_device(config.device)

        try:
            import safetensors.torch as safetensors_torch
        except ModuleNotFoundError as exc:
            if exc.name != "safetensors":
                raise
            raise ModuleNotFoundError(
                "Residual policy runtime requires the `safetensors` package in the Franka Python environment. "
                "Install it with `uv pip install safetensors`, or disable residual.checkpoint_dir in "
                "`examples/franka/real_env_config.yaml`."
            ) from exc

        checkpoint_dir = Path(download.maybe_download(config.checkpoint_dir))
        metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
        stats = torch.load(checkpoint_dir / "residual_stats.pt", map_location="cpu", weights_only=False)

        if metadata.get("action_representation") != "xyz_r6d_gripper":
            raise ValueError(
                "Unsupported residual checkpoint action_representation="
                f"{metadata.get('action_representation')!r}, expected 'xyz_r6d_gripper'"
            )

        raw_model_cfg = metadata.get("config", {}).get("model", {})
        if isinstance(raw_model_cfg, dict):
            raw_model_cfg = dict(raw_model_cfg)
            raw_model_cfg.setdefault("kind", get_model_kind_from_metadata(metadata))
            model_cfg = build_model_config(raw_model_cfg)
        else:
            model_cfg = ResidualModelConfig(kind=get_model_kind_from_metadata(metadata))
        self._model_cfg = model_cfg
        self._model = build_residual_model(model_cfg, low_dim_input_dim=_EXPECTED_INPUT_DIM, output_dim=_EXPECTED_OUTPUT_DIM).to(
            self._device
        )
        safetensors_torch.load_model(self._model, checkpoint_dir / "model.safetensors", device=str(self._device))
        self._model.eval()

        self._input_mean = np.asarray(stats["input_mean"], dtype=np.float32)
        self._input_std = np.asarray(stats["input_std"], dtype=np.float32)
        self._target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        self._target_std = np.asarray(stats["target_std"], dtype=np.float32)

        if self._input_mean.shape != (_EXPECTED_INPUT_DIM,) or self._input_std.shape != (_EXPECTED_INPUT_DIM,):
            raise ValueError("Residual checkpoint input stats must have shape (20,)")
        if self._target_mean.shape != (_EXPECTED_OUTPUT_DIM,) or self._target_std.shape != (_EXPECTED_OUTPUT_DIM,):
            raise ValueError("Residual checkpoint target stats must have shape (10,)")

    def infer(self, obs: dict) -> dict:
        result = self._policy.infer(obs)
        base_action_raw = np.asarray(result["actions"], dtype=np.float32)
        if base_action_raw.ndim != 1 or base_action_raw.shape[-1] != _EXPECTED_ACTION_DIM:
            raise ValueError(
                "FrankaResidualStepPolicy requires a single-step 10D base action. "
                f"Got shape {base_action_raw.shape}. Wrap it after the execution broker."
            )

        state_pose10 = _extract_state_pose10(obs)
        base_action10 = base_action_raw.copy()
        xense = None
        if self._model_cfg.kind == "xense_single_step_mlp":
            xense = _extract_xense_tactile(
                obs,
                expected_shape=self._model_cfg.xense_shape,
                required=self._model_cfg.xense_required,
            )
        start = time.perf_counter()
        residual10 = self._predict_residual(state_pose10, base_action10, xense=xense)
        limited_residual10 = _apply_residual_limits(residual10, self._config)
        applied_residual10 = limited_residual10.copy()
        if not self._config.apply_gripper_delta:
            applied_residual10[9] = 0.0
        final_pose10 = decode_residual_pose10(base_action10, applied_residual10)
        residual_ms = (time.perf_counter() - start) * 1000.0

        outputs = dict(result)
        outputs["actions"] = final_pose10.astype(np.float32)
        outputs["base_action"] = base_action10.copy()
        outputs["base_action_pose10"] = base_action10.copy()
        outputs["residual_action_pose10"] = applied_residual10.astype(np.float32)

        timing = result.get("policy_timing")
        outputs["policy_timing"] = dict(timing) if isinstance(timing, dict) else {}
        outputs["policy_timing"]["residual_ms"] = residual_ms
        return outputs

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = getattr(self._policy, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _predict_residual(
        self,
        state_pose10: np.ndarray,
        base_action10: np.ndarray,
        *,
        xense: np.ndarray | None = None,
    ) -> np.ndarray:
        inputs = build_input_features(state_pose10, base_action10)
        normalized_inputs = (inputs - self._input_mean) / self._input_std
        input_tensor = torch.from_numpy(normalized_inputs[None, ...]).to(self._device)
        with torch.no_grad():
            if self._model_cfg.kind == "legacy_mlp":
                normalized_residual = self._model(input_tensor).detach().cpu().numpy()[0]
            else:
                if xense is None:
                    raise ValueError("xense tactile input is required for xense_single_step_mlp inference")
                xense_tensor = torch.from_numpy(xense[None, ...]).to(self._device)
                normalized_residual = self._model(input_tensor, xense_tensor).detach().cpu().numpy()[0]
        residual = (normalized_residual * self._target_std) + self._target_mean
        return residual.astype(np.float32)


class FrankaPolicyPose10Wrapper(_base_policy.BasePolicy):
    """Expose Franka policy outputs in absolute [xyz, r6d, gripper] space."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    def infer(self, obs: dict) -> dict:
        result = self._policy.infer(obs)
        return self._convert_outputs(result)

    def infer_realtime(
        self,
        obs: dict,
        *,
        action_prefix: np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> dict:
        infer_realtime = getattr(self._policy, "infer_realtime", None)
        if not callable(infer_realtime):
            return self.infer(obs)

        result = infer_realtime(obs, action_prefix=action_prefix, noise=noise)
        return self._convert_outputs(result)

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = getattr(self._policy, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _convert_outputs(result: dict) -> dict:
        outputs = dict(result)
        outputs["actions"] = _canonical_pose10_action(np.asarray(result["actions"], dtype=np.float32))
        return outputs
