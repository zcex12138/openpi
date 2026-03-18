"""Residual-policy action representations and conversions."""

from __future__ import annotations

import numpy as np

from openpi.shared.rotation import quat_to_rotate6d
from openpi.shared.rotation import rotate6d_to_quat
from openpi.shared.rotation import rotate6d_to_rotmat
from openpi.shared.rotation import rotmat_to_rotate6d


def canonicalize_quaternion_sign(quat: np.ndarray) -> np.ndarray:
    """Flip quaternion sign so the dominant component is always positive."""
    arr = np.asarray(quat)
    if arr.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {arr.shape}")
    dominant_idx = np.argmax(np.abs(arr), axis=-1)
    dominant = np.take_along_axis(arr, dominant_idx[..., None], axis=-1)
    return np.where(dominant < 0, -arr, arr)


def pose8_to_pose10(pose8: np.ndarray) -> np.ndarray:
    """Convert [xyz, quat(wxyz), gripper] to [xyz, r6d, gripper]."""
    pose = np.asarray(pose8, dtype=np.float32)
    if pose.shape[-1] != 8:
        raise ValueError(f"Expected pose8 last dim 8, got {pose.shape}")
    xyz = pose[..., :3]
    quat = canonicalize_quaternion_sign(pose[..., 3:7])
    gripper = pose[..., 7:8]
    r6d = quat_to_rotate6d(quat)
    return np.concatenate([xyz, r6d, gripper], axis=-1).astype(np.float32)


def pose10_to_pose8(pose10: np.ndarray) -> np.ndarray:
    """Convert [xyz, r6d, gripper] to [xyz, quat(wxyz), gripper]."""
    pose = np.asarray(pose10, dtype=np.float32)
    if pose.shape[-1] != 10:
        raise ValueError(f"Expected pose10 last dim 10, got {pose.shape}")
    xyz = pose[..., :3]
    r6d = pose[..., 3:9]
    gripper = pose[..., 9:10]
    quat = canonicalize_quaternion_sign(rotate6d_to_quat(r6d))
    return np.concatenate([xyz, quat, gripper], axis=-1).astype(np.float32)


def build_input_features(state_pose10: np.ndarray, base_action_pose10: np.ndarray) -> np.ndarray:
    """Build residual-model inputs from canonical pose10 state and base action."""
    state = np.asarray(state_pose10, dtype=np.float32)
    base_action = np.asarray(base_action_pose10, dtype=np.float32)
    if state.shape[-1] != 10:
        raise ValueError(f"Expected state pose10 last dim 10, got {state.shape}")
    if base_action.shape[-1] != 10:
        raise ValueError(f"Expected base action pose10 last dim 10, got {base_action.shape}")
    return np.concatenate([state, base_action], axis=-1).astype(np.float32)


def encode_residual_action(base_action_pose10: np.ndarray, corrected_action_pose10: np.ndarray) -> np.ndarray:
    """Encode corrected_action relative to base_action in canonical pose10 space."""
    base10 = np.asarray(base_action_pose10, dtype=np.float32)
    corrected10 = np.asarray(corrected_action_pose10, dtype=np.float32)
    if base10.shape[-1] != 10:
        raise ValueError(f"Expected base action pose10 last dim 10, got {base10.shape}")
    if corrected10.shape[-1] != 10:
        raise ValueError(f"Expected corrected action pose10 last dim 10, got {corrected10.shape}")

    delta_xyz = corrected10[..., :3] - base10[..., :3]
    base_rot = rotate6d_to_rotmat(base10[..., 3:9])
    corrected_rot = rotate6d_to_rotmat(corrected10[..., 3:9])
    delta_rot = np.einsum("...ij,...jk->...ik", corrected_rot, np.swapaxes(base_rot, -2, -1))
    delta_r6d = rotmat_to_rotate6d(delta_rot)
    delta_gripper = corrected10[..., 9:10] - base10[..., 9:10]
    return np.concatenate([delta_xyz, delta_r6d, delta_gripper], axis=-1).astype(np.float32)


def decode_residual_pose10(base_action_pose10: np.ndarray, residual10: np.ndarray) -> np.ndarray:
    """Decode residual target back to an absolute [xyz, r6d, gripper] pose."""
    base10 = np.asarray(base_action_pose10, dtype=np.float32)
    if base10.shape[-1] != 10:
        raise ValueError(f"Expected base action pose10 last dim 10, got {base10.shape}")
    residual = np.asarray(residual10, dtype=np.float32)
    if residual.shape[-1] != 10:
        raise ValueError(f"Expected residual last dim 10, got {residual.shape}")

    target_xyz = base10[..., :3] + residual[..., :3]
    base_rot = rotate6d_to_rotmat(base10[..., 3:9])
    delta_rot = rotate6d_to_rotmat(residual[..., 3:9])
    target_rot = np.einsum("...ij,...jk->...ik", delta_rot, base_rot)
    target_r6d = rotmat_to_rotate6d(target_rot)
    target_gripper = base10[..., 9:10] + residual[..., 9:10]
    return np.concatenate([target_xyz, target_r6d, target_gripper], axis=-1).astype(np.float32)
