"""Residual-policy action representations and conversions."""

from __future__ import annotations

import numpy as np

from openpi.shared.rotation import quat_to_rotate6d
from openpi.shared.rotation import rotate6d_to_quat
from openpi.shared.rotation import rotate6d_to_rotmat
from openpi.shared.rotation import rotmat_to_rotate6d


def as_pose10(pose: np.ndarray) -> np.ndarray:
    """Convert pose to [xyz, r6d, gripper], accepting pose8 or pose10."""
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape[-1] == 10:
        return arr.copy()
    if arr.shape[-1] == 8:
        return pose8_to_pose10(arr)
    raise ValueError(f"Expected pose last dim 8 or 10, got {arr.shape}")


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


def build_input_features(state_pose8: np.ndarray, base_action8: np.ndarray) -> np.ndarray:
    """Build model inputs from current state and base action."""
    state10 = as_pose10(state_pose8)
    base10 = as_pose10(base_action8)
    return np.concatenate([state10, base10], axis=-1).astype(np.float32)


def encode_residual_action(base_action8: np.ndarray, corrected_action8: np.ndarray) -> np.ndarray:
    """Encode corrected_action relative to base_action in xyz+r6d+gripper space."""
    base10 = pose8_to_pose10(base_action8)
    corrected10 = pose8_to_pose10(corrected_action8)

    delta_xyz = corrected10[..., :3] - base10[..., :3]
    base_rot = rotate6d_to_rotmat(base10[..., 3:9])
    corrected_rot = rotate6d_to_rotmat(corrected10[..., 3:9])
    delta_rot = np.einsum("...ij,...jk->...ik", corrected_rot, np.swapaxes(base_rot, -2, -1))
    delta_r6d = rotmat_to_rotate6d(delta_rot)
    delta_gripper = corrected10[..., 9:10] - base10[..., 9:10]
    return np.concatenate([delta_xyz, delta_r6d, delta_gripper], axis=-1).astype(np.float32)


def decode_residual_pose10(base_action8: np.ndarray, residual10: np.ndarray) -> np.ndarray:
    """Decode residual target back to an absolute [xyz, r6d, gripper] pose."""
    base10 = as_pose10(base_action8)
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


def decode_residual_action(base_action8: np.ndarray, residual10: np.ndarray) -> np.ndarray:
    """Decode residual target back to an absolute [xyz, quat, gripper] action."""
    return pose10_to_pose8(decode_residual_pose10(base_action8, residual10))
