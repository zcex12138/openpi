"""Vectorized rotation conversions: quaternion ↔ rotation matrix ↔ 6D rotation.

All functions support arbitrary batch dimensions (...).
Internal computation uses float64 for numerical stability; output is cast back to input dtype.

Convention: quaternions are (w, x, y, z) order.
Reference: Zhou et al. 2019, "On the Continuity of Rotation Representations in Neural Networks"
"""

import numpy as np


def _safe_normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def _orthogonal_fallback(b1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return a vector guaranteed non-parallel to b1."""
    ex = np.zeros_like(b1)
    ex[..., 0] = 1.0
    cross_ex = np.cross(b1, ex, axis=-1)
    norm_ex = np.linalg.norm(cross_ex, axis=-1, keepdims=True)

    ey = np.zeros_like(b1)
    ey[..., 1] = 1.0
    cross_ey = np.cross(b1, ey, axis=-1)

    return np.where(norm_ex > eps, cross_ex, cross_ey)


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """(…, 4) wxyz quaternion → (…, 3, 3) rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    q = _safe_normalize(q)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R.astype(quat.dtype)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(…, 3, 3) rotation matrix → (…, 4) wxyz quaternion.

    Shepperd's method with sqrt(max(x, 0)) guard.
    Output is L2-normalized and sign-canonicalized (w >= 0).
    """
    r = np.asarray(R, dtype=np.float64)
    batch_shape = r.shape[:-2]

    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]

    # Four candidates for max stability
    s0 = 1.0 + trace
    s1 = 1.0 + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2]
    s2 = 1.0 - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2]
    s3 = 1.0 - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2]

    # Branch 0: trace is largest
    sq0 = np.sqrt(np.maximum(s0, 0.0))
    w0 = 0.5 * sq0
    x0 = (r[..., 2, 1] - r[..., 1, 2]) / np.maximum(2.0 * sq0, 1e-12)
    y0 = (r[..., 0, 2] - r[..., 2, 0]) / np.maximum(2.0 * sq0, 1e-12)
    z0 = (r[..., 1, 0] - r[..., 0, 1]) / np.maximum(2.0 * sq0, 1e-12)

    # Branch 1: R00 is largest diagonal
    sq1 = np.sqrt(np.maximum(s1, 0.0))
    x1 = 0.5 * sq1
    w1 = (r[..., 2, 1] - r[..., 1, 2]) / np.maximum(2.0 * sq1, 1e-12)
    y1 = (r[..., 0, 1] + r[..., 1, 0]) / np.maximum(2.0 * sq1, 1e-12)
    z1 = (r[..., 0, 2] + r[..., 2, 0]) / np.maximum(2.0 * sq1, 1e-12)

    # Branch 2: R11 is largest diagonal
    sq2 = np.sqrt(np.maximum(s2, 0.0))
    y2 = 0.5 * sq2
    w2 = (r[..., 0, 2] - r[..., 2, 0]) / np.maximum(2.0 * sq2, 1e-12)
    x2 = (r[..., 0, 1] + r[..., 1, 0]) / np.maximum(2.0 * sq2, 1e-12)
    z2 = (r[..., 1, 2] + r[..., 2, 1]) / np.maximum(2.0 * sq2, 1e-12)

    # Branch 3: R22 is largest diagonal
    sq3 = np.sqrt(np.maximum(s3, 0.0))
    z3 = 0.5 * sq3
    w3 = (r[..., 1, 0] - r[..., 0, 1]) / np.maximum(2.0 * sq3, 1e-12)
    x3 = (r[..., 0, 2] + r[..., 2, 0]) / np.maximum(2.0 * sq3, 1e-12)
    y3 = (r[..., 1, 2] + r[..., 2, 1]) / np.maximum(2.0 * sq3, 1e-12)

    # Stack all branches: (…, 4, 4) where axis=-2 is branch, axis=-1 is wxyz
    all_w = np.stack([w0, w1, w2, w3], axis=-1)
    all_x = np.stack([x0, x1, x2, x3], axis=-1)
    all_y = np.stack([y0, y1, y2, y3], axis=-1)
    all_z = np.stack([z0, z1, z2, z3], axis=-1)
    all_q = np.stack([all_w, all_x, all_y, all_z], axis=-1)  # (…, 4_branches, 4_wxyz)

    # Select best branch
    discriminants = np.stack([s0, s1, s2, s3], axis=-1)  # (…, 4)
    best = np.argmax(discriminants, axis=-1)  # (…,)

    # Gather best branch
    idx = best[..., np.newaxis, np.newaxis]  # (…, 1, 1)
    idx = np.broadcast_to(idx, batch_shape + (1, 4))
    q = np.take_along_axis(all_q, idx, axis=-2).squeeze(-2)  # (…, 4)

    # L2-normalize
    q = _safe_normalize(q)

    # Sign-canonicalize: w >= 0
    sign = np.where(q[..., 0:1] < 0, -1.0, 1.0)
    q = q * sign

    return q.astype(R.dtype)


def rotmat_to_rotate6d(R: np.ndarray) -> np.ndarray:
    """(…, 3, 3) → (…, 6). First two columns, flattened."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1).astype(R.dtype)


def rotate6d_to_rotmat(r6d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(…, 6) → (…, 3, 3). Gram-Schmidt orthogonalization with collinear fallback."""
    x = np.asarray(r6d, dtype=np.float64)
    a1, a2 = x[..., :3], x[..., 3:6]

    b1 = _safe_normalize(a1, eps)
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    u2_norm = np.linalg.norm(u2, axis=-1, keepdims=True)
    fallback = _orthogonal_fallback(b1, eps)
    b2 = np.where(u2_norm > eps, _safe_normalize(u2, eps), _safe_normalize(fallback, eps))
    b3 = np.cross(b1, b2, axis=-1)
    # Re-orthogonalize for maximum precision
    b2 = np.cross(b3, b1, axis=-1)

    return np.stack([b1, b2, b3], axis=-1).astype(r6d.dtype)


def quat_to_rotate6d(quat: np.ndarray) -> np.ndarray:
    """(…, 4) wxyz → (…, 6)."""
    return rotmat_to_rotate6d(quat_to_rotmat(quat))


def rotate6d_to_quat(r6d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(…, 6) → (…, 4) wxyz."""
    return rotmat_to_quat(rotate6d_to_rotmat(r6d, eps=eps))
