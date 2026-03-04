"""Unit tests for rotation conversions (PBT properties P1-P5)."""

import numpy as np
import pytest

from openpi.shared.rotation import (
    quat_to_rotate6d,
    quat_to_rotmat,
    rotate6d_to_quat,
    rotate6d_to_rotmat,
    rotmat_to_quat,
    rotmat_to_rotate6d,
)

ATOL = 1e-5
RNG = np.random.RandomState(42)


def _random_unit_quats(n: int) -> np.ndarray:
    q = RNG.randn(n, 4).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _rotation_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation equivalence via |dot(q1, q2)| ≈ 1 (handles q/-q ambiguity)."""
    return 1.0 - np.abs(np.sum(q1 * q2, axis=-1))


# --- P1: quat → r6d → quat round-trip ---

def test_p1_roundtrip_random():
    q = _random_unit_quats(100)
    q_rt = rotate6d_to_quat(quat_to_rotate6d(q))
    assert np.all(_rotation_distance(q, q_rt) < ATOL)


def test_p1_identity():
    q = np.array([[1, 0, 0, 0]], dtype=np.float32)
    q_rt = rotate6d_to_quat(quat_to_rotate6d(q))
    assert np.all(_rotation_distance(q, q_rt) < ATOL)


def test_p1_180_degree_rotations():
    quats_180 = np.array([
        [0, 1, 0, 0],  # 180° around x
        [0, 0, 1, 0],  # 180° around y
        [0, 0, 0, 1],  # 180° around z
    ], dtype=np.float32)
    q_rt = rotate6d_to_quat(quat_to_rotate6d(quats_180))
    assert np.all(_rotation_distance(quats_180, q_rt) < ATOL)


# --- P2: rotmat → r6d → rotmat round-trip ---

def test_p2_roundtrip():
    q = _random_unit_quats(50)
    R = quat_to_rotmat(q)
    R_rt = rotate6d_to_rotmat(rotmat_to_rotate6d(R))
    assert np.allclose(R, R_rt, atol=ATOL)


def test_p2_single_matrix():
    R = quat_to_rotmat(np.array([1, 0, 0, 0], dtype=np.float32))
    R_rt = rotate6d_to_rotmat(rotmat_to_rotate6d(R))
    assert np.allclose(R, np.eye(3, dtype=np.float32), atol=ATOL)
    assert np.allclose(R_rt, np.eye(3, dtype=np.float32), atol=ATOL)


# --- P3: SO(3) membership ---

def test_p3_orthogonality_and_det():
    q = _random_unit_quats(100)
    R = quat_to_rotmat(q)

    # R^T R ≈ I
    RtR = np.einsum("...ji,...jk->...ik", R, R)
    I = np.eye(3, dtype=R.dtype)
    assert np.allclose(RtR, I, atol=ATOL)

    # det(R) ≈ +1
    dets = np.linalg.det(R)
    assert np.allclose(dets, 1.0, atol=ATOL)


def test_p3_no_nan_inf():
    q = _random_unit_quats(100)
    R = quat_to_rotmat(q)
    r6d = rotmat_to_rotate6d(R)
    R2 = rotate6d_to_rotmat(r6d)
    assert not np.any(np.isnan(R2))
    assert not np.any(np.isinf(R2))


# --- P3 edge: near-collinear input for Gram-Schmidt ---

def test_p3_near_collinear_fallback():
    """When a1 ≈ a2, Gram-Schmidt should use fallback and still produce valid SO(3)."""
    r6d = np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float32)  # a1 == a2
    R = rotate6d_to_rotmat(r6d)
    RtR = np.einsum("...ji,...jk->...ik", R, R)
    assert np.allclose(RtR, np.eye(3), atol=ATOL)
    assert np.allclose(np.linalg.det(R), 1.0, atol=ATOL)


def test_p3_near_zero_input():
    """Near-zero r6d is degenerate input; guarantee no NaN/Inf only."""
    r6d = np.array([[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]], dtype=np.float32)
    R = rotate6d_to_rotmat(r6d)
    assert not np.any(np.isnan(R))
    assert not np.any(np.isinf(R))


# --- P4: Column orthogonality ---

def test_p4_column_orthogonality():
    q = _random_unit_quats(50)
    r6d = quat_to_rotate6d(q)
    R = rotate6d_to_rotmat(r6d)

    for i in range(3):
        col_norm = np.linalg.norm(R[..., :, i], axis=-1)
        assert np.allclose(col_norm, 1.0, atol=ATOL)

    # Pairwise orthogonal
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        dots = np.sum(R[..., :, i] * R[..., :, j], axis=-1)
        assert np.allclose(dots, 0.0, atol=ATOL)


# --- P5: Determinant always +1 ---

def test_p5_determinant_positive():
    q = _random_unit_quats(100)
    r6d = quat_to_rotate6d(q)
    R = rotate6d_to_rotmat(r6d)
    dets = np.linalg.det(R)
    assert np.all(dets > 0)
    assert np.allclose(dets, 1.0, atol=ATOL)


# --- Batch dimensions ---

def test_batch_shapes():
    for shape in [(4,), (10, 4), (5, 6, 4)]:
        q = RNG.randn(*shape).astype(np.float32)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        r6d = quat_to_rotate6d(q)
        assert r6d.shape == shape[:-1] + (6,)
        q_rt = rotate6d_to_quat(r6d)
        assert q_rt.shape == shape
        assert np.all(_rotation_distance(q, q_rt) < ATOL)


# --- rotmat_to_quat sign canonicalization ---

def test_rotmat_to_quat_w_nonnegative():
    q = _random_unit_quats(100)
    R = quat_to_rotmat(q)
    q_out = rotmat_to_quat(R)
    assert np.all(q_out[..., 0] >= -ATOL)
