import numpy as np

from residual_policy.action_repr import build_input_features
from residual_policy.action_repr import decode_residual_pose10
from residual_policy.action_repr import encode_residual_action
from residual_policy.action_repr import pose8_to_pose10
from residual_policy.action_repr import pose10_to_pose8


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    sign = np.where(quat[..., :1] < 0, -1.0, 1.0)
    return quat * sign


def _pose(xyz: tuple[float, float, float], quat: tuple[float, float, float, float], gripper: float) -> np.ndarray:
    return np.array([*xyz, *quat, gripper], dtype=np.float32)


def test_build_input_features_shape():
    state = pose8_to_pose10(_pose((0.0, 0.1, 0.2), (1.0, 0.0, 0.0, 0.0), 0.3))
    base = pose8_to_pose10(_pose((0.2, 0.1, 0.0), (0.0, 1.0, 0.0, 0.0), 0.4))
    features = build_input_features(state, base)
    assert features.shape == (20,)


def test_encode_decode_roundtrip():
    base = pose8_to_pose10(_pose((0.2, 0.1, 0.3), (0.9238795, 0.0, 0.3826834, 0.0), 0.2))
    corrected = pose8_to_pose10(_pose((0.25, 0.05, 0.35), (0.8660254, 0.0, 0.5, 0.0), 0.6))
    residual = encode_residual_action(base, corrected)
    reconstructed_pose10 = decode_residual_pose10(base, residual)
    reconstructed = pose10_to_pose8(reconstructed_pose10)
    corrected_pose8 = pose10_to_pose8(corrected)

    assert reconstructed_pose10.shape == (10,)
    assert np.allclose(reconstructed[:3], corrected_pose8[:3], atol=1e-5)
    assert np.allclose(reconstructed[7], corrected_pose8[7], atol=1e-5)
    assert np.allclose(_normalize_quat(reconstructed[3:7]), _normalize_quat(corrected_pose8[3:7]), atol=1e-5)


def test_quaternion_sign_canonicalization_removes_fake_residual():
    base_pose8 = _pose((0.0, 0.0, 0.0), (0.70710677, 0.0, 0.70710677, 0.0), 0.1)
    corrected_pose8 = _pose((0.0, 0.0, 0.0), (-0.70710677, -0.0, -0.70710677, -0.0), 0.1)
    base = pose8_to_pose10(base_pose8)
    corrected = pose8_to_pose10(corrected_pose8)
    residual = encode_residual_action(base, corrected)
    assert np.allclose(residual[:3], 0.0, atol=1e-6)
    assert np.allclose(residual[9], 0.0, atol=1e-6)
    reconstructed = pose10_to_pose8(decode_residual_pose10(base, residual))
    assert np.allclose(_normalize_quat(reconstructed[3:7]), _normalize_quat(base_pose8[3:7]), atol=1e-5)
