from __future__ import annotations

import sys
import types

import numpy as np

sys.modules.setdefault(
    "frankx",
    types.SimpleNamespace(
        Affine=object,
        ImpedanceMotion=object,
        JointMotion=object,
        MotionData=object,
        Robot=object,
        Waypoint=object,
        WaypointMotion=object,
    ),
)

from examples.franka import env as _env
from examples.franka import real_env as _real_env
from residual_policy.action_repr import pose8_to_pose10


class _DummyRealEnv:
    def __init__(self) -> None:
        self.executed_actions: list[np.ndarray] = []
        self.is_teaching_mode = False
        self.safety_stop_reasons: list[str] = []

    def reset(self) -> None:
        pass

    def get_state(self) -> np.ndarray:
        return np.zeros(14, dtype=np.float32)

    def get_tcp_velocity(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def execute_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).copy()
        self.executed_actions.append(arr)
        return arr

    def enable_teaching_mode(self) -> None:
        self.is_teaching_mode = True

    def disable_teaching_mode(self) -> None:
        self.is_teaching_mode = False

    def safety_stop_control(self, reason: str | None = None) -> None:
        self.safety_stop_reasons.append("" if reason is None else reason)


class _DummyCamera:
    def get_frames_with_markers(self):
        raise RuntimeError("camera disconnected")


def test_apply_action_converts_pose10_to_executable_pose8() -> None:
    real_env = _DummyRealEnv()
    environment = _env.FrankaEnvironment(real_env=real_env, camera=_DummyCamera())

    pose8 = np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    pose10 = pose8_to_pose10(pose8)
    action = {"actions": pose10}

    environment.apply_action(action)

    np.testing.assert_allclose(real_env.executed_actions[0], pose8, atol=1e-6)
    np.testing.assert_allclose(action["actions"], pose10, atol=1e-6)
    np.testing.assert_allclose(action["actions_pose10"], pose10, atol=1e-6)
    np.testing.assert_allclose(action["executed_action"], pose8, atol=1e-6)


def test_real_env_default_config_disables_residual_checkpoint() -> None:
    config = _real_env.RealEnvConfig.from_yaml()

    assert config.residual_checkpoint_dir is None


def test_camera_failure_triggers_safety_stop() -> None:
    real_env = _DummyRealEnv()
    environment = _env.FrankaEnvironment(real_env=real_env, camera=_DummyCamera())

    try:
        environment.get_observation()
    except _env.CameraSafetyStop as exc:
        assert "camera disconnected" in str(exc)
    else:
        raise AssertionError("Expected CameraSafetyStop when camera retrieval fails")

    assert environment.is_episode_complete()
    assert real_env.safety_stop_reasons
    assert "camera disconnected" in real_env.safety_stop_reasons[0]
