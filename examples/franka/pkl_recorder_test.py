from __future__ import annotations

import sys
import types

import numpy as np
from residual_policy.action_repr import pose8_to_pose10

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

from examples.franka.pkl_recorder import EpisodePklRecorder
from examples.franka.pkl_recorder import RecorderConfig


def _observation() -> dict:
    return {
        "__openpi": {
            "control_timestamp": 123.0,
            "episode_step": 0,
            "recording_snapshot": {
                "frames": {},
                "marker3d": {},
                "timestamp_ns": 0,
                "seq": 1,
                "state": np.array(
                    [0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "tcp_velocity": np.zeros(6, dtype=np.float32),
            },
            "is_human_teaching": False,
        }
    }


def test_recorder_prefers_explicit_base_action_when_present(tmp_path) -> None:
    recorder = EpisodePklRecorder(object(), RecorderConfig(record_dir=tmp_path, control_hz=30.0))
    recorder.on_episode_start()

    final_action_pose8 = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    base_action_pose8 = np.array([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    final_action = pose8_to_pose10(final_action_pose8)
    base_action = pose8_to_pose10(base_action_pose8)
    recorder.on_step(
        _observation(),
        {
            "actions": final_action,
            "base_action": base_action,
            "executed_action": final_action_pose8,
        },
    )

    frame = recorder._frames[0]
    np.testing.assert_allclose(frame["base_action"], base_action, atol=1e-6)
    np.testing.assert_allclose(frame["action"], final_action, atol=1e-6)
    np.testing.assert_allclose(frame["executed_action"], final_action_pose8, atol=1e-6)
    assert "raw_action" not in frame
    assert "base_action_pose10" not in frame
    assert "raw_action_pose10" not in frame
    assert "policy_timing" not in frame
    assert "action_deltas" not in frame


def test_recorder_persists_residual_debug_fields_into_policy_steps(tmp_path) -> None:
    recorder = EpisodePklRecorder(object(), RecorderConfig(record_dir=tmp_path, control_hz=30.0))
    recorder.on_episode_start()

    base_action_pose8 = np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    final_action_pose8 = np.array([0.21, -0.08, 0.31, 1.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    base_action = pose8_to_pose10(base_action_pose8)
    final_action = pose8_to_pose10(final_action_pose8)
    residual_action_pose10 = np.array([0.01, 0.02, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    recorder.on_step(
        _observation(),
        {
            "actions": final_action,
            "base_action": base_action,
            "base_action_pose10": base_action,
            "residual_action_pose10": residual_action_pose10,
            "executed_action": final_action_pose8,
            "policy_timing": {"residual_ms": 1.25},
            "__chunk_meta": {
                "mode": "rtc",
                "horizon_id": 3,
                "chunk_idx": 2,
                "chunk_size": 10,
                "requested_execute_horizon": 10,
                "effective_horizon": 10,
                "new_chunk": False,
                "infer_count": 5,
                "infer_ms": 111.0,
                "horizon_start_timestamp": 123.0,
                "planned_timestamp": 123.1,
            },
        },
    )

    frame = recorder._frames[0]
    step = recorder._policy_steps[0]
    assert "residual_action_pose10" not in frame
    assert "policy_timing" not in frame
    np.testing.assert_allclose(step["base_action"], base_action, atol=1e-6)
    np.testing.assert_allclose(step["raw_action"], final_action, atol=1e-6)
    np.testing.assert_allclose(step["raw_action_pose10"], final_action, atol=1e-6)
    np.testing.assert_allclose(step["base_action_pose10"], base_action, atol=1e-6)
    np.testing.assert_allclose(step["residual_action_pose10"], residual_action_pose10, atol=1e-6)
    assert step["policy_timing"]["residual_ms"] == 1.25
