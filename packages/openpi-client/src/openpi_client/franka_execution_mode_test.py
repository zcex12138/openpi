import sys
import types

import pytest


class _StubAffine:
    def __init__(self, *args, **kwargs) -> None:
        pass


sys.modules.setdefault(
    "frankx",
    types.SimpleNamespace(
        Affine=_StubAffine,
        ImpedanceMotion=object,
        JointMotion=object,
        MotionData=object,
        Robot=object,
        Waypoint=object,
        WaypointMotion=object,
    ),
)
sys.modules.setdefault("cv2", types.SimpleNamespace(destroyAllWindows=lambda: None))

from examples.franka import main as _franka_main  # noqa: E402
from examples.franka import real_env as _real_env  # noqa: E402


def _config(**overrides):
    values = {
        "execution_mode": None,
        "policy_default_mode": "service",
        "policy_remote_host": "localhost",
        "policy_remote_port": 8000,
        "rtc_enabled": False,
        "rtc_inference_delay": 3,
        "rtc_execute_horizon": 5,
        "cr_dagger_execute_horizon": 10,
        "cr_dagger_max_skip_steps": 2,
        "control_fps": 30.0,
    }
    values.update(overrides)
    return _real_env.RealEnvConfig(**values)


def test_legacy_rtc_is_normalized_when_execution_mode_is_unset() -> None:
    args = _franka_main.Args()
    settings = _franka_main._resolve_execution_settings(args, _config(rtc_enabled=True), action_horizon=30)

    assert settings.mode == "rtc"
    assert settings.rtc_execute_horizon == 5


def test_explicit_mode_conflicts_with_legacy_rtc_inputs() -> None:
    args = _franka_main.Args(execution_mode="cr_dagger_baseline")

    with pytest.raises(ValueError, match="conflicts with legacy RTC settings"):
        _franka_main._resolve_execution_settings(args, _config(rtc_enabled=True), action_horizon=30)


def test_cr_dagger_horizon_validation_fails_fast() -> None:
    args = _franka_main.Args(
        execution_mode="cr_dagger_baseline",
        cr_dagger_execute_horizon=12,
    )

    with pytest.raises(ValueError, match="exceeds the known model action horizon"):
        _franka_main._resolve_execution_settings(args, _config(), action_horizon=10)


def test_service_mode_defaults_from_config_when_no_cli_policy_args() -> None:
    args = _franka_main.Args()

    settings = _franka_main._resolve_policy_settings(args, _config())

    assert settings.mode == "service"
    assert settings.remote_host == "localhost"
    assert settings.remote_port == 8000


def test_explicit_local_checkpoint_overrides_service_default() -> None:
    args = _franka_main.Args(
        checkpoint_dir="/tmp/checkpoint",
        config="pi05_franka_cola_relative_r6d_lora",
    )

    settings = _franka_main._resolve_policy_settings(args, _config())

    assert settings.mode == "local"
    assert settings.checkpoint_dir == "/tmp/checkpoint"
    assert settings.config_name == "pi05_franka_cola_relative_r6d_lora"


def test_mixing_local_and_remote_policy_args_fails_fast() -> None:
    args = _franka_main.Args(
        checkpoint_dir="/tmp/checkpoint",
        config="pi05_franka_cola_relative_r6d_lora",
        remote_host="localhost",
    )

    with pytest.raises(ValueError, match="Cannot specify both local checkpoint arguments and remote service arguments"):
        _franka_main._resolve_policy_settings(args, _config())
