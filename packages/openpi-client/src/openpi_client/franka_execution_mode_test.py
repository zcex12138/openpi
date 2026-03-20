import importlib.machinery
import sys
from pathlib import Path
import types

import pytest


class _StubAffine:
    def __init__(self, *args, **kwargs) -> None:
        pass


_frankx = types.ModuleType("frankx")
_frankx.__spec__ = importlib.machinery.ModuleSpec("frankx", loader=None)
_frankx.Affine = _StubAffine
_frankx.ImpedanceMotion = object
_frankx.JointMotion = object
_frankx.MotionData = object
_frankx.Robot = object
_frankx.Waypoint = object
_frankx.WaypointMotion = object
sys.modules.setdefault("frankx", _frankx)

_cv2 = types.ModuleType("cv2")
_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
_cv2.destroyAllWindows = lambda: None
sys.modules.setdefault("cv2", _cv2)

from examples.franka import main as _franka_main  # noqa: E402
from examples.franka import real_env as _real_env  # noqa: E402
from openpi.training import config as _train_config  # noqa: E402


def _config(**overrides):
    values = {
        "execution_mode": None,
        "policy_default_mode": "service",
        "policy_remote_host": "localhost",
        "policy_remote_port": 8000,
        "residual_checkpoint_dir": None,
        "residual_scale": 1.0,
        "residual_translation_cap_m": None,
        "residual_rotation_cap_rad": None,
        "record_dir": "eval_records",
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


def test_canonical_mode_ignores_legacy_rtc_config_flag() -> None:
    args = _franka_main.Args()
    settings = _franka_main._resolve_execution_settings(
        args,
        _config(execution_mode="cr_dagger_baseline", rtc_enabled=True),
        action_horizon=30,
    )

    assert settings.mode == "cr_dagger_baseline"


def test_explicit_mode_conflicts_with_legacy_rtc_cli_shorthand() -> None:
    args = _franka_main.Args(execution_mode="cr_dagger_baseline", rtc=True)

    with pytest.raises(ValueError, match="legacy `--args.rtc` shorthand"):
        _franka_main._resolve_execution_settings(args, _config(), action_horizon=30)


def test_from_yaml_prefers_canonical_execution_mode_over_legacy_rtc_flag(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "real_env_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  fps: 30.0",
                "execution:",
                "  mode: \"cr_dagger_baseline\"",
                "rtc:",
                "  enabled: true",
                "  inference_delay: 4",
                "  execute_horizon: 10",
                "cr_dagger:",
                "  execute_horizon: 8",
                "  max_skip_steps: 2",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        cfg = _real_env.RealEnvConfig.from_yaml(config_path)

    assert cfg.execution_mode == "cr_dagger_baseline"
    assert cfg.rtc_enabled is False
    assert "Ignoring deprecated rtc.enabled" in caplog.text


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
        config="pi05_franka_cola_relative_lora",
    )

    settings = _franka_main._resolve_policy_settings(args, _config())

    assert settings.mode == "local"
    assert settings.checkpoint_dir == "/tmp/checkpoint"
    assert settings.config_name == "pi05_franka_cola_relative_lora"


def test_legacy_pose8_policy_requires_pose10_wrapper() -> None:
    train_cfg = _train_config.get_config("pi05_franka_cola_relative_pose8_lora")

    assert _franka_main._needs_pose10_wrapper(train_cfg) is True


def test_pose10_policy_does_not_require_pose10_wrapper() -> None:
    train_cfg = _train_config.get_config("pi05_franka_cola_relative_lora")

    assert _franka_main._needs_pose10_wrapper(train_cfg) is False


def test_mixing_local_and_remote_policy_args_fails_fast() -> None:
    args = _franka_main.Args(
        checkpoint_dir="/tmp/checkpoint",
        config="pi05_franka_cola_relative_lora",
        remote_host="localhost",
    )

    with pytest.raises(ValueError, match="Cannot specify both local checkpoint arguments and remote service arguments"):
        _franka_main._resolve_policy_settings(args, _config())


def test_residual_settings_are_disabled_by_default() -> None:
    settings = _franka_main._resolve_residual_settings(_franka_main.Args(), _config())

    assert settings.enabled is False
    assert settings.checkpoint_dir is None
    assert settings.scale == 1.0


def test_residual_scale_defaults_from_config() -> None:
    settings = _franka_main._resolve_residual_settings(_franka_main.Args(), _config(residual_scale=0.25))

    assert settings.scale == 0.25


def test_residual_checkpoint_dir_defaults_from_config(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "best"
    checkpoint_dir.mkdir()

    settings = _franka_main._resolve_residual_settings(
        _franka_main.Args(),
        _config(residual_checkpoint_dir=str(checkpoint_dir)),
    )

    assert settings.enabled is True
    assert settings.checkpoint_dir == str(checkpoint_dir.resolve())


def test_residual_caps_default_from_config() -> None:
    settings = _franka_main._resolve_residual_settings(
        _franka_main.Args(),
        _config(residual_translation_cap_m=0.01, residual_rotation_cap_rad=0.02),
    )

    assert settings.translation_cap_m == 0.01
    assert settings.rotation_cap_rad == 0.02


def test_residual_settings_validate_positive_caps_and_scale() -> None:
    with pytest.raises(ValueError, match="residual_scale"):
        _franka_main._resolve_residual_settings(_franka_main.Args(residual_scale=-0.1), _config())

    with pytest.raises(ValueError, match="residual_translation_cap_m"):
        _franka_main._resolve_residual_settings(_franka_main.Args(residual_translation_cap_m=0.0), _config())


def test_from_yaml_loads_residual_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "real_env_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  fps: 30.0",
                "residual:",
                "  scale: 0.35",
            ]
        ),
        encoding="utf-8",
    )

    cfg = _real_env.RealEnvConfig.from_yaml(config_path)

    assert cfg.residual_scale == 0.35


def test_from_yaml_loads_residual_caps(tmp_path: Path) -> None:
    config_path = tmp_path / "real_env_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  fps: 30.0",
                "residual:",
                "  translation_cap_m: 0.01",
                "  rotation_cap_rad: 0.02",
            ]
        ),
        encoding="utf-8",
    )

    cfg = _real_env.RealEnvConfig.from_yaml(config_path)

    assert cfg.residual_translation_cap_m == 0.01
    assert cfg.residual_rotation_cap_rad == 0.02


def test_from_yaml_loads_residual_checkpoint_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "real_env_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  fps: 30.0",
                "residual:",
                "  checkpoint_dir: /tmp/residual/best",
            ]
        ),
        encoding="utf-8",
    )

    cfg = _real_env.RealEnvConfig.from_yaml(config_path)

    assert cfg.residual_checkpoint_dir == "/tmp/residual/best"


def test_record_dir_defaults_from_config() -> None:
    record_dir = _franka_main._resolve_record_dir(_franka_main.Args(), _config(record_dir="eval_records/custom"))

    assert record_dir == Path("eval_records/custom")


def test_cli_record_dir_overrides_config() -> None:
    record_dir = _franka_main._resolve_record_dir(
        _franka_main.Args(record_dir="eval_records/cli"),
        _config(record_dir="eval_records/config"),
    )

    assert record_dir == Path("eval_records/cli")


def test_from_yaml_loads_record_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "real_env_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "control:",
                "  fps: 30.0",
                "evaluation:",
                "  record_dir: eval_records/custom",
            ]
        ),
        encoding="utf-8",
    )

    cfg = _real_env.RealEnvConfig.from_yaml(config_path)

    assert cfg.record_dir == "eval_records/custom"
