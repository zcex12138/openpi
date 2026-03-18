from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def extract_config_arg(argv: Sequence[str], *, flag: str = "--config") -> tuple[str | None, list[str]]:
    config_path: str | None = None
    remaining_args: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == flag:
            if config_path is not None:
                raise ValueError(f"{flag} may only be passed once")
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires a path")
            config_path = argv[index + 1]
            index += 2
            continue
        if arg.startswith(f"{flag}="):
            if config_path is not None:
                raise ValueError(f"{flag} may only be passed once")
            config_path = arg.partition("=")[2]
            if not config_path:
                raise ValueError(f"{flag} requires a path")
            index += 1
            continue
        remaining_args.append(arg)
        index += 1
    return config_path, remaining_args


def require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a YAML mapping")
    return dict(value)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return require_mapping(data, field_name="config file")
