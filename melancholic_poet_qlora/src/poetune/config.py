from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a dictionary.")
    data["_config_path"] = str(cfg_path)
    return data


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = cfg
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must look like key=value, got: {raw}")
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = copy.deepcopy(cfg)
    for raw in overrides or []:
        key, value = parse_override(raw)
        _set_nested(merged, key, value)
    return merged


def resolve_project_path(value: str | None, project_root: str | Path) -> str | None:
    if value is None:
        return None
    if "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(project_root) / path).resolve())


def infer_project_root(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve().parent.parent
