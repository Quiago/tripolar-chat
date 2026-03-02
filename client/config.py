"""Manages the local ~/.fmind/config.yaml file."""

from pathlib import Path
from typing import Optional

import yaml

_CONFIG_DIR = Path.home() / ".fmind"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"


def _load() -> dict:
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w") as f:
        yaml.safe_dump(data, f)


def get_server_url() -> Optional[str]:
    return _load().get("server_url")


def get_api_key() -> Optional[str]:
    return _load().get("api_key")


def get_default_model() -> Optional[str]:
    return _load().get("default_model")


def save_credentials(server_url: str, api_key: str) -> None:
    data = _load()
    data["server_url"] = server_url.rstrip("/")
    data["api_key"] = api_key
    _save(data)


def clear_credentials() -> None:
    data = _load()
    data.pop("server_url", None)
    data.pop("api_key", None)
    _save(data)


def save_default_model(model: str) -> None:
    data = _load()
    data["default_model"] = model
    _save(data)
