"""Configuration loader for hal-voice.

Loads config.yaml, then lets environment variables override individual values using
the convention HALVOICE_<SECTION>_<KEY> (uppercased, dots/nesting joined by '_').
Access is dict-with-dots: cfg.tts.speed, cfg["tts"]["speed"], or cfg.get("tts.speed").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """A dict that also supports attribute access and dotted .get()."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(val) if isinstance(val, dict) else val

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Cfg(node) if isinstance(node, dict) else node


def _coerce(text: str) -> Any:
    """Turn an env-var string into the most plausible scalar."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _apply_env_overrides(cfg: dict) -> None:
    """HALVOICE_TTS_SPEED=0.9 -> cfg['tts']['speed'] = 0.9 (only if the path exists)."""
    prefix = "HALVOICE_"
    skip = {"HALVOICE_ROOT", "HALVOICE_CONFIG"}
    for key, raw in os.environ.items():
        if not key.startswith(prefix) or key in skip:
            continue
        parts = key[len(prefix) :].lower().split("_")
        # Walk as far down the existing tree as the parts allow; the remainder, if any,
        # is treated as a single leaf key (covers leaf names that contain underscores).
        node: Any = cfg
        for i, part in enumerate(parts):
            if isinstance(node, dict) and part in node and isinstance(node[part], dict):
                node = node[part]
            else:
                leaf = "_".join(parts[i:])
                if isinstance(node, dict) and leaf in node:
                    node[leaf] = _coerce(raw)
                break


def _config_path() -> Path:
    env = os.environ.get("HALVOICE_CONFIG")
    if env:
        return Path(env)
    root = os.environ.get("HALVOICE_ROOT")
    if root:
        return Path(root) / "config.yaml"
    return Path(__file__).resolve().parent.parent / "config.yaml"


def root_dir() -> Path:
    """Absolute path to the hal-voice package root (where config.yaml lives)."""
    env = os.environ.get("HALVOICE_ROOT")
    return Path(env) if env else Path(__file__).resolve().parent.parent


def resolve(path_like: str | None) -> str | None:
    """Resolve a (possibly relative) config path against the hal-voice root."""
    if not path_like:
        return path_like
    p = Path(path_like).expanduser()
    return str(p if p.is_absolute() else root_dir() / p)


_CACHE: Cfg | None = None


def load(force: bool = False) -> Cfg:
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    path = _config_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        data = {}
    _apply_env_overrides(data)
    _CACHE = Cfg(data)
    return _CACHE


if __name__ == "__main__":  # `python -m halvoice.config` prints the resolved config
    import json

    print(json.dumps(load(), indent=2, default=str))
